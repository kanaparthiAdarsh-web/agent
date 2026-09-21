"""Evidence extraction with provenance tracking and verification."""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel

from ..schemas import (
    SourceClaim, Evidence, EvidenceType, VerificationStatus, 
    EntailmentCheck, ChunkData, PaperMetadata
)
from ..exceptions import EvidenceError, VerificationError
from ..llm.client_factory import invoke_with_structured_output
from ..llm.config import get_llm_config

logger = logging.getLogger(__name__)


EVIDENCE_EXTRACTION_PROMPT = """You are an expert research analyst. Extract evidence claims from the following paper text.

For each claim, identify:
1. The type of claim (claim, result, methodology, dataset, metric, limitation, contribution, future_work, definition)
2. The content of the claim
3. An exact quote from the source if possible
4. Your confidence in the extraction (0.0 to 1.0)

Return your response as a JSON array of objects with these fields:
- claim_type: string - One of: claim, result, methodology, dataset, metric, limitation, contribution, future_work, definition
- content: string - The claim content in your own words
- exact_quote: string or null - Exact quote from the text supporting this claim
- confidence: float - Confidence in extraction accuracy (0.0 to 1.0)

IMPORTANT RULES:
1. Only extract claims that are explicitly supported by the text
2. Do not fabricate or exaggerate claims
3. For results, include specific numbers/metrics when available
4. For limitations, focus on what authors explicitly acknowledge
5. Include the exact quote whenever possible for verification
6. Be conservative with confidence scores

Text to analyze:
{text}

Research focus: {research_focus}

Return ONLY a valid JSON array of claim objects."""


VERIFICATION_PROMPT = """You are verifying whether a claim is entailed by the source text.

CLAIM: {claim}

SOURCE TEXT: {source_text}

Determine if the source text ENTAILS (logically implies) the claim.

Consider:
- Does the source explicitly state the claim?
- Does the source logically imply the claim?
- Is the claim consistent with the source?
- Could the claim be false while the source is true? (if yes, not entailment)

Return JSON with:
- entails: boolean - Whether the source entails the claim
- confidence: float - Confidence in your judgment (0.0 to 1.0)
- explanation: string - Brief explanation of your reasoning

Return ONLY valid JSON."""


class EvidenceExtractor:
    """Extracts and verifies evidence from paper text."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the evidence extractor.
        
        Args:
            llm: LLM client for extraction (uses default if not provided)
        """
        self.llm = llm
        self._config = None
    
    def _get_llm(self) -> BaseChatModel:
        """Get LLM client, initializing if needed."""
        if self.llm is None:
            from ..llm.client_factory import create_llm_client
            if self._config is None:
                self._config = get_llm_config()
            self.llm = create_llm_client(self._config)
        return self.llm
    
    async def extract_claims(
        self,
        chunks: List[ChunkData],
        paper_metadata: PaperMetadata,
        research_focus: str,
    ) -> List[SourceClaim]:
        """
        Extract claims from paper chunks.
        
        Args:
            chunks: Text chunks from the paper
            paper_metadata: Paper metadata
            research_focus: Research topic/question to focus extraction on
        
        Returns:
            List of extracted SourceClaim objects
        
        Raises:
            EvidenceError: If extraction fails
        """
        try:
            llm = self._get_llm()
            claims = []
            
            # Process each chunk
            for chunk in chunks:
                chunk_claims = await self._extract_from_chunk(
                    chunk=chunk,
                    paper_id=paper_metadata.id,
                    research_focus=research_focus,
                )
                claims.extend(chunk_claims)
            
            logger.info("Extracted %d claims from paper %s", len(claims), paper_metadata.id)
            return claims
            
        except Exception as e:
            logger.error("Failed to extract claims from paper %s: %s", paper_metadata.id, e)
            raise EvidenceError(f"Claim extraction failed: {e}")
    
    async def _extract_from_chunk(
        self,
        chunk: ChunkData,
        paper_id: str,
        research_focus: str,
    ) -> List[SourceClaim]:
        """Extract claims from a single chunk."""
        llm = self._get_llm()
        
        prompt = EVIDENCE_EXTRACTION_PROMPT.format(
            text=chunk.text[:4000],  # Limit chunk text
            research_focus=research_focus,
        )
        
        try:
            result = await invoke_with_structured_output(llm, prompt)
            
            if not isinstance(result, list):
                return []
            
            claims = []
            for i, item in enumerate(result[:10]):  # Limit to 10 claims per chunk
                if not isinstance(item, dict):
                    continue
                
                claim_type_str = item.get("claim_type", "claim")
                try:
                    claim_type = EvidenceType(claim_type_str)
                except ValueError:
                    claim_type = EvidenceType.CLAIM
                
                confidence = item.get("confidence", 0.5)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                confidence = max(0.0, min(1.0, confidence))
                
                claim = SourceClaim(
                    claim_id=f"claim_{paper_id}_{uuid4().hex[:8]}_{i}",
                    paper_id=paper_id,
                    chunk_id=chunk.chunk_id,
                    claim_type=claim_type,
                    content=item.get("content", ""),
                    exact_quote=item.get("exact_quote"),
                    page_numbers=chunk.page_numbers,
                    section=chunk.section_type,
                    confidence=confidence,
                    verification_status=VerificationStatus.UNVERIFIED,
                )
                
                if claim.content.strip():  # Only add non-empty claims
                    claims.append(claim)
            
            return claims
            
        except Exception as e:
            logger.warning("Failed to extract from chunk %s: %s", chunk.chunk_id, e)
            return []
    
    async def verify_claim(
        self,
        claim: SourceClaim,
        source_text: str,
    ) -> Tuple[VerificationStatus, EntailmentCheck]:
        """
        Verify a claim against source text.
        
        Args:
            claim: Claim to verify
            source_text: Source text to verify against
        
        Returns:
            Tuple of (VerificationStatus, EntailmentCheck)
        
        Raises:
            VerificationError: If verification fails
        """
        try:
            llm = self._get_llm()
            
            prompt = VERIFICATION_PROMPT.format(
                claim=claim.content,
                source_text=source_text[:2000],  # Limit source text
            )
            
            result = await invoke_with_structured_output(llm, prompt)
            
            if not isinstance(result, dict):
                raise VerificationError("Invalid verification response")
            
            entails = result.get("entails", False)
            confidence = result.get("confidence", 0.5)
            explanation = result.get("explanation", "")
            
            # Determine verification status
            if entails and confidence >= 0.8:
                status = VerificationStatus.VERIFIED
            elif entails and confidence >= 0.5:
                status = VerificationStatus.PARTIALLY_VERIFIED
            elif not entails and confidence >= 0.7:
                status = VerificationStatus.REJECTED
            else:
                status = VerificationStatus.UNVERIFIED
            
            entailment_check = EntailmentCheck(
                claim_id=claim.claim_id,
                source_text=source_text[:2000],
                entails=entails,
                confidence=confidence,
                explanation=explanation,
            )
            
            return status, entailment_check
            
        except Exception as e:
            logger.error("Failed to verify claim %s: %s", claim.claim_id, e)
            raise VerificationError(f"Claim verification failed: {e}")
    
    async def aggregate_evidence(
        self,
        claims: List[SourceClaim],
        research_topic: str,
        category: str,
    ) -> Evidence:
        """
        Aggregate claims into structured evidence.
        
        Args:
            claims: List of related claims
            research_topic: Research topic this evidence relates to
            category: Evidence category
        
        Returns:
            Aggregated Evidence object
        """
        # Group claims by type
        verified_claims = [c for c in claims if c.verification_status == VerificationStatus.VERIFIED]
        partial_claims = [c for c in claims if c.verification_status == VerificationStatus.PARTIALLY_VERIFIED]
        
        # Calculate quality score
        if claims:
            avg_confidence = sum(c.confidence for c in claims) / len(claims)
            verification_ratio = len(verified_claims) / len(claims)
            quality_score = (avg_confidence + verification_ratio) / 2
        else:
            quality_score = 0.0
        
        # Determine overall status
        if verified_claims:
            overall_status = VerificationStatus.VERIFIED
        elif partial_claims:
            overall_status = VerificationStatus.PARTIALLY_VERIFIED
        else:
            overall_status = VerificationStatus.UNVERIFIED
        
        # Generate summary
        claim_contents = [c.content for c in claims[:5]]
        summary = "; ".join(claim_contents) if claim_contents else "No evidence extracted"
        
        # Get unique sources
        source_papers = set(c.paper_id for c in claims)
        
        evidence = Evidence(
            evidence_id=f"evidence_{uuid4().hex[:12]}",
            research_topic=research_topic,
            claims=claims,
            summary=summary,
            category=category,
            quality_score=min(quality_score, 1.0),
            source_count=len(source_papers),
            verification_status=overall_status,
        )
        
        return evidence


async def extract_evidence(
    chunks: List[ChunkData],
    paper_metadata: PaperMetadata,
    research_focus: str,
    llm: Optional[BaseChatModel] = None,
) -> List[SourceClaim]:
    """
    Convenience function to extract evidence from a paper.
    
    Args:
        chunks: Text chunks from the paper
        paper_metadata: Paper metadata
        research_focus: Research topic/question
        llm: Optional LLM client
    
    Returns:
        List of extracted claims
    """
    extractor = EvidenceExtractor(llm=llm)
    return await extractor.extract_claims(chunks, paper_metadata, research_focus)


async def verify_evidence(
    claim: SourceClaim,
    source_text: str,
    llm: Optional[BaseChatModel] = None,
) -> Tuple[VerificationStatus, EntailmentCheck]:
    """
    Convenience function to verify a claim.
    
    Args:
        claim: Claim to verify
        source_text: Source text
        llm: Optional LLM client
    
    Returns:
        Tuple of (status, entailment check)
    """
    extractor = EvidenceExtractor(llm=llm)
    return await extractor.verify_claim(claim, source_text)

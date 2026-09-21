"""Evidence verification with entailment checking."""

import logging
from typing import Optional, Tuple, List
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel

from ..schemas import (
    SourceClaim, Evidence, VerificationStatus, 
    EntailmentCheck, ChunkData
)
from ..exceptions import VerificationError
from ..llm.client_factory import invoke_with_structured_output
from ..llm.config import get_llm_config

logger = logging.getLogger(__name__)


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


class EvidenceVerifier:
    """Verifies evidence claims against source text."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the evidence verifier.
        
        Args:
            llm: LLM client for verification (uses default if not provided)
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
    
    async def verify_multiple_claims(
        self,
        claims: List[SourceClaim],
        chunk_map: dict,
    ) -> List[Tuple[SourceClaim, VerificationStatus, Optional[EntailmentCheck]]]:
        """
        Verify multiple claims efficiently.
        
        Args:
            claims: Claims to verify
            chunk_map: Mapping of chunk_id to chunk text
        
        Returns:
            List of tuples (claim, status, entailment_check)
        """
        results = []
        
        for claim in claims:
            try:
                # Get source text from chunk map
                source_text = chunk_map.get(claim.chunk_id, "")
                
                if not source_text:
                    # No source text available
                    results.append((
                        claim,
                        VerificationStatus.UNVERIFIED,
                        None
                    ))
                    continue
                
                status, entailment = await self.verify_claim(claim, source_text)
                
                # Update claim with verification status
                claim.verification_status = status
                
                results.append((claim, status, entailment))
                
            except Exception as e:
                logger.warning("Failed to verify claim %s: %s", claim.claim_id, e)
                results.append((
                    claim,
                    VerificationStatus.UNVERIFIED,
                    None
                ))
        
        return results
    
    def calculate_evidence_quality(
        self,
        verified_claims: List[SourceClaim],
    ) -> float:
        """
        Calculate overall quality score for evidence.
        
        Args:
            verified_claims: List of verified claims
        
        Returns:
            Quality score 0.0-1.0
        """
        if not verified_claims:
            return 0.0
        
        # Count by verification status
        verified_count = sum(
            1 for c in verified_claims 
            if c.verification_status == VerificationStatus.VERIFIED
        )
        partial_count = sum(
            1 for c in verified_claims 
            if c.verification_status == VerificationStatus.PARTIALLY_VERIFIED
        )
        
        # Calculate weighted score
        total = len(verified_claims)
        score = (verified_count * 1.0 + partial_count * 0.5) / total
        
        return min(score, 1.0)


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
    verifier = EvidenceVerifier(llm=llm)
    return await verifier.verify_claim(claim, source_text)

"""Research gap generation, verification, and counterevidence search."""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel

from ..schemas import (
    GapCandidate, GapVerification, GapCoverageStatus, GapCategory,
    Counterevidence, ResearchDirection, GapAnalysisResult,
    LimitationCluster, PaperCard, SourceClaim
)
from ..exceptions import GapAnalysisError
from ..llm.client_factory import invoke_with_structured_output
from ..llm.config import get_llm_config

logger = logging.getLogger(__name__)


GAP_GENERATION_PROMPT = """Identify potential research gaps based on analyzed limitations.

Given the limitation clusters and paper analysis, identify patterns that suggest
potential research gaps. These are areas where the literature shows recurring
limitations or unaddressed concerns.

IMPORTANT: 
- Do NOT claim absolute novelty or that something is "unexplored"
- Use cautious language like "limited evidence", "potentially underexplored"
- Focus on patterns in the analyzed papers, not definitive claims

LIMITATION CLUSTERS:
{limitation_clusters}

PAPER SUMMARY:
{paper_summary}

For each potential gap candidate, provide:
- description: Clear description of the potential gap
- category: One of: methodological, empirical, theoretical, dataset, evaluation, application, reproducibility, scalability, interpretability, other
- supporting_evidence: Array of evidence snippets from the analysis
- affected_methods: Array of methods/models affected
- affected_datasets: Array of datasets affected
- initial_confidence: Float 0.0-1.0 indicating initial confidence

Return JSON array of gap candidate objects.

Return ONLY valid JSON array."""


VERIFICATION_PROMPT = """Verify a potential research gap against the available literature.

GAP CANDIDATE: {gap_description}

SUPPORTING EVIDENCE: {supporting_evidence}

COUNTEREVIDENCE FOUND: {counterevidence}

Analyze the literature coverage for this potential gap. Consider:
1. How many papers in our analysis address this area?
2. Is there counterevidence that partially or fully addresses the gap?
3. What is the quality/depth of existing work?

Determine coverage status as one of:
- limited_evidence: Few papers address this, or treatment is superficial
- partially_addressed: Some work exists but significant aspects remain open
- substantially_addressed: Most aspects have been addressed in literature
- insufficient_literature: Not enough papers in our analysis to determine

Return JSON with:
- coverage_status: One of the four statuses above
- coverage_summary: String summarizing the literature coverage
- reasoning: Detailed reasoning referencing specific papers/evidence
- assessment_confidence: Float 0.0-1.0

CRITICAL: Never claim absolute novelty. Use phrases like:
- "Limited evidence in the analyzed literature"
- "Partially addressed in current work"
- "Few papers in our sample address this"

Return ONLY valid JSON."""


class GapAnalyzer:
    """Analyzes limitations to generate and verify potential research gaps."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the gap analyzer.
        
        Args:
            llm: LLM client for analysis (uses default if not provided)
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
    
    async def generate_gap_candidates(
        self,
        limitation_clusters: List[LimitationCluster],
        paper_cards: List[PaperCard],
        research_topic: str,
    ) -> List[GapCandidate]:
        """
        Generate potential gap candidates from limitation analysis.
        
        Args:
            limitation_clusters: Clustered limitations
            paper_cards: Analyzed paper cards
            research_topic: Research topic being analyzed
        
        Returns:
            List of GapCandidate objects
        
        Raises:
            GapAnalysisError: If generation fails
        """
        try:
            llm = self._get_llm()
            
            # Format limitation clusters
            clusters_text = []
            for cluster in limitation_clusters[:10]:  # Limit to top clusters
                clusters_text.append(
                    f"Cluster: {cluster.theme}\n"
                    f"Category: {cluster.category.value}\n"
                    f"Description: {cluster.description}\n"
                    f"Papers: {', '.join(cluster.source_papers)}\n"
                    f"Frequency: {cluster.frequency}\n"
                )
            
            # Format paper summary
            paper_summary = f"Analyzed {len(paper_cards)} papers on {research_topic}.\n"
            methods = set()
            datasets = set()
            for pc in paper_cards:
                methods.update(pc.models)
                datasets.update(pc.datasets)
            paper_summary += f"Methods studied: {', '.join(list(methods)[:10])}\n"
            paper_summary += f"Datasets used: {', '.join(list(datasets)[:10])}"
            
            prompt = GAP_GENERATION_PROMPT.format(
                limitation_clusters="\n---\n".join(clusters_text),
                paper_summary=paper_summary,
            )
            
            result = await invoke_with_structured_output(llm, prompt)
            
            if not isinstance(result, list):
                return []
            
            candidates = []
            for i, item in enumerate(result[:8]):  # Limit to 8 candidates
                if not isinstance(item, dict):
                    continue
                
                category_str = item.get("category", "other")
                try:
                    category = GapCategory(category_str)
                except ValueError:
                    category = GapCategory.OTHER
                
                confidence = item.get("initial_confidence", 0.5)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                confidence = max(0.0, min(1.0, confidence))
                
                # Extract supporting papers from related clusters
                supporting_papers = []
                for cluster in limitation_clusters:
                    if cluster.theme.lower() in item.get("description", "").lower():
                        supporting_papers.extend(cluster.source_papers)
                supporting_papers = list(set(supporting_papers))[:5]
                
                candidate = GapCandidate(
                    gap_id=f"gap_{uuid4().hex[:8]}_{i}",
                    research_topic=research_topic,
                    description=item.get("description", ""),
                    category=category,
                    supporting_papers=supporting_papers,
                    supporting_evidence=item.get("supporting_evidence", []) or [],
                    related_limitation_clusters=[
                        c.cluster_id for c in limitation_clusters[:5]
                    ],
                    affected_methods=item.get("affected_methods", []) or [],
                    affected_datasets=item.get("affected_datasets", []) or [],
                    initial_confidence=confidence,
                )
                
                if candidate.description.strip():
                    candidates.append(candidate)
            
            logger.info("Generated %d gap candidates", len(candidates))
            return candidates
            
        except Exception as e:
            logger.error("Failed to generate gap candidates: %s", e)
            raise GapAnalysisError(f"Gap generation failed: {e}")
    
    async def search_counterevidence(
        self,
        gap_candidate: GapCandidate,
        all_claims: List[SourceClaim],
        paper_cards: List[PaperCard],
    ) -> List[Counterevidence]:
        """
        Search for counterevidence that might address the gap.
        
        This uses simple keyword matching and claim analysis.
        In production, this would use semantic retrieval.
        
        Args:
            gap_candidate: The gap candidate to check
            all_claims: All extracted claims from papers
            paper_cards: All analyzed paper cards
        
        Returns:
            List of Counterevidence objects
        """
        counterevidence = []
        gap_lower = gap_candidate.description.lower()
        
        # Look for claims that might address the gap
        for claim in all_claims:
            if claim.claim_type not in ["contribution", "result", "methodology"]:
                continue
            
            claim_lower = claim.content.lower()
            
            # Check if claim addresses any of the gap's affected areas
            addresses_gap = False
            
            # Check method overlap
            for method in gap_candidate.affected_methods:
                if method.lower() in claim_lower:
                    addresses_gap = True
                    break
            
            # Check if claim suggests solution to limitation
            if any(kw in claim_lower for kw in ["overcome", "address", "solve", "improve", "enhance"]):
                if any(kw in gap_lower for kw in ["limitation", "challenge", "issue", "problem"]):
                    addresses_gap = True
            
            if addresses_gap:
                # Calculate relevance score
                relevance = 0.5
                if claim.confidence > 0.7:
                    relevance += 0.2
                if claim.verification_status.value in ["verified", "partially_verified"]:
                    relevance += 0.2
                
                ce = Counterevidence(
                    counterevidence_id=f"ce_{uuid4().hex[:8]}",
                    gap_candidate_id=gap_candidate.gap_id,
                    paper_id=claim.paper_id,
                    chunk_id=claim.chunk_id,
                    description=f"Paper claims: {claim.content[:200]}",
                    exact_quote=claim.exact_quote,
                    relevance_score=min(relevance, 1.0),
                    strength="weak" if relevance < 0.6 else "moderate" if relevance < 0.8 else "strong",
                )
                counterevidence.append(ce)
        
        return counterevidence[:5]  # Limit to 5 counterevidence items
    
    async def verify_gap(
        self,
        gap_candidate: GapCandidate,
        counterevidence: List[Counterevidence],
    ) -> GapVerification:
        """
        Verify a gap candidate and determine coverage status.
        
        Args:
            gap_candidate: Gap candidate to verify
            counterevidence: Counterevidence found
        
        Returns:
            GapVerification object
        
        Raises:
            GapAnalysisError: If verification fails
        """
        try:
            llm = self._get_llm()
            
            # Format evidence
            evidence_text = "; ".join(gap_candidate.supporting_evidence[:5])
            ce_text = "; ".join([ce.description for ce in counterevidence]) if counterevidence else "No counterevidence found"
            
            prompt = VERIFICATION_PROMPT.format(
                gap_description=gap_candidate.description,
                supporting_evidence=evidence_text,
                counterevidence=ce_text,
            )
            
            result = await invoke_with_structured_output(llm, prompt)
            
            if not isinstance(result, dict):
                # Fallback to heuristic-based verification
                return self._heuristic_verification(gap_candidate, counterevidence)
            
            coverage_str = result.get("coverage_status", "limited_evidence")
            try:
                coverage_status = GapCoverageStatus(coverage_str)
            except ValueError:
                coverage_status = GapCoverageStatus.LIMITED_EVIDENCE
            
            confidence = result.get("assessment_confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            
            verification = GapVerification(
                verification_id=f"ver_{uuid4().hex[:12]}",
                gap_candidate_id=gap_candidate.gap_id,
                coverage_status=coverage_status,
                supporting_evidence_count=len(gap_candidate.supporting_evidence),
                counterevidence=counterevidence,
                coverage_summary=result.get("coverage_summary", ""),
                reasoning=result.get("reasoning", ""),
                assessment_confidence=min(confidence, 1.0),
            )
            
            return verification
            
        except Exception as e:
            logger.error("Failed to verify gap %s: %s", gap_candidate.gap_id, e)
            return self._heuristic_verification(gap_candidate, counterevidence)
    
    def _heuristic_verification(
        self,
        gap_candidate: GapCandidate,
        counterevidence: List[Counterevidence],
    ) -> GapVerification:
        """Fallback heuristic-based verification when LLM fails."""
        # Simple heuristics based on counterevidence
        strong_ce = sum(1 for ce in counterevidence if ce.strength == "strong")
        moderate_ce = sum(1 for ce in counterevidence if ce.strength == "moderate")
        
        if strong_ce >= 2:
            status = GapCoverageStatus.SUBSTANTIALLY_ADDRESSED
        elif strong_ce >= 1 or moderate_ce >= 2:
            status = GapCoverageStatus.PARTIALLY_ADDRESSED
        elif len(counterevidence) > 0:
            status = GapCoverageStatus.LIMITED_EVIDENCE
        else:
            status = GapCoverageStatus.INSUFFICIENT_LITERATURE
        
        return GapVerification(
            verification_id=f"ver_{uuid4().hex[:12]}",
            gap_candidate_id=gap_candidate.gap_id,
            coverage_status=status,
            supporting_evidence_count=len(gap_candidate.supporting_evidence),
            counterevidence=counterevidence,
            coverage_summary=f"Heuristic assessment: {status.value}. Found {len(counterevidence)} counterevidence items.",
            reasoning=f"Based on {strong_ce} strong and {moderate_ce} moderate counterevidence items.",
            assessment_confidence=0.5,  # Lower confidence for heuristic
        )
    
    async def analyze_gaps(
        self,
        limitation_clusters: List[LimitationCluster],
        paper_cards: List[PaperCard],
        all_claims: List[SourceClaim],
        research_topic: str,
    ) -> GapAnalysisResult:
        """
        Run complete gap analysis pipeline.
        
        Args:
            limitation_clusters: Clustered limitations
            paper_cards: Analyzed paper cards
            all_claims: All extracted claims
            research_topic: Research topic
        
        Returns:
            Complete GapAnalysisResult
        """
        # Generate candidates
        candidates = await self.generate_gap_candidates(
            limitation_clusters, paper_cards, research_topic
        )
        
        # Verify each candidate
        verifications = []
        for candidate in candidates:
            counterevidence = await self.search_counterevidence(
                candidate, all_claims, paper_cards
            )
            verification = await self.verify_gap(candidate, counterevidence)
            verifications.append(verification)
        
        # Calculate statistics
        by_coverage: Dict[str, int] = {}
        for v in verifications:
            status = v.coverage_status.value
            by_coverage[status] = by_coverage.get(status, 0) + 1
        
        # Generate executive summary
        summary_parts = [
            f"Gap analysis identified {len(candidates)} potential research gaps.",
        ]
        if by_coverage:
            most_common = max(by_coverage.items(), key=lambda x: x[1])
            summary_parts.append(
                f"Most gaps ({most_common[1]}) show {most_common[0].replace('_', ' ')}."
            )
        
        result = GapAnalysisResult(
            analysis_id=f"analysis_{uuid4().hex[:12]}",
            research_topic=research_topic,
            gap_candidates=candidates,
            verifications=verifications,
            research_directions=[],  # Will be populated by directions generator
            total_candidates=len(candidates),
            by_coverage_status=by_coverage,
            executive_summary=" ".join(summary_parts),
        )
        
        return result


async def analyze_gaps(
    limitation_clusters: List[LimitationCluster],
    paper_cards: List[PaperCard],
    all_claims: List[SourceClaim],
    research_topic: str,
    llm: Optional[BaseChatModel] = None,
) -> GapAnalysisResult:
    """
    Convenience function for complete gap analysis.
    
    Args:
        limitation_clusters: Clustered limitations
        paper_cards: Analyzed paper cards
        all_claims: All extracted claims
        research_topic: Research topic
        llm: Optional LLM client
    
    Returns:
        GapAnalysisResult
    """
    analyzer = GapAnalyzer(llm=llm)
    return await analyzer.analyze_gaps(
        limitation_clusters, paper_cards, all_claims, research_topic
    )

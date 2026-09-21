"""Research direction generation based on gap analysis."""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel

from ..schemas import (
    ResearchDirection, GapCandidate, GapVerification, GapAnalysisResult,
    PaperCard, LimitationCluster
)
from ..exceptions import GapAnalysisError
from ..llm.client_factory import invoke_with_structured_output
from ..llm.config import get_llm_config

logger = logging.getLogger(__name__)


DIRECTION_GENERATION_PROMPT = """Generate evidence-grounded research directions based on gap analysis.

Given the identified potential gaps and supporting evidence, suggest concrete
research directions that could address these gaps.

IMPORTANT:
- Clearly distinguish between documented evidence and proposed suggestions
- Do NOT claim the directions are definitely novel
- Ground each direction in specific evidence from the analysis
- Be practical about feasibility

GAP ANALYSIS:
{gap_analysis}

LIMITATION CLUSTERS:
{limitation_clusters}

PAPER INSIGHTS:
{paper_insights}

For each research direction, provide:
- proposed_problem: Clear statement of the research problem to investigate
- motivation: Why this direction is worth pursuing (grounded in evidence)
- suggested_methodology: Suggested approach/methodology
- possible_datasets: Datasets that could be used
- candidate_models: Models/architectures to consider
- evaluation_strategy: How to evaluate the proposed work
- evaluation_metrics: Specific metrics to use
- feasibility_considerations: Practical considerations
- required_resources: Compute, data, or other resources needed
- assumptions: Key assumptions underlying this direction
- supporting_evidence: Evidence from analysis supporting this direction
- evidence_paper_ids: IDs of papers providing evidence
- related_gap_ids: IDs of gap candidates this addresses
- priority: "low", "medium", or "high"
- novelty_potential: "uncertain", "potential", or "plausible" (NEVER "definite")

Return JSON array of direction objects.

Return ONLY valid JSON array."""


class DirectionGenerator:
    """Generates evidence-grounded research directions."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the direction generator.
        
        Args:
            llm: LLM client for generation (uses default if not provided)
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
    
    async def generate_directions(
        self,
        gap_analysis: GapAnalysisResult,
        limitation_clusters: List[LimitationCluster],
        paper_cards: List[PaperCard],
    ) -> List[ResearchDirection]:
        """
        Generate research directions based on gap analysis.
        
        Args:
            gap_analysis: Complete gap analysis result
            limitation_clusters: Clustered limitations
            paper_cards: Analyzed paper cards
        
        Returns:
            List of ResearchDirection objects
        
        Raises:
            GapAnalysisError: If generation fails
        """
        try:
            llm = self._get_llm()
            
            # Format gap analysis summary
            gap_summary = []
            for i, (candidate, verification) in enumerate(
                zip(gap_analysis.gap_candidates[:5], gap_analysis.verifications[:5])
            ):
                gap_summary.append(
                    f"Gap {i+1}: {candidate.description}\n"
                    f"Category: {candidate.category.value}\n"
                    f"Coverage: {verification.coverage_status.value}\n"
                    f"Evidence: {'; '.join(candidate.supporting_evidence[:2])}\n"
                )
            
            # Format limitation clusters
            cluster_summary = []
            for cluster in limitation_clusters[:5]:
                cluster_summary.append(
                    f"- {cluster.theme} ({cluster.category.value}): "
                    f"{cluster.description[:100]}"
                )
            
            # Extract paper insights
            insights = []
            methods = set()
            datasets = set()
            for pc in paper_cards:
                methods.update(pc.models)
                datasets.update(pc.datasets)
                if pc.key_results:
                    insights.append(f"Result: {pc.key_results[0][:100]}")
            
            paper_insights = "\n".join([
                f"Methods studied: {', '.join(list(methods)[:8])}",
                f"Datasets used: {', '.join(list(datasets)[:8])}",
            ] + insights[:5])
            
            prompt = DIRECTION_GENERATION_PROMPT.format(
                gap_analysis="\n---\n".join(gap_summary),
                limitation_clusters="\n".join(cluster_summary),
                paper_insights=paper_insights,
            )
            
            result = await invoke_with_structured_output(llm, prompt)
            
            if not isinstance(result, list):
                return []
            
            directions = []
            for i, item in enumerate(result[:6]):  # Limit to 6 directions
                if not isinstance(item, dict):
                    continue
                
                # Validate and normalize fields
                priority = item.get("priority", "medium")
                if priority not in ["low", "medium", "high"]:
                    priority = "medium"
                
                novelty_potential = item.get("novelty_potential", "uncertain")
                if novelty_potential not in ["uncertain", "potential", "plausible"]:
                    novelty_potential = "uncertain"
                
                # Collect evidence paper IDs
                evidence_papers = item.get("evidence_paper_ids", []) or []
                if not evidence_papers and item.get("supporting_evidence"):
                    # Try to extract from gap candidates
                    for candidate in gap_analysis.gap_candidates:
                        if any(
                            ev in str(item.get("supporting_evidence", []))
                            for ev in candidate.supporting_evidence
                        ):
                            evidence_papers.extend(candidate.supporting_papers)
                evidence_papers = list(set(evidence_papers))[:5]
                
                # Collect related gap IDs
                related_gaps = item.get("related_gap_ids", []) or []
                if not related_gaps:
                    # Link to first few gaps by default
                    related_gaps = [g.gap_id for g in gap_analysis.gap_candidates[:3]]
                
                direction = ResearchDirection(
                    direction_id=f"dir_{uuid4().hex[:8]}_{i}",
                    research_topic=gap_analysis.research_topic,
                    proposed_problem=item.get("proposed_problem", ""),
                    motivation=item.get("motivation", ""),
                    suggested_methodology=item.get("suggested_methodology"),
                    possible_datasets=item.get("possible_datasets", []) or [],
                    candidate_models=item.get("candidate_models", []) or [],
                    evaluation_strategy=item.get("evaluation_strategy"),
                    evaluation_metrics=item.get("evaluation_metrics", []) or [],
                    feasibility_considerations=item.get("feasibility_considerations", []) or [],
                    required_resources=item.get("required_resources", []) or [],
                    assumptions=item.get("assumptions", []) or [],
                    supporting_evidence=item.get("supporting_evidence", []) or [],
                    evidence_paper_ids=evidence_papers,
                    related_gap_ids=related_gaps,
                    priority=priority,
                    novelty_potential=novelty_potential,
                )
                
                if direction.proposed_problem.strip() and direction.motivation.strip():
                    directions.append(direction)
            
            logger.info("Generated %d research directions", len(directions))
            return directions
            
        except Exception as e:
            logger.error("Failed to generate research directions: %s", e)
            raise GapAnalysisError(f"Direction generation failed: {e}")
    
    async def complete_gap_analysis(
        self,
        gap_analysis: GapAnalysisResult,
        limitation_clusters: List[LimitationCluster],
        paper_cards: List[PaperCard],
    ) -> GapAnalysisResult:
        """
        Complete a gap analysis by adding research directions.
        
        Args:
            gap_analysis: Gap analysis result (without directions)
            limitation_clusters: Clustered limitations
            paper_cards: Analyzed paper cards
        
        Returns:
            Complete GapAnalysisResult with directions
        """
        directions = await self.generate_directions(
            gap_analysis, limitation_clusters, paper_cards
        )
        
        # Update the result with directions
        gap_analysis.research_directions = directions
        
        # Update executive summary
        summary_parts = [gap_analysis.executive_summary]
        if directions:
            high_priority = sum(1 for d in directions if d.priority == "high")
            summary_parts.append(
                f"Suggested {len(directions)} research directions "
                f"({high_priority} high priority)."
            )
        gap_analysis.executive_summary = " ".join(summary_parts)
        
        return gap_analysis


async def generate_research_directions(
    gap_analysis: GapAnalysisResult,
    limitation_clusters: List[LimitationCluster],
    paper_cards: List[PaperCard],
    llm: Optional[BaseChatModel] = None,
) -> List[ResearchDirection]:
    """
    Convenience function to generate research directions.
    
    Args:
        gap_analysis: Gap analysis result
        limitation_clusters: Clustered limitations
        paper_cards: Analyzed paper cards
        llm: Optional LLM client
    
    Returns:
        List of ResearchDirection objects
    """
    generator = DirectionGenerator(llm=llm)
    return await generator.generate_directions(
        gap_analysis, limitation_clusters, paper_cards
    )


async def complete_gap_analysis(
    gap_analysis: GapAnalysisResult,
    limitation_clusters: List[LimitationCluster],
    paper_cards: List[PaperCard],
    llm: Optional[BaseChatModel] = None,
) -> GapAnalysisResult:
    """
    Complete gap analysis by adding research directions.
    
    Args:
        gap_analysis: Gap analysis result (partial)
        limitation_clusters: Clustered limitations
        paper_cards: Analyzed paper cards
        llm: Optional LLM client
    
    Returns:
        Complete GapAnalysisResult
    """
    generator = DirectionGenerator(llm=llm)
    return await generator.complete_gap_analysis(
        gap_analysis, limitation_clusters, paper_cards
    )

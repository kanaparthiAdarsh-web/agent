"""Structured comparison of papers across multiple dimensions."""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel

from ..schemas import (
    PaperComparisonMatrix, MethodologyComparison, DatasetComparison,
    ResultsComparison, PaperCard, LimitationCategory
)
from ..exceptions import ComparisonError
from ..llm.client_factory import invoke_with_structured_output
from ..llm.config import get_llm_config

logger = logging.getLogger(__name__)


METHODOLOGY_COMPARISON_PROMPT = """Compare the methodologies of these research papers.

PAPERS:
{paper_summaries}

Extract and compare:
1. Each paper's methodology/approach
2. Shared approaches across papers
3. Key differences between approaches
4. Strengths of each paper's methodology
5. Weaknesses of each paper's methodology

Return JSON with:
- methodologies: object mapping paper_id to methodology description
- shared_approaches: array of strings - approaches common across papers
- key_differences: array of strings - key methodological differences
- strengths_by_paper: object mapping paper_id to array of strengths
- weaknesses_by_paper: object mapping paper_id to array of weaknesses

Return ONLY valid JSON."""


DATASET_COMPARISON_PROMPT = """Compare the datasets used across these research papers.

PAPERS:
{paper_summaries}

Extract and compare:
1. Datasets used by each paper
2. Datasets shared across multiple papers
3. Unique datasets per paper
4. Coverage analysis (how many papers use each dataset)

Return JSON with:
- datasets_by_paper: object mapping paper_id to array of dataset names
- shared_datasets: array of dataset names used by multiple papers
- unique_datasets: object mapping paper_id to array of unique datasets
- dataset_coverage: object mapping dataset name to count of papers using it

Return ONLY valid JSON."""


RESULTS_COMPARISON_PROMPT = """Compare the results across these research papers.

PAPERS:
{paper_summaries}

Extract and compare:
1. Evaluation metrics used by each paper
2. Results reported for each metric
3. Best results per metric
4. Consistent findings across papers
5. Conflicting findings between papers

Return JSON with:
- metrics_comparison: object mapping metric name to array of paper_ids reporting it
- results_by_metric: object mapping metric to object of paper_id -> result value
- best_results: object mapping metric to {paper_id, value}
- consistent_findings: array of findings consistent across papers
- conflicting_findings: array of conflicting findings

Return ONLY valid JSON."""


class ComparisonEngine:
    """Engine for comparing papers across multiple dimensions."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the comparison engine.
        
        Args:
            llm: LLM client for comparison (uses default if not provided)
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
    
    async def compare_papers(
        self,
        paper_cards: List[PaperCard],
        research_topic: str,
    ) -> PaperComparisonMatrix:
        """
        Generate comprehensive comparison matrix for papers.
        
        Args:
            paper_cards: List of PaperCard objects to compare
            research_topic: Research topic being analyzed
        
        Returns:
            PaperComparisonMatrix with all comparison dimensions
        
        Raises:
            ComparisonError: If comparison fails
        """
        if len(paper_cards) < 2:
            raise ComparisonError("Need at least 2 papers to compare")
        
        paper_ids = [pc.paper_id for pc in paper_cards]
        
        try:
            # Run comparisons in parallel
            import asyncio
            
            methodology_task = self.compare_methodologies(paper_cards)
            datasets_task = self.compare_datasets(paper_cards)
            results_task = self.compare_results(paper_cards)
            
            methodology, datasets, results = await asyncio.gather(
                methodology_task, datasets_task, results_task,
                return_exceptions=True
            )
            
            # Handle exceptions gracefully
            if isinstance(methodology, Exception):
                logger.warning("Methodology comparison failed: %s", methodology)
                methodology = None
            if isinstance(datasets, Exception):
                logger.warning("Dataset comparison failed: %s", datasets)
                datasets = None
            if isinstance(results, Exception):
                logger.warning("Results comparison failed: %s", results)
                results = None
            
            # Aggregate limitation counts by category
            limitations_summary = self._aggregate_limitations(paper_cards)
            
            # Generate overall summary
            summary = self._generate_summary(
                paper_ids=paper_ids,
                methodology=methodology,
                datasets=datasets,
                results=results,
                limitations_summary=limitations_summary,
            )
            
            # Extract key insights
            key_insights = self._extract_insights(
                methodology=methodology,
                datasets=datasets,
                results=results,
            )
            
            comparison = PaperComparisonMatrix(
                comparison_id=f"comp_{uuid4().hex[:12]}",
                research_topic=research_topic,
                paper_ids=paper_ids,
                methodology=methodology,
                datasets=datasets,
                results=results,
                limitations_summary=limitations_summary,
                summary=summary,
                key_insights=key_insights,
            )
            
            logger.info("Generated comparison matrix for %d papers", len(paper_ids))
            return comparison
            
        except Exception as e:
            logger.error("Failed to generate comparison matrix: %s", e)
            raise ComparisonError(f"Comparison failed: {e}")
    
    async def compare_methodologies(
        self, paper_cards: List[PaperCard]
    ) -> MethodologyComparison:
        """Compare methodologies across papers."""
        llm = self._get_llm()
        
        paper_summaries = self._format_paper_summaries(paper_cards)
        prompt = METHODOLOGY_COMPARISON_PROMPT.format(paper_summaries=paper_summaries)
        
        result = await invoke_with_structured_output(llm, prompt)
        
        if not isinstance(result, dict):
            raise ComparisonError("Invalid methodology comparison response")
        
        paper_ids = [pc.paper_id for pc in paper_cards]
        
        comparison = MethodologyComparison(
            paper_ids=paper_ids,
            methodologies=result.get("methodologies", {}),
            shared_approaches=result.get("shared_approaches", []),
            key_differences=result.get("key_differences", []),
            strengths_by_paper=result.get("strengths_by_paper", {}),
            weaknesses_by_paper=result.get("weaknesses_by_paper", {}),
        )
        
        return comparison
    
    async def compare_datasets(
        self, paper_cards: List[PaperCard]
    ) -> DatasetComparison:
        """Compare datasets across papers."""
        llm = self._get_llm()
        
        paper_summaries = self._format_paper_summaries(paper_cards)
        prompt = DATASET_COMPARISON_PROMPT.format(paper_summaries=paper_summaries)
        
        result = await invoke_with_structured_output(llm, prompt)
        
        if not isinstance(result, dict):
            raise ComparisonError("Invalid dataset comparison response")
        
        paper_ids = [pc.paper_id for pc in paper_cards]
        
        # Also extract directly from PaperCards for accuracy
        direct_datasets = {pc.paper_id: pc.datasets for pc in paper_cards}
        
        # Merge LLM extraction with direct extraction
        datasets_by_paper = result.get("datasets_by_paper", direct_datasets)
        
        # Calculate shared datasets
        all_datasets = []
        for datasets in datasets_by_paper.values():
            all_datasets.extend(datasets)
        
        dataset_counts: Dict[str, int] = {}
        for ds in all_datasets:
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        
        shared_datasets = [ds for ds, count in dataset_counts.items() if count >= 2]
        unique_datasets = {
            pid: [ds for ds in datasets if dataset_counts.get(ds, 0) == 1]
            for pid, datasets in datasets_by_paper.items()
        }
        
        comparison = DatasetComparison(
            paper_ids=paper_ids,
            datasets_by_paper=datasets_by_paper,
            shared_datasets=shared_datasets,
            unique_datasets=unique_datasets,
            dataset_coverage=dataset_counts,
        )
        
        return comparison
    
    async def compare_results(
        self, paper_cards: List[PaperCard]
    ) -> ResultsComparison:
        """Compare results across papers."""
        llm = self._get_llm()
        
        paper_summaries = self._format_paper_summaries(paper_cards)
        prompt = RESULTS_COMPARISON_PROMPT.format(paper_summaries=paper_summaries)
        
        result = await invoke_with_structured_output(llm, prompt)
        
        if not isinstance(result, dict):
            raise ComparisonError("Invalid results comparison response")
        
        paper_ids = [pc.paper_id for pc in paper_cards]
        
        comparison = ResultsComparison(
            paper_ids=paper_ids,
            metrics_comparison=result.get("metrics_comparison", {}),
            results_by_metric=result.get("results_by_metric", {}),
            best_results=result.get("best_results", {}),
            consistent_findings=result.get("consistent_findings", []),
            conflicting_findings=result.get("conflicting_findings", []),
        )
        
        return comparison
    
    def _format_paper_summaries(self, paper_cards: List[PaperCard]) -> str:
        """Format paper cards into a summary string for LLM prompts."""
        summaries = []
        for pc in paper_cards:
            summary_parts = [f"Paper ID: {pc.paper_id}"]
            
            if pc.research_problem:
                summary_parts.append(f"Problem: {pc.research_problem}")
            if pc.methodology:
                summary_parts.append(f"Methodology: {pc.methodology}")
            if pc.models:
                summary_parts.append(f"Models: {', '.join(pc.models)}")
            if pc.datasets:
                summary_parts.append(f"Datasets: {', '.join(pc.datasets)}")
            if pc.evaluation_metrics:
                summary_parts.append(f"Metrics: {', '.join(pc.evaluation_metrics)}")
            if pc.key_results:
                summary_parts.append(f"Key Results: {'; '.join(pc.key_results[:3])}")
            if pc.limitations:
                summary_parts.append(f"Limitations: {'; '.join(pc.limitations[:2])}")
            
            summaries.append("\n".join(summary_parts))
        
        return "\n\n---\n\n".join(summaries)
    
    def _aggregate_limitations(
        self, paper_cards: List[PaperCard]
    ) -> Dict[str, int]:
        """Aggregate limitations by category across papers."""
        # This is a simplified version - full implementation would classify
        # each limitation text into categories using LLM
        limitation_counts: Dict[str, int] = {}
        
        for pc in paper_cards:
            for lim in pc.limitations:
                # Simple keyword-based categorization
                lim_lower = lim.lower()
                
                if any(kw in lim_lower for kw in ["dataset", "data", "training data"]):
                    cat = LimitationCategory.DATASET.value
                elif any(kw in lim_lower for kw in ["compute", "gpu", "memory", "cost", "expensive"]):
                    cat = LimitationCategory.COMPUTATIONAL_COST.value
                elif any(kw in lim_lower for kw in ["scale", "scalability", "large"]):
                    cat = LimitationCategory.SCALABILITY.value
                elif any(kw in lim_lower for kw in ["interpret", "explain", "black box"]):
                    cat = LimitationCategory.INTERPRETABILITY.value
                elif any(kw in lim_lower for kw in ["generalize", "generalisation", "domain"]):
                    cat = LimitationCategory.GENERALIZATION.value
                elif any(kw in lim_lower for kw in ["reproduce", "reproducibility", "code"]):
                    cat = LimitationCategory.REPRODUCIBILITY.value
                elif any(kw in lim_lower for kw in ["bias", "fairness"]):
                    cat = LimitationCategory.BIAS.value
                else:
                    cat = LimitationCategory.OTHER.value
                
                limitation_counts[cat] = limitation_counts.get(cat, 0) + 1
        
        return limitation_counts
    
    def _generate_summary(
        self,
        paper_ids: List[str],
        methodology: Optional[MethodologyComparison],
        datasets: Optional[DatasetComparison],
        results: Optional[ResultsComparison],
        limitations_summary: Dict[str, int],
    ) -> str:
        """Generate overall comparison summary."""
        parts = [f"Comparison of {len(paper_ids)} papers."]
        
        if methodology and methodology.shared_approaches:
            parts.append(f"Shared approaches: {', '.join(methodology.shared_approaches[:2])}")
        
        if datasets and datasets.shared_datasets:
            parts.append(f"Common datasets: {', '.join(datasets.shared_datasets[:3])}")
        
        if results and results.consistent_findings:
            parts.append(f"Consistent findings: {results.consistent_findings[0]}")
        
        if limitations_summary:
            top_lim = max(limitations_summary.items(), key=lambda x: x[1])
            parts.append(f"Most common limitation category: {top_lim[0]} ({top_lim[1]} papers)")
        
        return " ".join(parts)
    
    def _extract_insights(
        self,
        methodology: Optional[MethodologyComparison],
        datasets: Optional[DatasetComparison],
        results: Optional[ResultsComparison],
    ) -> List[str]:
        """Extract key insights from comparisons."""
        insights = []
        
        if methodology:
            if methodology.key_differences:
                insights.append(methodology.key_differences[0])
            if methodology.shared_approaches:
                insights.append(f"All papers use: {methodology.shared_approaches[0]}")
        
        if datasets:
            if datasets.shared_datasets:
                insights.append(f"{len(datasets.shared_datasets)} datasets used across multiple papers")
        
        if results:
            if results.conflicting_findings:
                insights.append(f"Conflicting finding: {results.conflicting_findings[0]}")
        
        return insights


async def compare_papers(
    paper_cards: List[PaperCard],
    research_topic: str,
    llm: Optional[BaseChatModel] = None,
) -> PaperComparisonMatrix:
    """
    Convenience function to compare papers.
    
    Args:
        paper_cards: List of PaperCard objects
        research_topic: Research topic
        llm: Optional LLM client
    
    Returns:
        PaperComparisonMatrix
    """
    engine = ComparisonEngine(llm=llm)
    return await engine.compare_papers(paper_cards, research_topic)

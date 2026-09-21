"""Limitation extraction, classification, and clustering."""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel

from ..schemas import (
    LimitationExtraction, LimitationCluster, LimitationCategory,
    PaperCard, ChunkData
)
from ..exceptions import LimitationError
from ..llm.client_factory import invoke_with_structured_output
from ..llm.config import get_llm_config

logger = logging.getLogger(__name__)


LIMITATION_EXTRACTION_PROMPT = """Extract limitations acknowledged in this research paper text.

For each limitation, identify:
1. The original text describing the limitation
2. A normalized/cleaned description
3. The category of limitation (see list below)
4. Which methods/models are affected
5. Which datasets are affected (if any)
6. Your confidence in the extraction

CONTROLLED TAXONOMY - Use exactly these categories:
- Dataset: Issues with datasets used
- Generalization: Limited generalizability to other domains/settings
- Robustness: Sensitivity to adversarial examples or distribution shifts
- Evaluation: Limitations in evaluation methodology
- Reproducibility: Difficulty reproducing results
- Computational Cost: High compute/memory requirements
- Scalability: Issues scaling to larger problems
- Interpretability: Lack of model interpretability/explainability
- Deployment: Challenges deploying in real-world settings
- Bias: Potential biases in model or data
- Temporal: Time-related limitations (outdated data, etc.)
- Modality: Limited to specific data modalities
- Annotation: Issues with data annotation quality
- Data Quality: General data quality issues
- Methodological: Methodological limitations
- Other: Limitations not fitting other categories

Return JSON array of objects with:
- original_text: string - Exact or near-exact text from paper
- normalized_description: string - Cleaned description
- category: string - One of the categories above
- subcategory: string or null - Optional subcategory
- affected_methods: array of strings - Methods/models affected
- affected_datasets: array of strings - Datasets affected
- confidence: float - Confidence (0.0 to 1.0)

Text to analyze:
{text}

Return ONLY valid JSON array."""


CLUSTERING_PROMPT = """Cluster these extracted limitations by theme.

LIMITATIONS:
{limitations_json}

Group limitations that express similar concerns or themes.

For each cluster, provide:
1. A theme name describing the cluster
2. The primary category
3. A detailed description of the pattern
4. IDs of limitations in the cluster
5. Representative quotes (1-2 per cluster)
6. Frequency (number of papers with this limitation)
7. Severity assessment (minor/moderate/significant)

Return JSON array of cluster objects with:
- theme: string
- category: string - One of the controlled taxonomy categories
- description: string
- member_limitations: array of limitation IDs
- representative_quotes: array of strings
- frequency: integer
- severity: string - "minor", "moderate", or "significant"

Return ONLY valid JSON array."""


class LimitationAnalyzer:
    """Analyzes and clusters limitations from research papers."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the limitation analyzer.
        
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
    
    async def extract_limitations(
        self,
        paper_card: PaperCard,
        chunks: Optional[List[ChunkData]] = None,
    ) -> List[LimitationExtraction]:
        """
        Extract limitations from a paper.
        
        Args:
            paper_card: PaperCard with extracted information
            chunks: Optional text chunks for additional context
        
        Returns:
            List of LimitationExtraction objects
        
        Raises:
            LimitationError: If extraction fails
        """
        try:
            llm = self._get_llm()
            
            # Combine explicit limitations from PaperCard with chunk text
            text_parts = []
            
            # Add explicit limitations from PaperCard
            if paper_card.limitations:
                for lim in paper_card.limitations:
                    text_parts.append(f"Limitation: {lim}")
            
            # Add relevant chunk text
            if chunks:
                for chunk in chunks[:20]:  # Limit chunks
                    if chunk.section_type and any(
                        kw in chunk.section_type.lower() 
                        for kw in ["limitation", "discussion", "conclusion", "future"]
                    ):
                        text_parts.append(f"[{chunk.section_type}]\n{chunk.text[:500]}")
            
            if not text_parts:
                logger.debug("No limitation text available for paper %s", paper_card.paper_id)
                return []
            
            full_text = "\n\n".join(text_parts)
            prompt = LIMITATION_EXTRACTION_PROMPT.format(text=full_text)
            
            result = await invoke_with_structured_output(llm, prompt)
            
            if not isinstance(result, list):
                return []
            
            limitations = []
            for i, item in enumerate(result[:10]):  # Limit to 10 limitations
                if not isinstance(item, dict):
                    continue
                
                category_str = item.get("category", "Other")
                try:
                    category = LimitationCategory(category_str)
                except ValueError:
                    category = LimitationCategory.OTHER
                
                confidence = item.get("confidence", 0.5)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                confidence = max(0.0, min(1.0, confidence))
                
                # Get chunk reference if available
                chunk_id = None
                page_numbers = None
                section = None
                if chunks and i < len(chunks):
                    chunk_id = chunks[i].chunk_id
                    page_numbers = chunks[i].page_numbers
                    section = chunks[i].section_type
                
                limitation = LimitationExtraction(
                    limitation_id=f"lim_{paper_card.paper_id}_{uuid4().hex[:8]}_{i}",
                    paper_id=paper_card.paper_id,
                    chunk_id=chunk_id,
                    original_text=item.get("original_text", ""),
                    normalized_description=item.get("normalized_description", ""),
                    category=category,
                    subcategory=item.get("subcategory"),
                    affected_methods=item.get("affected_methods", []) or [],
                    affected_datasets=item.get("affected_datasets", []) or [],
                    page_numbers=page_numbers,
                    section=section,
                    confidence=confidence,
                )
                
                if limitation.original_text.strip() or limitation.normalized_description.strip():
                    limitations.append(limitation)
            
            logger.info("Extracted %d limitations from paper %s", len(limitations), paper_card.paper_id)
            return limitations
            
        except Exception as e:
            logger.error("Failed to extract limitations from paper %s: %s", paper_card.paper_id, e)
            raise LimitationError(f"Limitation extraction failed: {e}")
    
    async def cluster_limitations(
        self,
        limitations: List[LimitationExtraction],
    ) -> List[LimitationCluster]:
        """
        Cluster limitations by theme across papers.
        
        Args:
            limitations: List of all extracted limitations
        
        Returns:
            List of LimitationCluster objects
        
        Raises:
            LimitationError: If clustering fails
        """
        if len(limitations) < 2:
            # Not enough limitations to cluster
            return []
        
        try:
            llm = self._get_llm()
            
            # Format limitations for LLM
            limitations_json = []
            for lim in limitations:
                limitations_json.append({
                    "id": lim.limitation_id,
                    "paper_id": lim.paper_id,
                    "description": lim.normalized_description,
                    "original_text": lim.original_text,
                    "category": lim.category.value,
                    "affected_methods": lim.affected_methods,
                })
            
            prompt = CLUSTERING_PROMPT.format(
                limitations_json="\n".join(str(l) for l in limitations_json)
            )
            
            result = await invoke_with_structured_output(llm, prompt)
            
            if not isinstance(result, list):
                return []
            
            clusters = []
            for i, item in enumerate(result[:15]):  # Limit clusters
                if not isinstance(item, dict):
                    continue
                
                category_str = item.get("category", "Other")
                try:
                    category = LimitationCategory(category_str)
                except ValueError:
                    category = LimitationCategory.OTHER
                
                severity = item.get("severity", "moderate")
                if severity not in ["minor", "moderate", "significant"]:
                    severity = "moderate"
                
                member_ids = item.get("member_limitations", [])
                source_papers = list(set(
                    lim.paper_id for lim in limitations 
                    if lim.limitation_id in member_ids
                ))
                
                cluster = LimitationCluster(
                    cluster_id=f"cluster_{uuid4().hex[:8]}_{i}",
                    theme=item.get("theme", "Unclassified limitations"),
                    category=category,
                    description=item.get("description", ""),
                    member_limitations=member_ids,
                    source_papers=source_papers,
                    representative_quotes=item.get("representative_quotes", []) or [],
                    frequency=len(source_papers),
                    severity=severity,
                )
                
                if cluster.description.strip():
                    clusters.append(cluster)
            
            logger.info("Created %d limitation clusters", len(clusters))
            return clusters
            
        except Exception as e:
            logger.error("Failed to cluster limitations: %s", e)
            raise LimitationError(f"Limitation clustering failed: {e}")
    
    def classify_limitation_category(
        self,
        limitation_text: str,
    ) -> LimitationCategory:
        """
        Classify a limitation into the controlled taxonomy.
        
        This is a rule-based fallback when LLM classification is unavailable.
        
        Args:
            limitation_text: Text describing the limitation
        
        Returns:
            LimitationCategory
        """
        text_lower = limitation_text.lower()
        
        # Check keywords for each category
        if any(kw in text_lower for kw in ["dataset", "data ", "training data", "test data"]):
            return LimitationCategory.DATASET
        elif any(kw in text_lower for kw in ["generalize", "generalisation", "domain shift", "out-of-distribution"]):
            return LimitationCategory.GENERALIZATION
        elif any(kw in text_lower for kw in ["robust", "adversarial", "perturbation", "noise"]):
            return LimitationCategory.ROBUSTNESS
        elif any(kw in text_lower for kw in ["evaluate", "evaluation", "benchmark", "metric"]):
            return LimitationCategory.EVALUATION
        elif any(kw in text_lower for kw in ["reproduce", "reproducibility", "replicate", "code available"]):
            return LimitationCategory.REPRODUCIBILITY
        elif any(kw in text_lower for kw in ["compute", "gpu", "tpu", "memory", "expensive", "cost", "flops"]):
            return LimitationCategory.COMPUTATIONAL_COST
        elif any(kw in text_lower for kw in ["scale", "scalability", "large-scale", "million"]):
            return LimitationCategory.SCALABILITY
        elif any(kw in text_lower for kw in ["interpret", "explain", "black box", "transparent"]):
            return LimitationCategory.INTERPRETABILITY
        elif any(kw in text_lower for kw in ["deploy", "deployment", "production", "real-world"]):
            return LimitationCategory.DEPLOYMENT
        elif any(kw in text_lower for kw in ["bias", "fairness", "demographic", "stereotype"]):
            return LimitationCategory.BIAS
        elif any(kw in text_lower for kw in ["temporal", "time", "outdated", "recent"]):
            return LimitationCategory.TEMPORAL
        elif any(kw in text_lower for kw in ["modality", "multimodal", "text-only", "image-only"]):
            return LimitationCategory.MODALITY
        elif any(kw in text_lower for kw in ["annotation", "annotated", "labeled", "labeling"]):
            return LimitationCategory.ANNOTATION
        elif any(kw in text_lower for kw in ["quality", "noisy", "clean", "error"]):
            return LimitationCategory.DATA_QUALITY
        elif any(kw in text_lower for kw in ["methodology", "methodological", "approach", "design"]):
            return LimitationCategory.METHODOLOGICAL
        else:
            return LimitationCategory.OTHER


async def extract_limitations(
    paper_card: PaperCard,
    chunks: Optional[List[ChunkData]] = None,
    llm: Optional[BaseChatModel] = None,
) -> List[LimitationExtraction]:
    """
    Convenience function to extract limitations from a paper.
    
    Args:
        paper_card: PaperCard with extracted info
        chunks: Optional text chunks
        llm: Optional LLM client
    
    Returns:
        List of LimitationExtraction objects
    """
    analyzer = LimitationAnalyzer(llm=llm)
    return await analyzer.extract_limitations(paper_card, chunks)


async def cluster_limitations(
    limitations: List[LimitationExtraction],
    llm: Optional[BaseChatModel] = None,
) -> List[LimitationCluster]:
    """
    Convenience function to cluster limitations.
    
    Args:
        limitations: List of extracted limitations
        llm: Optional LLM client
    
    Returns:
        List of LimitationCluster objects
    """
    analyzer = LimitationAnalyzer(llm=llm)
    return await analyzer.cluster_limitations(limitations)

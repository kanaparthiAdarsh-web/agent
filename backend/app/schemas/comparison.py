"""Pydantic schemas for comparison and limitations."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class LimitationCategory(str, Enum):
    """Controlled taxonomy of limitation types."""
    DATASET = "Dataset"
    GENERALIZATION = "Generalization"
    ROBUSTNESS = "Robustness"
    EVALUATION = "Evaluation"
    REPRODUCIBILITY = "Reproducibility"
    COMPUTATIONAL_COST = "Computational Cost"
    SCALABILITY = "Scalability"
    INTERPRETABILITY = "Interpretability"
    DEPLOYMENT = "Deployment"
    BIAS = "Bias"
    TEMPORAL = "Temporal"
    MODALITY = "Modality"
    ANNOTATION = "Annotation"
    DATA_QUALITY = "Data Quality"
    METHODOLOGICAL = "Methodological"
    OTHER = "Other"


class MethodologyComparison(BaseModel):
    """Comparison of methodologies between papers."""
    paper_ids: List[str] = Field(..., min_length=2, description="Papers being compared")
    
    # Methodology details per paper
    methodologies: Dict[str, str] = Field(default_factory=dict, 
                                          description="paper_id -> methodology description")
    
    # Commonalities
    shared_approaches: List[str] = Field(default_factory=list, 
                                         description="Approaches common across papers")
    
    # Differences
    key_differences: List[str] = Field(default_factory=list, 
                                       description="Key methodological differences")
    
    # Analysis
    strengths_by_paper: Dict[str, List[str]] = Field(default_factory=dict,
                                                     description="paper_id -> list of strengths")
    weaknesses_by_paper: Dict[str, List[str]] = Field(default_factory=dict,
                                                      description="paper_id -> list of weaknesses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "paper_ids": ["paper_001", "paper_002"],
                "shared_approaches": ["Both use transformer architectures"],
                "key_differences": ["Paper 1 uses LoRA, Paper 2 uses prefix tuning"],
                "strengths_by_paper": {"paper_001": ["Parameter efficient"]},
                "weaknesses_by_paper": {"paper_002": ["Requires more memory"]}
            }
        }


class DatasetComparison(BaseModel):
    """Comparison of datasets used across papers."""
    paper_ids: List[str] = Field(..., min_length=2, description="Papers being compared")
    
    # Datasets per paper
    datasets_by_paper: Dict[str, List[str]] = Field(default_factory=dict,
                                                    description="paper_id -> list of datasets")
    
    # Common datasets
    shared_datasets: List[str] = Field(default_factory=list, 
                                       description="Datasets used by multiple papers")
    
    # Unique datasets
    unique_datasets: Dict[str, List[str]] = Field(default_factory=dict,
                                                  description="paper_id -> unique datasets")
    
    # Coverage analysis
    dataset_coverage: Dict[str, int] = Field(default_factory=dict,
                                             description="dataset_name -> number of papers using it")
    
    class Config:
        json_schema_extra = {
            "example": {
                "paper_ids": ["paper_001", "paper_002"],
                "datasets_by_paper": {"paper_001": ["GLUE", "SuperGLUE"]},
                "shared_datasets": ["GLUE"],
                "dataset_coverage": {"GLUE": 2}
            }
        }


class ResultsComparison(BaseModel):
    """Comparison of results across papers."""
    paper_ids: List[str] = Field(..., min_length=2, description="Papers being compared")
    
    # Metrics used
    metrics_comparison: Dict[str, List[str]] = Field(default_factory=dict,
                                                    description="metric -> papers reporting it")
    
    # Results by metric and paper
    results_by_metric: Dict[str, Dict[str, Any]] = Field(default_factory=dict,
                                                         description="metric -> paper_id -> result")
    
    # Best results per metric
    best_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict,
                                                    description="metric -> {paper_id, value}")
    
    # Consistency analysis
    consistent_findings: List[str] = Field(default_factory=list,
                                           description="Findings consistent across papers")
    conflicting_findings: List[str] = Field(default_factory=list,
                                            description="Conflicting findings between papers")
    
    class Config:
        json_schema_extra = {
            "example": {
                "paper_ids": ["paper_001", "paper_002"],
                "metrics_comparison": {"accuracy": ["paper_001", "paper_002"]},
                "consistent_findings": ["LoRA outperforms full fine-tuning on GLUE"],
                "conflicting_findings": []
            }
        }


class LimitationExtraction(BaseModel):
    """Extracted limitation from a paper."""
    limitation_id: str = Field(..., description="Unique limitation identifier")
    paper_id: str = Field(..., description="Reference to source paper")
    chunk_id: Optional[str] = Field(default=None, description="Reference to source chunk")
    
    # Content
    original_text: str = Field(..., min_length=1, description="Original text describing limitation")
    normalized_description: str = Field(..., min_length=1, 
                                        description="Normalized/cleaned description")
    
    # Classification
    category: LimitationCategory = Field(..., description="Limitation category")
    subcategory: Optional[str] = Field(default=None, description="Optional subcategory")
    
    # Evidence
    page_numbers: Optional[List[int]] = Field(default=None, description="Page numbers")
    section: Optional[str] = Field(default=None, description="Section where limitation appears")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Extraction confidence")
    
    # Affected aspects
    affected_methods: List[str] = Field(default_factory=list, 
                                        description="Methods/models affected by this limitation")
    affected_datasets: List[str] = Field(default_factory=list,
                                         description="Datasets affected by this limitation")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Extraction time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "limitation_id": "lim_001",
                "paper_id": "paper_001",
                "original_text": "Our method requires significant GPU memory during training",
                "normalized_description": "High GPU memory requirements during training",
                "category": "Computational Cost",
                "affected_methods": ["LoRA"],
                "confidence": 0.89
            }
        }


class LimitationCluster(BaseModel):
    """Cluster of similar limitations across papers."""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    
    # Cluster content
    theme: str = Field(..., min_length=1, description="Theme describing the cluster")
    category: LimitationCategory = Field(..., description="Primary limitation category")
    description: str = Field(..., min_length=1, description="Detailed description of the pattern")
    
    # Member limitations
    member_limitations: List[str] = Field(default_factory=list,
                                          description="IDs of limitations in this cluster")
    
    # Source papers
    source_papers: List[str] = Field(default_factory=list,
                                     description="Papers that have limitations in this cluster")
    
    # Representative examples
    representative_quotes: List[str] = Field(default_factory=list,
                                             description="Representative quotes from cluster members")
    
    # Frequency analysis
    frequency: int = Field(default=0, ge=0, description="Number of papers with this limitation pattern")
    
    # Severity assessment
    severity: str = Field(default="moderate", 
                          description="Assessed severity (minor/moderate/significant)")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Cluster creation time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cluster_id": "cluster_001",
                "theme": "High computational requirements",
                "category": "Computational Cost",
                "description": "Multiple papers report high GPU memory or compute requirements",
                "source_papers": ["paper_001", "paper_002", "paper_003"],
                "frequency": 3,
                "severity": "significant"
            }
        }


class PaperComparisonMatrix(BaseModel):
    """Complete comparison matrix for multiple papers."""
    comparison_id: str = Field(..., description="Unique comparison identifier")
    research_topic: str = Field(..., description="Research topic being analyzed")
    
    # Papers included
    paper_ids: List[str] = Field(..., min_length=2, description="All papers in comparison")
    
    # Comparison components
    methodology: Optional[MethodologyComparison] = Field(default=None, 
                                                         description="Methodology comparison")
    datasets: Optional[DatasetComparison] = Field(default=None,
                                                  description="Dataset comparison")
    results: Optional[ResultsComparison] = Field(default=None,
                                                 description="Results comparison")
    limitations_summary: Dict[str, int] = Field(default_factory=dict,
                                                description="category -> count of limitations")
    
    # Overall analysis
    summary: str = Field(default="", description="Overall comparison summary")
    key_insights: List[str] = Field(default_factory=list, description="Key insights from comparison")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Comparison creation time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "comparison_id": "comp_001",
                "research_topic": "Parameter-efficient fine-tuning",
                "paper_ids": ["paper_001", "paper_002", "paper_003"],
                "key_insights": ["LoRA is most parameter-efficient", "All methods evaluated on GLUE"],
                "limitations_summary": {"Computational Cost": 2, "Scalability": 1}
            }
        }

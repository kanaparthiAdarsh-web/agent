"""Pydantic schemas for research gaps and directions."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class GapCoverageStatus(str, Enum):
    """Status indicating how well a potential gap is addressed in literature."""
    LIMITED_EVIDENCE = "limited_evidence"
    PARTIALLY_ADDRESSED = "partially_addressed"
    SUBSTANTIALLY_ADDRESSED = "substantially_addressed"
    INSUFFICIENT_LITERATURE = "insufficient_literature"


class GapCategory(str, Enum):
    """Categories of potential research gaps."""
    METHODOLOGICAL = "methodological"
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    DATASET = "dataset"
    EVALUATION = "evaluation"
    APPLICATION = "application"
    REPRODUCIBILITY = "reproducibility"
    SCALABILITY = "scalability"
    INTERPRETABILITY = "interpretability"
    OTHER = "other"


class Counterevidence(BaseModel):
    """Evidence that potentially contradicts a gap candidate."""
    counterevidence_id: str = Field(..., description="Unique identifier")
    gap_candidate_id: str = Field(..., description="Reference to gap candidate")
    
    paper_id: str = Field(..., description="Paper containing counterevidence")
    chunk_id: Optional[str] = Field(default=None, description="Source chunk if available")
    
    # Content
    description: str = Field(..., min_length=1, description="Description of counterevidence")
    exact_quote: Optional[str] = Field(default=None, description="Exact quote if available")
    
    # Relevance assessment
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0, 
                                   description="How relevant this is to the gap")
    strength: str = Field(default="moderate", 
                          description="Strength of counterevidence (weak/moderate/strong)")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "counterevidence_id": "ce_001",
                "gap_candidate_id": "gap_001",
                "paper_id": "paper_005",
                "description": "Paper addresses similar problem with different approach",
                "relevance_score": 0.72,
                "strength": "moderate"
            }
        }


class GapCandidate(BaseModel):
    """A potential research gap candidate before verification."""
    gap_id: str = Field(..., description="Unique gap candidate identifier")
    research_topic: str = Field(..., description="Research topic this gap relates to")
    
    # Gap description
    description: str = Field(..., min_length=1, 
                            description="Clear description of the potential gap")
    category: GapCategory = Field(..., description="Category of this gap")
    
    # Supporting evidence
    supporting_papers: List[str] = Field(default_factory=list,
                                         description="Papers that support this gap existence")
    supporting_evidence: List[str] = Field(default_factory=list,
                                           description="Evidence snippets supporting the gap")
    
    # Related limitations
    related_limitation_clusters: List[str] = Field(default_factory=list,
                                                   description="Related limitation cluster IDs")
    
    # Affected aspects
    affected_methods: List[str] = Field(default_factory=list,
                                        description="Methods/models affected by this gap")
    affected_datasets: List[str] = Field(default_factory=list,
                                         description="Datasets affected by this gap")
    
    # Initial confidence
    initial_confidence: float = Field(default=0.5, ge=0.0, le=1.0,
                                      description="Initial confidence before verification")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gap_id": "gap_001",
                "research_topic": "Parameter-efficient fine-tuning",
                "description": "Limited evaluation of LoRA on multimodal tasks",
                "category": "empirical",
                "supporting_papers": ["paper_001", "paper_002"],
                "affected_methods": ["LoRA", "Adapter"],
                "initial_confidence": 0.65
            }
        }


class GapVerification(BaseModel):
    """Verification result for a gap candidate."""
    verification_id: str = Field(..., description="Unique verification identifier")
    gap_candidate_id: str = Field(..., description="Reference to gap candidate")
    
    # Verification status - CRITICAL: Never claim absolute novelty
    coverage_status: GapCoverageStatus = Field(
        ..., 
        description="How well this gap is addressed in literature. "
                    "NEVER interpret as definitive novelty claim."
    )
    
    # Evidence analysis
    supporting_evidence_count: int = Field(default=0, ge=0, 
                                           description="Count of supporting evidence items")
    counterevidence: List[Counterevidence] = Field(default_factory=list,
                                                   description="Counterevidence found")
    
    # Coverage summary
    coverage_summary: str = Field(..., min_length=1,
                                  description="Summary of literature coverage analysis. "
                                             "Use cautious language like 'limited evidence', "
                                             "'partially addressed', etc.")
    
    # Reasoning
    reasoning: str = Field(..., min_length=1,
                          description="Detailed reasoning for the coverage status assessment. "
                                     "Must reference specific papers and evidence.")
    
    # Confidence in assessment
    assessment_confidence: float = Field(default=0.5, ge=0.0, le=1.0,
                                         description="Confidence in this verification assessment")
    
    # Important disclaimer
    novelty_disclaimer: str = Field(
        default="This assessment indicates limited evidence in the analyzed literature. "
                "It does NOT claim the gap is definitively novel or unexplored. "
                "Further literature review may reveal additional relevant work.",
        description="Required disclaimer about novelty claims"
    )
    
    verified_at: datetime = Field(default_factory=datetime.utcnow, description="Verification time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "verification_id": "ver_001",
                "gap_candidate_id": "gap_001",
                "coverage_status": "limited_evidence",
                "coverage_summary": "Limited evidence found for multimodal evaluation of parameter-efficient methods",
                "reasoning": "Among 15 analyzed papers, only 2 evaluated on multimodal tasks. Papers [1,2] focus on text-only benchmarks.",
                "assessment_confidence": 0.78,
                "counterevidence": []
            }
        }


class ResearchDirection(BaseModel):
    """Evidence-grounded research direction suggestion."""
    direction_id: str = Field(..., description="Unique direction identifier")
    research_topic: str = Field(..., description="Research topic this direction relates to")
    
    # Proposed research
    proposed_problem: str = Field(..., min_length=1,
                                  description="Proposed research problem to investigate")
    motivation: str = Field(..., min_length=1,
                           description="Motivation grounded in evidence analysis")
    
    # Methodology suggestions
    suggested_methodology: Optional[str] = Field(default=None,
                                                 description="Suggested methodology/approach")
    possible_datasets: List[str] = Field(default_factory=list,
                                         description="Suggested datasets for evaluation")
    candidate_models: List[str] = Field(default_factory=list,
                                        description="Suggested models/architectures")
    
    # Evaluation strategy
    evaluation_strategy: Optional[str] = Field(default=None,
                                               description="Suggested evaluation approach")
    evaluation_metrics: List[str] = Field(default_factory=list,
                                          description="Suggested evaluation metrics")
    
    # Feasibility
    feasibility_considerations: List[str] = Field(default_factory=list,
                                                  description="Practical feasibility considerations")
    required_resources: List[str] = Field(default_factory=list,
                                          description="Required resources (compute, data, etc.)")
    
    # Assumptions
    assumptions: List[str] = Field(default_factory=list,
                                   description="Key assumptions underlying this direction")
    
    # Evidence grounding - CRITICAL: Distinguish evidence from speculation
    supporting_evidence: List[str] = Field(default_factory=list,
                                           description="Evidence from analysis supporting this direction")
    evidence_paper_ids: List[str] = Field(default_factory=list,
                                          description="IDs of papers providing supporting evidence")
    
    # Relation to gaps
    related_gap_ids: List[str] = Field(default_factory=list,
                                       description="Gap candidates this direction addresses")
    
    # Priority assessment
    priority: str = Field(default="medium", 
                         description="Suggested priority (low/medium/high)")
    novelty_potential: str = Field(
        default="uncertain",
        description="Potential novelty level. NEVER claim definite novelty. "
                    "Use: uncertain/potential/plausible."
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "direction_id": "dir_001",
                "research_topic": "Parameter-efficient fine-tuning",
                "proposed_problem": "Evaluate LoRA variants on multimodal vision-language tasks",
                "motivation": "Current literature shows limited evaluation beyond NLP benchmarks",
                "suggested_methodology": "Apply LoRA to vision transformer backbones",
                "possible_datasets": ["VQA", "GQA", "Visual Genome"],
                "supporting_evidence": ["Only 2/15 papers evaluated on non-text modalities"],
                "evidence_paper_ids": ["paper_003", "paper_007"],
                "priority": "high",
                "novelty_potential": "potential"
            }
        }


class GapAnalysisResult(BaseModel):
    """Complete gap analysis result including candidates, verification, and directions."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    research_topic: str = Field(..., description="Research topic analyzed")
    
    # Gap candidates
    gap_candidates: List[GapCandidate] = Field(default_factory=list,
                                               description="All identified gap candidates")
    
    # Verification results
    verifications: List[GapVerification] = Field(default_factory=list,
                                                 description="Verification results for each candidate")
    
    # Research directions
    research_directions: List[ResearchDirection] = Field(default_factory=list,
                                                         description="Suggested research directions")
    
    # Summary statistics
    total_candidates: int = Field(default=0, ge=0, description="Total gap candidates identified")
    by_coverage_status: Dict[str, int] = Field(default_factory=dict,
                                               description="Count by coverage status")
    
    # Overall summary
    executive_summary: str = Field(default="",
                                   description="Executive summary of gap analysis findings")
    
    # Important overall disclaimer
    overall_disclaimer: str = Field(
        default="This analysis identifies POTENTIAL research gaps based on the analyzed literature. "
                "These are NOT claims of definitive novelty or unexplored areas. "
                "The assessment reflects limited evidence within the scope of analyzed papers. "
                "Comprehensive literature review may reveal additional relevant work.",
        description="Required overall disclaimer"
    )
    
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": "analysis_001",
                "research_topic": "Parameter-efficient fine-tuning",
                "total_candidates": 5,
                "by_coverage_status": {"limited_evidence": 2, "partially_addressed": 3},
                "executive_summary": "Analysis identified 5 potential gaps, primarily in multimodal evaluation"
            }
        }

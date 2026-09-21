"""Pydantic schemas for workflow and job state management."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class WorkflowStatus(str, Enum):
    """Status of a research workflow/job."""
    PENDING = "pending"
    DISCOVERING = "discovering"
    PROCESSING_PAPERS = "processing_papers"
    EXTRACTING_EVIDENCE = "extracting_evidence"
    COMPARING = "comparing"
    ANALYZING_LIMITATIONS = "analyzing_limitations"
    GENERATING_GAPS = "generating_gaps"
    VERIFYING_GAPS = "verifying_gaps"
    SUGGESTING_DIRECTIONS = "suggesting_directions"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStep(str, Enum):
    """Individual steps in the research workflow."""
    QUERY_VALIDATION = "query_validation"
    PAPER_DISCOVERY = "paper_discovery"
    DEDUPLICATION = "deduplication"
    PDF_ACQUISITION = "pdf_acquisition"
    PDF_PROCESSING = "pdf_processing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    PAPER_CARD_EXTRACTION = "paper_card_extraction"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    EVIDENCE_VERIFICATION = "evidence_verification"
    METHODOLOGY_COMPARISON = "methodology_comparison"
    DATASET_COMPARISON = "dataset_comparison"
    RESULTS_COMPARISON = "results_comparison"
    LIMITATION_EXTRACTION = "limitation_extraction"
    LIMITATION_CLUSTERING = "limitation_clustering"
    GAP_GENERATION = "gap_generation"
    COUNTEREVIDENCE_SEARCH = "counterevidence_search"
    GAP_VERIFICATION = "gap_verification"
    DIRECTION_SUGGESTION = "direction_suggestion"


class StepResult(BaseModel):
    """Result of a single workflow step."""
    step: WorkflowStep = Field(..., description="The workflow step")
    status: str = Field(..., description="Step status (success/failed/skipped)")
    
    # Output data (step-specific)
    output_summary: str = Field(default="", description="Summary of step output")
    items_processed: int = Field(default=0, ge=0, description="Number of items processed")
    items_failed: int = Field(default=0, ge=0, description="Number of items that failed")
    
    # Error handling
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    
    # Timing
    started_at: Optional[datetime] = Field(default=None, description="Step start time")
    completed_at: Optional[datetime] = Field(default=None, description="Step completion time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "step": "paper_discovery",
                "status": "success",
                "output_summary": "Found 25 papers from OpenAlex and Semantic Scholar",
                "items_processed": 25,
                "items_failed": 0
            }
        }


class ResearchJob(BaseModel):
    """Complete research job/workflow state."""
    job_id: str = Field(..., description="Unique job identifier")
    
    # Query information
    topic: str = Field(..., min_length=10, description="Research topic/question")
    subquestions: Optional[List[str]] = Field(default=None, description="Sub-questions")
    year_min: Optional[int] = Field(default=None, description="Minimum publication year")
    year_max: Optional[int] = Field(default=None, description="Maximum publication year")
    conferences: Optional[List[str]] = Field(default=None, description="Target conferences")
    max_papers: int = Field(default=50, ge=5, le=500, description="Maximum papers to analyze")
    
    # Workflow state
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING, 
                                   description="Current workflow status")
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0,
                                       description="Overall progress percentage")
    current_step: Optional[WorkflowStep] = Field(default=None,
                                                 description="Currently executing step")
    
    # Step results
    step_results: Dict[str, StepResult] = Field(default_factory=dict,
                                                description="Results for each completed step")
    
    # Results references
    discovered_paper_ids: List[str] = Field(default_factory=list,
                                            description="IDs of discovered papers")
    processed_paper_ids: List[str] = Field(default_factory=list,
                                           description="IDs of successfully processed papers")
    failed_paper_ids: List[str] = Field(default_factory=list,
                                        description="IDs of papers that failed processing")
    
    # Result references (populated as workflow progresses)
    paper_card_ids: List[str] = Field(default_factory=list, description="Generated PaperCard IDs")
    evidence_ids: List[str] = Field(default_factory=list, description="Generated evidence IDs")
    comparison_id: Optional[str] = Field(default=None, description="Generated comparison ID")
    limitation_cluster_ids: List[str] = Field(default_factory=list, 
                                              description="Generated limitation cluster IDs")
    gap_candidate_ids: List[str] = Field(default_factory=list,
                                         description="Generated gap candidate IDs")
    verification_ids: List[str] = Field(default_factory=list,
                                        description="Generated verification IDs")
    direction_ids: List[str] = Field(default_factory=list,
                                     description="Generated research direction IDs")
    
    # Error tracking
    errors: List[str] = Field(default_factory=list, description="Accumulated error messages")
    warnings: List[str] = Field(default_factory=list, description="Accumulated warnings")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Job creation time")
    started_at: Optional[datetime] = Field(default=None, description="Workflow start time")
    updated_at: Optional[datetime] = Field(default=None, description="Last update time")
    completed_at: Optional[datetime] = Field(default=None, description="Workflow completion time")
    
    # Resumability
    can_resume: bool = Field(default=False, description="Whether job can be resumed if interrupted")
    checkpoint_data: Optional[Dict[str, Any]] = Field(default=None,
                                                      description="Checkpoint data for resumption")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_001",
                "topic": "Parameter-efficient fine-tuning methods for LLMs",
                "status": "processing_papers",
                "progress_percentage": 45.0,
                "current_step": "pdf_processing",
                "discovered_paper_ids": ["paper_001", "paper_002"],
                "processed_paper_ids": ["paper_001"],
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class JobProgress(BaseModel):
    """Progress information for a research job."""
    job_id: str = Field(..., description="Job identifier")
    status: WorkflowStatus = Field(..., description="Current status")
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    current_step: Optional[WorkflowStep] = Field(default=None, description="Current step")
    completed_steps: List[str] = Field(default_factory=list, description="Completed step names")
    remaining_steps: List[str] = Field(default_factory=list, description="Remaining step names")
    estimated_time_remaining: Optional[int] = Field(default=None, ge=0,
                                                    description="Estimated seconds remaining")
    items_completed: int = Field(default=0, ge=0, description="Items completed")
    items_total: int = Field(default=0, ge=0, description="Total items to process")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_001",
                "status": "processing_papers",
                "progress_percentage": 60.0,
                "current_step": "pdf_processing",
                "completed_steps": ["paper_discovery", "deduplication"],
                "remaining_steps": ["evidence_extraction", "comparison"],
                "items_completed": 15,
                "items_total": 25
            }
        }


class JobSummary(BaseModel):
    """Summary of a completed research job."""
    job_id: str = Field(..., description="Job identifier")
    topic: str = Field(..., description="Research topic")
    status: WorkflowStatus = Field(..., description="Final status")
    
    # Statistics
    papers_discovered: int = Field(default=0, ge=0, description="Papers discovered")
    papers_processed: int = Field(default=0, ge=0, description="Papers successfully processed")
    papers_failed: int = Field(default=0, ge=0, description="Papers failed")
    
    gap_candidates_found: int = Field(default=0, ge=0, description="Gap candidates identified")
    research_directions: int = Field(default=0, ge=0, description="Research directions suggested")
    
    # Duration
    total_duration_seconds: Optional[float] = Field(default=None, ge=0,
                                                    description="Total workflow duration")
    
    # Access to results
    has_paper_cards: bool = Field(default=False, description="PaperCards available")
    has_evidence: bool = Field(default=False, description="Evidence available")
    has_comparison: bool = Field(default=False, description="Comparison available")
    has_limitations: bool = Field(default=False, description="Limitations available")
    has_gaps: bool = Field(default=False, description="Gap analysis available")
    has_directions: bool = Field(default=False, description="Directions available")
    
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_001",
                "topic": "Parameter-efficient fine-tuning",
                "status": "completed",
                "papers_discovered": 25,
                "papers_processed": 22,
                "gap_candidates_found": 5,
                "research_directions": 3,
                "total_duration_seconds": 1847.5
            }
        }

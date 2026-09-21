"""Pydantic schemas for the AI Research Gap Analysis Agent."""

from .query import (
    PaperSource,
    ResearchQuery,
    UploadedPaper,
    PaperMetadata,
    PaperCard,
    ChunkData,
)

from .evidence import (
    VerificationStatus,
    EvidenceType,
    SourceClaim,
    Evidence,
    EntailmentCheck,
)

from .comparison import (
    LimitationCategory,
    MethodologyComparison,
    DatasetComparison,
    ResultsComparison,
    LimitationExtraction,
    LimitationCluster,
    PaperComparisonMatrix,
)

from .gaps import (
    GapCoverageStatus,
    GapCategory,
    Counterevidence,
    GapCandidate,
    GapVerification,
    ResearchDirection,
    GapAnalysisResult,
)

from .workflow import (
    WorkflowStatus,
    WorkflowStep,
    StepResult,
    ResearchJob,
    JobProgress,
    JobSummary,
)

__all__ = [
    # Query schemas
    "PaperSource",
    "ResearchQuery",
    "UploadedPaper",
    "PaperMetadata",
    "PaperCard",
    "ChunkData",
    # Evidence schemas
    "VerificationStatus",
    "EvidenceType",
    "SourceClaim",
    "Evidence",
    "EntailmentCheck",
    # Comparison schemas
    "LimitationCategory",
    "MethodologyComparison",
    "DatasetComparison",
    "ResultsComparison",
    "LimitationExtraction",
    "LimitationCluster",
    "PaperComparisonMatrix",
    # Gap schemas
    "GapCoverageStatus",
    "GapCategory",
    "Counterevidence",
    "GapCandidate",
    "GapVerification",
    "ResearchDirection",
    "GapAnalysisResult",
    # Workflow schemas
    "WorkflowStatus",
    "WorkflowStep",
    "StepResult",
    "ResearchJob",
    "JobProgress",
    "JobSummary",
]
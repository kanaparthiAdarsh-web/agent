"""Storage schemas for the literature review system."""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class Provider(str, Enum):
    OPENALEX = "openalex"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ARXIV = "arxiv"
    CROSSREF = "crossref"
    UPLOADED = "uploaded"
    TEST = "test"


class PaperStatus(str, Enum):
    DISCOVERED = "discovered"
    RETRIEVED = "retrieved"
    PROCESSED = "processed"
    FAILED = "failed"


class EvidenceVerificationState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class Paper(BaseModel):
    doi: str
    title: str
    authors: List[str] = []
    abstract: Optional[str] = None
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    full_text_url: Optional[str] = None
    provider: Provider | str = Provider.OPENALEX
    provider_id: str = ""
    status: PaperStatus = PaperStatus.DISCOVERED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    id: str
    paper_doi: str
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PaperCard(BaseModel):
    doi: str
    title: str
    summary: str
    key_findings: List[str] = []
    methodology: Optional[str] = None
    datasets: List[str] = []
    metrics: List[str] = []
    limitations: List[str] = []
    terminology: Dict[str, str] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Evidence(BaseModel):
    id: str
    paper_doi: str
    content: str
    quote: str
    context: str
    verification_state: EvidenceVerificationState | str = EvidenceVerificationState.UNVERIFIED
    confidence: float = 0.0
    provenance: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Comparison(BaseModel):
    id: str
    paper1_doi: str
    paper2_doi: str
    similarity_score: float
    methodology_comparison: Optional[str] = None
    dataset_comparison: Optional[str] = None
    metric_comparison: Optional[str] = None
    result_comparison: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Limitation(BaseModel):
    id: str
    paper_doi: str
    text: str
    category: str
    original_text: str
    evidence_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Gap(BaseModel):
    id: str
    description: str
    supporting_evidence: List[str] = []
    related_limitations: List[str] = []
    counterevidence: List[str] = []
    verified: bool = False
    coverage: float = 0.0
    uncertainty_expressed: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchDirection(BaseModel):
    id: str
    description: str
    methodology: Optional[str] = None
    datasets: List[str] = []
    feasibility_notes: Optional[str] = None
    supporting_evidence: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Job(BaseModel):
    id: str
    research_question: str
    status: str
    progress: float = 0.0
    total_papers: int = 0
    processed_papers: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

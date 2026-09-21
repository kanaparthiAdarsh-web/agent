"""Pydantic schemas for research query and paper-related models."""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class PaperSource(str, Enum):
    """Source of paper metadata."""
    OPENALEX = "openalex"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ARXIV = "arxiv"
    CROSSREF = "crossref"
    UPLOADED = "uploaded"


class ResearchQuery(BaseModel):
    """Research topic/question for analysis."""
    topic: str = Field(..., min_length=10, max_length=2000, description="Research topic or question")
    subquestions: Optional[List[str]] = Field(default=None, description="Optional sub-questions to guide analysis")
    year_min: Optional[int] = Field(default=2018, ge=2000, le=2030, description="Minimum publication year")
    year_max: Optional[int] = Field(default=None, ge=2000, le=2030, description="Maximum publication year")
    conferences: Optional[List[str]] = Field(default=None, description="Target conferences/venues")
    max_papers: int = Field(default=50, ge=5, le=500, description="Maximum papers to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Efficient fine-tuning methods for large language models",
                "subquestions": [
                    "What are the most parameter-efficient methods?",
                    "How do they compare on downstream tasks?"
                ],
                "year_min": 2020,
                "max_papers": 30
            }
        }


class UploadedPaper(BaseModel):
    """User-uploaded paper for analysis."""
    title: str = Field(..., min_length=1, description="Paper title")
    authors: Optional[List[str]] = Field(default=None, description="Paper authors")
    year: Optional[int] = Field(default=None, ge=1900, le=2030, description="Publication year")
    venue: Optional[str] = Field(default=None, description="Publication venue")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv identifier")
    url: Optional[HttpUrl] = Field(default=None, description="Paper URL")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "authors": ["Edward Hu", "Yelong Shen", "et al."],
                "year": 2021,
                "venue": "ICLR",
                "doi": "10.48550/arXiv.2106.09685"
            }
        }


class PaperMetadata(BaseModel):
    """Normalized paper metadata from discovery sources."""
    id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="Author names")
    year: Optional[int] = Field(default=None, description="Publication year")
    venue: Optional[str] = Field(default=None, description="Publication venue/conference")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv identifier")
    url: Optional[str] = Field(default=None, description="Canonical URL")
    pdf_url: Optional[str] = Field(default=None, description="Direct PDF download URL")
    abstract: Optional[str] = Field(default=None, description="Paper abstract")
    source: PaperSource = Field(..., description="Source of this metadata")
    citation_count: Optional[int] = Field(default=None, ge=0, description="Citation count if available")
    open_access: bool = Field(default=False, description="Whether paper is open access")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "paper_001",
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "authors": ["Edward Hu", "Yelong Shen"],
                "year": 2021,
                "venue": "ICLR 2022",
                "doi": "10.48550/arXiv.2106.09685",
                "source": "openalex",
                "open_access": True
            }
        }


class PaperCard(BaseModel):
    """Structured understanding of a paper extracted from full text."""
    paper_id: str = Field(..., description="Reference to paper metadata")
    
    # Core research elements
    research_problem: Optional[str] = Field(default=None, description="Problem the paper addresses")
    research_question: Optional[str] = Field(default=None, description="Specific research question(s)")
    
    # Methodology
    methodology: Optional[str] = Field(default=None, description="Research methodology/approach")
    models: List[str] = Field(default_factory=list, description="Models/architectures used")
    datasets: List[str] = Field(default_factory=list, description="Datasets used in experiments")
    evaluation_metrics: List[str] = Field(default_factory=list, description="Metrics used for evaluation")
    
    # Results
    key_results: List[str] = Field(default_factory=list, description="Key experimental results")
    contributions: List[str] = Field(default_factory=list, description="Main contributions claimed")
    
    # Critical analysis
    limitations: List[str] = Field(default_factory=list, description="Limitations acknowledged by authors")
    future_work: List[str] = Field(default_factory=list, description="Future work suggested by authors")
    
    # Terminology
    terminology: Dict[str, str] = Field(default_factory=dict, description="Key terms and definitions")
    
    # Evidence tracking
    evidence_references: List[Dict[str, Any]] = Field(default_factory=list, 
                                                     description="References to evidence chunks supporting extractions")
    
    # Processing metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this card was created")
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in extraction quality")
    
    class Config:
        json_schema_extra = {
            "example": {
                "paper_id": "paper_001",
                "research_problem": "Fine-tuning LLMs requires significant computational resources",
                "methodology": "Low-rank decomposition of weight matrices",
                "models": ["RoBERTa", "GPT-2", "GPT-3"],
                "datasets": ["GLUE", "E2E NLG", "WikiSQL"],
                "key_results": ["Comparable performance to full fine-tuning with 10000x fewer parameters"],
                "contributions": ["LoRA method", "Extensive empirical evaluation"],
                "extraction_confidence": 0.85
            }
        }


class ChunkData(BaseModel):
    """Chunk of text from a paper with metadata."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    paper_id: str = Field(..., description="Reference to parent paper")
    text: str = Field(..., description="Chunk text content")
    section_type: Optional[str] = Field(default=None, description="Section type (abstract/intro/methods/etc)")
    section_title: Optional[str] = Field(default=None, description="Section title if available")
    page_numbers: Optional[List[int]] = Field(default=None, description="Page numbers containing this chunk")
    position_in_section: Optional[int] = Field(default=None, description="Position within section")
    is_partial: bool = Field(default=False, description="Whether chunk is partial (cut off)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001_001",
                "paper_id": "paper_001",
                "text": "We propose LoRA, a low-rank adaptation method...",
                "section_type": "methodology",
                "page_numbers": [3, 4]
            }
        }

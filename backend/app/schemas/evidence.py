"""Pydantic schemas for evidence-related models."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class VerificationStatus(str, Enum):
    """Status of evidence verification."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class EvidenceType(str, Enum):
    """Type of evidence extracted from papers."""
    CLAIM = "claim"
    RESULT = "result"
    METHODOLOGY = "methodology"
    DATASET = "dataset"
    METRIC = "metric"
    LIMITATION = "limitation"
    CONTRIBUTION = "contribution"
    FUTURE_WORK = "future_work"
    DEFINITION = "definition"


class SourceClaim(BaseModel):
    """A claim extracted from a paper with source reference."""
    claim_id: str = Field(..., description="Unique claim identifier")
    paper_id: str = Field(..., description="Reference to source paper")
    chunk_id: Optional[str] = Field(default=None, description="Reference to source chunk")
    
    claim_type: EvidenceType = Field(..., description="Type of this claim")
    content: str = Field(..., min_length=1, description="The claim content")
    
    # Source verification
    exact_quote: Optional[str] = Field(default=None, description="Exact quote from paper if available")
    page_numbers: Optional[List[int]] = Field(default=None, description="Page numbers where claim appears")
    section: Optional[str] = Field(default=None, description="Section where claim appears")
    
    # Confidence assessment
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in extraction accuracy")
    entailment_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, 
                                               description="Entailment score if verified")
    
    # Verification status
    verification_status: VerificationStatus = Field(default=VerificationStatus.UNVERIFIED,
                                                    description="Current verification status")
    verification_notes: Optional[str] = Field(default=None, description="Notes on verification process")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When claim was extracted")
    
    class Config:
        json_schema_extra = {
            "example": {
                "claim_id": "claim_001",
                "paper_id": "paper_001",
                "chunk_id": "chunk_001_003",
                "claim_type": "result",
                "content": "LoRA achieves comparable performance to full fine-tuning",
                "exact_quote": "LoRA performs on par or better than full fine-tuning while using 10,000x fewer parameters",
                "page_numbers": [7],
                "section": "Experiments",
                "confidence": 0.92,
                "verification_status": "verified"
            }
        }


class Evidence(BaseModel):
    """Structured evidence with provenance and verification."""
    evidence_id: str = Field(..., description="Unique evidence identifier")
    research_topic: str = Field(..., description="Research topic this evidence relates to")
    
    claims: List[SourceClaim] = Field(default_factory=list, 
                                       description="Claims supporting this evidence")
    
    # Aggregated information
    summary: str = Field(..., min_length=1, description="Summary of the evidence")
    category: str = Field(..., description="Evidence category (e.g., methodology, results)")
    
    # Quality assessment
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0, 
                                 description="Overall quality/confidence score")
    source_count: int = Field(default=0, ge=0, description="Number of distinct sources")
    
    # Verification
    verification_status: VerificationStatus = Field(default=VerificationStatus.UNVERIFIED,
                                                    description="Overall verification status")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When evidence was collected")
    updated_at: Optional[datetime] = Field(default=None, description="Last update time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "evidence_id": "evidence_001",
                "research_topic": "Parameter-efficient fine-tuning",
                "summary": "Multiple papers demonstrate LoRA's effectiveness across models",
                "category": "results",
                "quality_score": 0.88,
                "source_count": 5,
                "verification_status": "verified"
            }
        }


class EntailmentCheck(BaseModel):
    """Result of entailment checking between claim and source."""
    claim_id: str = Field(..., description="Reference to claim")
    source_text: str = Field(..., description="Source text used for verification")
    
    entails: bool = Field(..., description="Whether source entails the claim")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in entailment decision")
    explanation: Optional[str] = Field(default=None, description="Explanation of the decision")
    
    class Config:
        json_schema_extra = {
            "example": {
                "claim_id": "claim_001",
                "source_text": "Our experiments show LoRA matches full fine-tuning...",
                "entails": True,
                "confidence": 0.94,
                "explanation": "Source explicitly states the claimed result"
            }
        }

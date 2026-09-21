"""Custom exceptions for the AI Research Gap Analysis Agent."""


class ResearchAgentError(Exception):
    """Base exception for all research agent errors."""
    
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            **self.details
        }


# Discovery errors
class DiscoveryError(ResearchAgentError):
    """Error during paper discovery."""
    pass


class ProviderUnavailableError(DiscoveryError):
    """A discovery provider is unavailable."""
    pass


class PaperNotFoundError(DiscoveryError):
    """Requested paper was not found."""
    pass


class PDFDownloadError(DiscoveryError):
    """Failed to download paper PDF."""
    pass


# Processing errors
class ProcessingError(ResearchAgentError):
    """Error during paper processing."""
    pass


class PDFProcessingError(ProcessingError):
    """Error processing PDF file."""
    pass


class ChunkingError(ProcessingError):
    """Error during text chunking."""
    pass


class ExtractionError(ProcessingError):
    """Error during information extraction."""
    pass


# Evidence errors
class EvidenceError(ResearchAgentError):
    """Error during evidence collection."""
    pass


class VerificationError(EvidenceError):
    """Error during evidence verification."""
    pass


# Comparison errors
class ComparisonError(ResearchAgentError):
    """Error during paper comparison."""
    pass


# Limitation errors
class LimitationError(ResearchAgentError):
    """Error during limitation analysis."""
    pass


# Gap analysis errors
class GapAnalysisError(ResearchAgentError):
    """Error during gap analysis."""
    pass


# Workflow errors
class WorkflowError(ResearchAgentError):
    """Error during workflow execution."""
    pass


class WorkflowCancelledError(WorkflowError):
    """Workflow was cancelled by user."""
    pass


class WorkflowTimeoutError(WorkflowError):
    """Workflow exceeded maximum execution time."""
    pass


# Storage errors
class StorageError(ResearchAgentError):
    """Error during data storage/retrieval."""
    pass


class PaperNotFoundError(StorageError):
    """Paper not found in storage."""
    pass


class IndexNotFoundError(StorageError):
    """Retrieval index not found."""
    pass


# Validation errors
class ValidationError(ResearchAgentError):
    """Input validation error."""
    pass


class QueryValidationError(ValidationError):
    """Research query validation error."""
    pass


class FileValidationError(ValidationError):
    """File upload validation error."""
    pass


# Security errors
class SecurityError(ResearchAgentError):
    """Security-related error."""
    pass


class PromptInjectionError(SecurityError):
    """Potential prompt injection detected."""
    pass


class UploadSecurityError(SecurityError):
    """Security issue with file upload."""
    pass


# LLM errors
class LLMError(ResearchAgentError):
    """Error during LLM interaction."""
    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""
    pass


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""
    pass


class StructuredOutputError(LLMError):
    """Failed to parse structured output from LLM."""
    pass

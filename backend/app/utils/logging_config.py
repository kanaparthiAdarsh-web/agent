"""Logging configuration for the AI Research Gap Analysis Agent."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_type: str = "detailed",
) -> None:
    """
    Configure root logger and common handlers.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_type: Format style ("simple" or "detailed")
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Define formatters
    if format_type == "detailed":
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            "%(levelname)s | %(name)s | %(message)s"
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(file_handler)
    
    # Reduce noise from verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Security-focused logger for sensitive operations
security_logger = logging.getLogger("security")


def log_security_event(
    event_type: str,
    details: dict,
    level: str = "WARNING",
) -> None:
    """
    Log a security-related event.
    
    Args:
        event_type: Type of security event
        details: Event details (will be sanitized)
        level: Logging level
    """
    # Sanitize sensitive information
    sanitized_details = _sanitize_for_logging(details)
    
    log_method = getattr(security_logger, level.lower(), security_logger.warning)
    log_method(f"[{event_type}] {sanitized_details}")


def _sanitize_for_logging(data: dict) -> dict:
    """Remove sensitive information from data before logging."""
    sensitive_keys = [
        "api_key", "apikey", "token", "secret", "password", "credential",
        "authorization", "auth", "private_key", "access_token"
    ]
    
    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_for_logging(value)
        else:
            sanitized[key] = value
    
    return sanitized


# Context-aware logging for workflow tracking
class WorkflowLogger:
    """Logger that includes workflow/job context in all messages."""
    
    def __init__(self, job_id: str, component: str):
        """
        Initialize workflow logger.
        
        Args:
            job_id: Unique job identifier
            component: Component name (e.g., 'discovery', 'extraction')
        """
        self.job_id = job_id
        self.component = component
        self.logger = logging.getLogger(f"workflow.{component}")
    
    def _format_message(self, message: str) -> str:
        """Add context to message."""
        return f"[Job:{self.job_id}] {message}"
    
    def info(self, message: str, **kwargs) -> None:
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self.logger.error(self._format_message(message), **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        self.logger.debug(self._format_message(message), **kwargs)
    
    def progress(self, current: int, total: int, item: str = "") -> None:
        """Log progress update."""
        percentage = (current / total * 100) if total > 0 else 0
        self.info(f"Progress: {percentage:.1f}% ({current}/{total}) - {item}")

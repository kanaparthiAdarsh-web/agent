"""Limitation extraction wrapper for backward compatibility."""

from .analyzer import LimitationAnalyzer, extract_limitations, cluster_limitations

# Re-export for backward compatibility
LimitationExtractor = LimitationAnalyzer

__all__ = ["LimitationExtractor", "extract_limitations", "cluster_limitations"]

"""Limitation clustering wrapper for backward compatibility."""

from .analyzer import LimitationAnalyzer, cluster_limitations

# Re-export for backward compatibility  
LimitationClustering = LimitationAnalyzer

__all__ = ["LimitationClustering", "cluster_limitations"]

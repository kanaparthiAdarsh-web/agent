"""Gap finder wrapper for backward compatibility."""

from .analyzer import GapAnalyzer, analyze_gaps

# Re-export for backward compatibility
GapFinder = GapAnalyzer

__all__ = ["GapFinder", "analyze_gaps"]

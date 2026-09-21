"""Gap finder wrapper for backward compatibility."""

from .analyzer import GapAnalyzer, analyze_gaps

# Add find_gaps method for backward compatibility with workflow engine
def _find_gaps_compat(self, papers, limitations):
    """Compatibility wrapper: call analyze_gaps instead."""
    return self.analyze_gaps(papers, limitations)

# Attach the compatibility method
GapAnalyzer.find_gaps = _find_gaps_compat

# Re-export for backward compatibility
GapFinder = GapAnalyzer

__all__ = ["GapFinder", "analyze_gaps"]

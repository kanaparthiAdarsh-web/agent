"""PaperCard Generator - Generates structured paper summaries from content."""
from typing import List, Dict, Any
from app.storage.schemas import Paper, PaperCard


class PaperCardGenerator:
    """Generate PaperCards from paper content."""
    
    def generate_papercard(self, paper: Paper) -> PaperCard:
        """
        Generate a PaperCard from a paper.
        
        Args:
            paper: Paper object with content
            
        Returns:
            PaperCard with extracted information
        """
        # Extract key information from abstract/content
        text = paper.abstract or ""
        
        # Simple extraction based on keywords
        summary = self._generate_summary(text)
        key_findings = self._extract_key_findings(text)
        methodology = self._extract_methodology(text)
        datasets = self._extract_datasets(text)
        metrics = self._extract_metrics(text)
        limitations = self._extract_limitations(text)
        terminology = self._extract_terminology(text)
        
        return PaperCard(
            doi=paper.doi,
            title=paper.title,
            summary=summary,
            key_findings=key_findings,
            methodology=methodology,
            datasets=datasets,
            metrics=metrics,
            limitations=limitations,
            terminology=terminology
        )
    
    def _generate_summary(self, text: str) -> str:
        """Generate a brief summary from text."""
        if not text:
            return "No summary available"
        
        # Take first few sentences as summary
        sentences = text.split('.')
        summary_sentences = []
        char_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                summary_sentences.append(sentence)
                char_count += len(sentence)
                if char_count >= 200 or len(summary_sentences) >= 3:
                    break
        
        return '. '.join(summary_sentences) + '.' if summary_sentences else text[:200]
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from text."""
        findings = []
        text_lower = text.lower()
        
        # Look for result indicators
        result_patterns = [
            "achieves", "demonstrates", "shows", "improves", 
            "outperforms", "results", "accuracy", "performance"
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(pattern in sentence.lower() for pattern in result_patterns):
                if len(sentence) > 20 and len(sentence) < 300:
                    findings.append(sentence + '.')
        
        return findings[:5]  # Limit to top 5
    
    def _extract_methodology(self, text: str) -> str:
        """Extract methodology description."""
        method_keywords = [
            "propose", "method", "approach", "architecture", 
            "model", "algorithm", "technique", "framework"
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in method_keywords):
                return sentence + '.'
        
        return None
    
    def _extract_datasets(self, text: str) -> List[str]:
        """Extract dataset mentions."""
        datasets = []
        
        # Common dataset patterns
        dataset_patterns = [
            "dataset", "benchmark", "corpus", "data"
        ]
        
        sentences = text.split(',')
        for segment in sentences:
            segment = segment.strip()
            if any(pattern in segment.lower() for pattern in dataset_patterns):
                # Try to extract potential dataset name
                words = segment.split()
                if len(words) <= 8:
                    datasets.append(segment.rstrip('.'))
        
        return datasets[:5]
    
    def _extract_metrics(self, text: str) -> List[str]:
        """Extract evaluation metrics."""
        metrics = []
        
        metric_keywords = [
            "accuracy", "precision", "recall", "f1", "f1-score",
            "auc", "roc", "mse", "rmse", "bleu", "rouge"
        ]
        
        text_lower = text.lower()
        for metric in metric_keywords:
            if metric in text_lower:
                # Find the context around the metric
                idx = text_lower.find(metric)
                start = max(0, idx - 30)
                end = min(len(text), idx + len(metric) + 30)
                context = text[start:end].strip()
                if len(context) > 10:
                    metrics.append(context)
        
        return metrics[:5]
    
    def _extract_limitations(self, text: str) -> List[str]:
        """Extract limitation statements."""
        limitations = []
        
        limitation_keywords = [
            "limitation", "however", "although", "but", 
            "challenge", "difficulty", "restricted", "limited"
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in limitation_keywords):
                if len(sentence) > 20 and len(sentence) < 300:
                    limitations.append(sentence + '.')
        
        return limitations[:5]
    
    def _extract_terminology(self, text: str) -> Dict[str, str]:
        """Extract key terminology and definitions."""
        terminology = {}
        
        # Simple pattern: look for "X is Y" or "X refers to Y"
        definition_patterns = [
            r'(\w+)\s+is\s+(?:a|an|the)\s+(\w+(?:\s+\w+)*)',
            r'(\w+)\s+refers\s+to\s+(\w+(?:\s+\w+)*)'
        ]
        
        import re
        for pattern in definition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match[0]) > 3 and len(match[1]) > 3:
                    terminology[match[0]] = match[1]
        
        return dict(list(terminology.items())[:10])

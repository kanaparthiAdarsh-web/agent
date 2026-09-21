"""Paper understanding and PaperCard generation."""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel

from ..schemas import PaperCard, PaperMetadata, ChunkData
from ..exceptions import ExtractionError, ProcessingError
from ..llm.client_factory import invoke_with_structured_output
from ..llm.config import get_llm_config

logger = logging.getLogger(__name__)


PAPER_CARD_EXTRACTION_PROMPT = """You are an expert research paper analyzer. Extract structured information from the following paper text.

Return your response as a valid JSON object with these fields:
- research_problem: string - The main problem the paper addresses
- research_question: string - Specific research question(s)
- methodology: string - Research methodology/approach
- models: array of strings - Models/architectures used
- datasets: array of strings - Datasets used in experiments
- evaluation_metrics: array of strings - Metrics used for evaluation
- key_results: array of strings - Key experimental results (include specific numbers when available)
- contributions: array of strings - Main contributions claimed by authors
- limitations: array of strings - Limitations acknowledged by authors
- future_work: array of strings - Future work suggested by authors
- terminology: object - Key terms and their definitions (string -> string)

IMPORTANT RULES:
1. Only extract information that is explicitly present in the text
2. Do not fabricate or infer information not in the source
3. For key_results, include specific metrics/numbers when mentioned
4. For limitations, focus on what authors explicitly acknowledge
5. If a field cannot be filled from the text, use null or empty array
6. Preserve exact terminology used in the paper

Paper text:
{paper_text}

Return ONLY the JSON object, no other text."""


TERMINOLOGY_EXTRACTION_PROMPT = """Extract key technical terms and their definitions from this text.

For each term, provide:
- The term itself
- A concise definition based on how it's used in the text

Return as JSON object mapping terms to definitions.

Text:
{text}

Return ONLY valid JSON."""


class PaperAnalyzer:
    """Analyzes papers and generates structured PaperCards."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the paper analyzer.
        
        Args:
            llm: LLM client for extraction (uses default if not provided)
        """
        self.llm = llm
        self._config = None
    
    def _get_llm(self) -> BaseChatModel:
        """Get LLM client, initializing if needed."""
        if self.llm is None:
            from ..llm.client_factory import create_llm_client
            if self._config is None:
                self._config = get_llm_config()
            self.llm = create_llm_client(self._config)
        return self.llm
    
    async def analyze_paper(
        self,
        paper_metadata: PaperMetadata,
        chunks: List[ChunkData],
        full_text: Optional[str] = None,
    ) -> PaperCard:
        """
        Analyze a paper and generate a PaperCard.
        
        Args:
            paper_metadata: Paper metadata
            chunks: Text chunks from the paper
            full_text: Optional full text (if available)
        
        Returns:
            Generated PaperCard with extracted information
        
        Raises:
            ExtractionError: If extraction fails
        """
        # Prepare text for analysis
        if full_text:
            # Use full text if available, but limit length
            analysis_text = self._truncate_text(full_text, max_tokens=8000)
        else:
            # Combine chunks, prioritizing important sections
            analysis_text = self._combine_chunks(chunks)
        
        if not analysis_text.strip():
            raise ExtractionError("No text available for analysis")
        
        try:
            llm = self._get_llm()
            
            # Extract structured information
            prompt = PAPER_CARD_EXTRACTION_PROMPT.format(paper_text=analysis_text)
            result = await invoke_with_structured_output(llm, prompt)
            
            # Parse result into PaperCard
            paper_card = self._create_paper_card(paper_metadata, result, chunks)
            
            logger.info("Generated PaperCard for paper %s", paper_metadata.id)
            return paper_card
            
        except Exception as e:
            logger.error("Failed to analyze paper %s: %s", paper_metadata.id, e)
            raise ExtractionError(f"Failed to analyze paper: {e}")
    
    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """Truncate text to fit within token limit."""
        # Rough token estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Keep beginning and end for context
        chunk_size = max_chars // 2
        return text[:chunk_size] + "\n\n... [truncated] ...\n\n" + text[-chunk_size:]
    
    def _combine_chunks(self, chunks: List[ChunkData]) -> str:
        """Combine chunks strategically for analysis."""
        if not chunks:
            return ""
        
        # Sort chunks by section importance
        section_order = {
            "abstract": 0,
            "introduction": 1,
            "methodology": 2,
            "methods": 2,
            "experiments": 3,
            "results": 3,
            "discussion": 4,
            "conclusion": 5,
            "limitations": 6,
            "future work": 7,
        }
        
        def get_priority(chunk: ChunkData) -> int:
            section = (chunk.section_type or "").lower()
            return section_order.get(section, 10)
        
        sorted_chunks = sorted(chunks, key=get_priority)
        
        # Take most important chunks first
        selected_chunks = sorted_chunks[:50]  # Limit to 50 chunks
        
        texts = []
        for chunk in selected_chunks:
            header = f"[Section: {chunk.section_type or 'unknown'}]"
            texts.append(f"{header}\n{chunk.text}")
        
        return "\n\n".join(texts)
    
    def _create_paper_card(
        self,
        paper_metadata: PaperMetadata,
        extraction_result: Dict[str, Any],
        chunks: List[ChunkData],
    ) -> PaperCard:
        """Create PaperCard from extraction result."""
        # Ensure we have lists, not None
        def ensure_list(val: Any) -> List[str]:
            if val is None:
                return []
            if isinstance(val, list):
                return [str(v) for v in val if v]
            return []
        
        def ensure_dict(val: Any) -> Dict[str, str]:
            if val is None:
                return {}
            if isinstance(val, dict):
                return {str(k): str(v) for k, v in val.items() if k and v}
            return {}
        
        # Build evidence references
        evidence_references = []
        for i, chunk in enumerate(chunks[:20]):  # Reference first 20 chunks
            evidence_references.append({
                "chunk_id": chunk.chunk_id,
                "section": chunk.section_type,
                "page_numbers": chunk.page_numbers,
            })
        
        # Calculate confidence based on completeness
        fields = [
            extraction_result.get("research_problem"),
            extraction_result.get("methodology"),
            extraction_result.get("key_results"),
            extraction_result.get("contributions"),
        ]
        filled_fields = sum(1 for f in fields if f and (isinstance(f, str) or (isinstance(f, list) and len(f) > 0)))
        confidence = filled_fields / len(fields) if fields else 0.0
        
        return PaperCard(
            paper_id=paper_metadata.id,
            research_problem=extraction_result.get("research_problem"),
            research_question=extraction_result.get("research_question"),
            methodology=extraction_result.get("methodology"),
            models=ensure_list(extraction_result.get("models")),
            datasets=ensure_list(extraction_result.get("datasets")),
            evaluation_metrics=ensure_list(extraction_result.get("evaluation_metrics")),
            key_results=ensure_list(extraction_result.get("key_results")),
            contributions=ensure_list(extraction_result.get("contributions")),
            limitations=ensure_list(extraction_result.get("limitations")),
            future_work=ensure_list(extraction_result.get("future_work")),
            terminology=ensure_dict(extraction_result.get("terminology")),
            evidence_references=evidence_references,
            created_at=datetime.utcnow(),
            extraction_confidence=min(confidence, 1.0),
        )
    
    async def extract_terminology(self, text: str) -> Dict[str, str]:
        """
        Extract terminology from text.
        
        Args:
            text: Text to extract terminology from
        
        Returns:
            Dictionary mapping terms to definitions
        """
        try:
            llm = self._get_llm()
            prompt = TERMINOLOGY_EXTRACTION_PROMPT.format(text=text)
            result = await invoke_with_structured_output(llm, prompt)
            
            if isinstance(result, dict):
                return {str(k): str(v) for k, v in result.items() if k and v}
            return {}
            
        except Exception as e:
            logger.warning("Terminology extraction failed: %s", e)
            return {}


async def generate_paper_card(
    paper_metadata: PaperMetadata,
    chunks: List[ChunkData],
    full_text: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
) -> PaperCard:
    """
    Convenience function to generate a PaperCard.
    
    Args:
        paper_metadata: Paper metadata
        chunks: Text chunks from the paper
        full_text: Optional full text
        llm: Optional LLM client
    
    Returns:
        Generated PaperCard
    """
    analyzer = PaperAnalyzer(llm=llm)
    return await analyzer.analyze_paper(paper_metadata, chunks, full_text)

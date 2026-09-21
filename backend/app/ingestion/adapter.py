"""
PDF Ingestion Adapter - Wires existing ingestion modules into the workflow.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from app.ingestion.pdf import PDFLoader
from app.ingestion.chunking.basic import BasicChunker
from app.ingestion.chunking.semantic import SemanticChunker
from app.ingestion.chunking.structured import HybridStructuredChunker
from app.storage.schemas import Chunk, Paper


class PDFIngestionAdapter:
    """
    Adapter that integrates existing PDF ingestion and chunking modules
    into the research workflow.
    """
    
    def __init__(self, chunking_strategy: str = "hybrid_structured"):
        """
        Initialize adapter with specified chunking strategy.
        
        Args:
            chunking_strategy: One of 'basic', 'semantic', 'hybrid_structured'
        """
        self.pdf_loader = PDFLoader()
        
        if chunking_strategy == "basic":
            self.chunker = BasicChunker()
        elif chunking_strategy == "semantic":
            self.chunker = SemanticChunker()
        else:  # Default to hybrid structured for academic papers
            self.chunker = HybridStructuredChunker()
    
    def ingest_pdf(self, pdf_path: str, paper: Paper) -> Dict[str, Any]:
        """
        Ingest a PDF file and produce structured chunks.
        
        Args:
            pdf_path: Path to PDF file
            paper: Paper metadata object
            
        Returns:
            Dictionary containing:
                - chunks: List of Chunk objects
                - structure: Document structure (sections, paragraphs)
                - metadata: Processing metadata
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Load PDF with structure
        load_result = self.pdf_loader.load_pdf_with_structure(pdf_path_obj)
        
        # Extract chunks using configured strategy
        chunk_result = None
        if hasattr(self.chunker, 'chunk_with_structure'):
            # Use structure-aware chunking if available
            chunk_result = self.chunker.chunk_with_structure(
                text=load_result["text"],
                structure=load_result["structure"],
                metadata={
                    "paper_doi": paper.doi,
                    "title": paper.title,
                    "source": pdf_path
                }
            )
        elif hasattr(self.chunker, 'chunk'):
            # Fallback to basic chunking
            chunk_result = self.chunker.chunk(
                text=load_result["text"],
                metadata={
                    "paper_doi": paper.doi,
                    "title": paper.title,
                    "source": pdf_path
                }
            )
        
        # Handle different return types from chunkers
        chunk_data_list = []
        if isinstance(chunk_result, dict):
            chunk_data_list = chunk_result.get("chunks", [])
        elif isinstance(chunk_result, list):
            chunk_data_list = chunk_result
        
        # Convert chunks to schema objects
        chunks = []
        for i, chunk_data in enumerate(chunk_data_list):
            chunk = Chunk(
                id=str(uuid.uuid4()),
                paper_doi=paper.doi,
                content=chunk_data.get("text", ""),
                page_number=chunk_data.get("page", None),
                section_title=chunk_data.get("section_title", None),
                embedding=None,  # Will be populated by embedding generator
                created_at=datetime.utcnow()
            )
            chunks.append(chunk)
        
        return {
            "chunks": chunks,
            "structure": load_result["structure"],
            "pages": load_result["pages"],
            "metadata": {
                "paper_doi": paper.doi,
                "pdf_path": pdf_path,
                "num_chunks": len(chunks),
                "num_pages": len(load_result["pages"]),
                "processed_at": datetime.utcnow().isoformat(),
                "chunking_strategy": self.__class__.__name__
            }
        }
    
    def ingest_uploaded_pdf(
        self, 
        pdf_content: bytes, 
        filename: str, 
        research_question: str = ""
    ) -> Dict[str, Any]:
        """
        Ingest an uploaded PDF (e.g., from API upload).
        
        Args:
            pdf_content: Raw PDF bytes
            filename: Original filename
            research_question: Optional research question context
            
        Returns:
            Same structure as ingest_pdf
        """
        import tempfile
        import hashlib
        
        # Validate PDF signature
        if not pdf_content.startswith(b'%PDF'):
            raise ValueError("Invalid PDF file")
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(pdf_content).hexdigest()
        
        # Write to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name
        
        try:
            # Load PDF with structure
            load_result = self.pdf_loader.load_pdf_with_structure(Path(tmp_path))
            
            # Create minimal paper metadata
            paper = Paper(
                doi=f"uploaded:{content_hash[:16]}",
                title=filename or "Uploaded Paper",
                authors=[],
                abstract=None,
                provider="uploaded",
                provider_id=content_hash,
                status="discovered"
            )
            
            # Extract chunks
            if hasattr(self.chunker, 'chunk_with_structure'):
                chunk_result = self.chunker.chunk_with_structure(
                    text=load_result["text"],
                    structure=load_result["structure"],
                    metadata={
                        "paper_doi": paper.doi,
                        "title": paper.title,
                        "source": f"upload:{filename}",
                        "research_question": research_question
                    }
                )
            else:
                chunk_result = self.chunker.chunk(
                    text=load_result["text"],
                    metadata={
                        "paper_doi": paper.doi,
                        "title": paper.title,
                        "source": f"upload:{filename}"
                    }
                )
            
            # Convert to schema objects
            chunks = []
            for i, chunk_data in enumerate(chunk_result.get("chunks", [])):
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    paper_doi=paper.doi,
                    content=chunk_data.get("text", ""),
                    page_number=chunk_data.get("page", None),
                    section_title=chunk_data.get("section_title", None),
                    embedding=None,
                    created_at=datetime.utcnow()
                )
                chunks.append(chunk)
            
            return {
                "chunks": chunks,
                "structure": load_result["structure"],
                "pages": load_result["pages"],
                "paper": paper,
                "metadata": {
                    "filename": filename,
                    "content_hash": content_hash,
                    "num_chunks": len(chunks),
                    "num_pages": len(load_result["pages"]),
                    "processed_at": datetime.utcnow().isoformat(),
                    "chunking_strategy": self.__class__.__name__
                }
            }
        finally:
            # Cleanup temporary file
            import os
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
    
    def get_chunk_provenance(self, chunk: Chunk, structure: Dict) -> Dict[str, Any]:
        """
        Get detailed provenance information for a chunk.
        
        Args:
            chunk: Chunk object
            structure: Document structure from ingestion
            
        Returns:
            Provenance dictionary with section, paragraph, page info
        """
        provenance = {
            "page": chunk.page_number,
            "section": chunk.section_title,
            "char_start": None,
            "char_end": None
        }
        
        # Try to find matching paragraph in structure
        if structure and "paragraphs" in structure:
            for para in structure["paragraphs"]:
                if para.get("text", "") == chunk.content:
                    provenance["char_start"] = para.get("char_start")
                    provenance["char_end"] = para.get("char_end")
                    
                    # Find parent section
                    if "section_idx" in para:
                        section_idx = para["section_idx"]
                        if section_idx < len(structure.get("sections", [])):
                            provenance["section"] = structure["sections"][section_idx].get("title")
                            provenance["section_type"] = structure["sections"][section_idx].get("type")
                    break
        
        return provenance

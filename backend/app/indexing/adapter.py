"""
Indexing Adapter - Connects ingested chunks to existing retrieval infrastructure.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.documents import Document

from app.retrieval.semantic import SemanticRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.embeddings import EmbeddingGenerator
from app.storage.schemas import Chunk


class IndexingAdapter:
    """
    Adapter that manages FAISS and BM25 indexes for ingested paper chunks.
    """
    
    def __init__(self, index_dir: str = "indexes"):
        """
        Initialize indexing adapter.
        
        Args:
            index_dir: Directory for storing indexes
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding generator
        self.embedding_gen = EmbeddingGenerator()
        
        # Indexes (lazy-loaded)
        self._semantic_index: Optional[SemanticRetriever] = None
        self._bm25_index: Optional[BM25Retriever] = None
        self._hybrid_index: Optional[HybridRetriever] = None
        
        # Track which papers are indexed
        self.indexed_papers: set = set()
    
    @property
    def semantic_index(self) -> SemanticRetriever:
        """Get or create semantic index."""
        if self._semantic_index is None:
            index_path = self.index_dir / "faiss_index"
            self._semantic_index = SemanticRetriever(
                index_path=str(index_path),
                embedding_generator=self.embedding_gen,
                top_k=10
            )
        return self._semantic_index
    
    @property
    def bm25_index(self) -> BM25Retriever:
        """Get or create BM25 index."""
        if self._bm25_index is None:
            index_path = self.index_dir / "bm25_index"
            self._bm25_index = BM25Retriever(
                index_path=str(index_path),
                top_k=10
            )
        return self._bm25_index
    
    @property
    def hybrid_index(self) -> HybridRetriever:
        """Get or create hybrid index."""
        if self._hybrid_index is None:
            self._hybrid_index = HybridRetriever(
                semantic_retriever=self.semantic_index,
                bm25_retriever=self.bm25_index,
                top_k=10,
                method="rrf"  # Reciprocal Rank Fusion
            )
        return self._hybrid_index
    
    def index_chunks(self, chunks: List[Chunk], paper_doi: str) -> Dict[str, Any]:
        """
        Index chunks from a paper into both semantic and BM25 indexes.
        
        Args:
            chunks: List of Chunk objects
            paper_doi: DOI of the paper
            
        Returns:
            Indexing metadata
        """
        if not chunks:
            return {"status": "no_chunks", "indexed_count": 0}
        
        # Prepare documents for indexing
        documents = []
        chunk_metadata = []
        
        for chunk in chunks:
            # Create LangChain Document
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "paper_doi": chunk.paper_doi,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_id": chunk.id
                }
            )
            documents.append(doc)
            chunk_metadata.append({
                "id": chunk.id,
                "doi": chunk.paper_doi,
                "page": chunk.page_number,
                "section": chunk.section_title
            })
        
        # Generate embeddings for semantic index
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_gen.embed_texts(texts)
        
        # Add to semantic index
        try:
            self.semantic_index.add_documents_with_embeddings(documents, embeddings)
        except Exception as e:
            print(f"Warning: Failed to add to semantic index: {e}")
        
        # Add to BM25 index
        try:
            self.bm25_index.add_documents(documents)
        except Exception as e:
            print(f"Warning: Failed to add to BM25 index: {e}")
        
        # Mark paper as indexed
        self.indexed_papers.add(paper_doi)
        
        # Save index metadata
        self._save_index_metadata()
        
        return {
            "status": "success",
            "indexed_count": len(chunks),
            "paper_doi": paper_doi,
            "total_indexed_papers": len(self.indexed_papers)
        }
    
    def search_semantic(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters (e.g., {"paper_doi": "10.1234/test"})
            
        Returns:
            List of results with text, metadata, and scores
        """
        results = self.semantic_index.invoke(query, config={"top_k": top_k})
        
        # Format results
        formatted = []
        for doc in results:
            formatted.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', None),
                "retrieval_type": "semantic"
            })
        
        return formatted
    
    def search_bm25(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search using BM25 lexical matching.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of results with text, metadata, and scores
        """
        results = self.bm25_index.invoke(query, config={"top_k": top_k})
        
        # Format results
        formatted = []
        for doc in results:
            formatted.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', None),
                "retrieval_type": "bm25"
            })
        
        return formatted
    
    def search_hybrid(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search using hybrid RRF fusion of semantic + BM25.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of results with text, metadata, and RRF scores
        """
        results = self.hybrid_index.invoke(query, config={"top_k": top_k})
        
        # Format results
        formatted = []
        for doc in results:
            formatted.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', None),
                "retrieval_type": "hybrid_rrf"
            })
        
        return formatted
    
    def _save_index_metadata(self):
        """Save metadata about indexed papers."""
        metadata_path = self.index_dir / "index_metadata.json"
        metadata = {
            "indexed_papers": list(self.indexed_papers),
            "last_updated": datetime.utcnow().isoformat(),
            "num_papers": len(self.indexed_papers)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_index_metadata(self):
        """Load metadata about indexed papers."""
        metadata_path = self.index_dir / "index_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.indexed_papers = set(metadata.get("indexed_papers", []))
                return metadata
        return None
    
    def clear_indexes(self):
        """Clear all indexes (for testing or reset)."""
        import shutil
        
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self._semantic_index = None
        self._bm25_index = None
        self._hybrid_index = None
        self.indexed_papers = set()
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        # This would require maintaining a chunk lookup table
        # For now, we rely on the retrievers to find chunks
        # In production, you'd want a direct lookup mechanism
        return None

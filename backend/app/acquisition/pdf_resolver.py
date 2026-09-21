"""
Paper Acquisition Module - Resolves and retrieves full-text PDFs for discovered papers.
"""
import hashlib
import tempfile
import aiohttp
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from app.storage.schemas import Paper, PaperStatus
from app.security.validators import InputValidator


class PDFResolver:
    """Resolves accessible full-text sources for papers."""
    
    def __init__(self, storage_dir: str = "pdf_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.validator = InputValidator()
        
    async def resolve_pdf_url(self, paper: Paper) -> Optional[str]:
        """
        Resolve an accessible PDF URL for a paper.
        
        Priority:
        1. Direct PDF URL from metadata
        2. arXiv PDF (if arXiv paper)
        3. OpenAccess URLs from providers
        4. DOI-based resolution attempts
        
        Returns:
            PDF URL if found, None otherwise
        """
        # Check for direct PDF URL first
        if paper.full_text_url:
            if self.validator.validate_url(paper.full_text_url):
                return paper.full_text_url
        
        # Try arXiv PDF if it's an arXiv paper
        if paper.provider.value == "arxiv" or "arxiv" in paper.doi.lower():
            arxiv_id = self._extract_arxiv_id(paper)
            if arxiv_id:
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # Try provider-specific PDF resolution
        if paper.provider.value == "openalex":
            pdf_url = await self._try_openalex_pdf(paper)
            if pdf_url:
                return pdf_url
        
        if paper.provider.value == "semantic_scholar":
            pdf_url = await self._try_semantic_scholar_pdf(paper)
            if pdf_url:
                return pdf_url
        
        # Try DOI-based resolution
        if paper.doi and paper.doi.startswith("10."):
            doi_pdf = await self._try_doi_resolution(paper.doi)
            if doi_pdf:
                return doi_pdf
        
        return None
    
    def _extract_arxiv_id(self, paper: Paper) -> Optional[str]:
        """Extract arXiv ID from paper metadata."""
        import re
        
        # Check DOI
        if paper.doi:
            match = re.search(r'arXiv[:\.]?(\d{4}\.\d{4,5})', paper.doi, re.IGNORECASE)
            if match:
                return match.group(1)
            
            match = re.search(r'arxiv:(\d{4}\.\d{4,5})', paper.doi, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Check URL
        if paper.url:
            match = re.search(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', paper.url)
            if match:
                return match.group(1)
        
        # Check provider_id
        if paper.provider_id and '.' in paper.provider_id:
            return paper.provider_id
        
        return None
    
    async def _try_openalex_pdf(self, paper: Paper) -> Optional[str]:
        """Try to get PDF URL from OpenAlex."""
        # OpenAlex often includes PDF URLs in the primary_location
        # This would require an API call, so we return None for now
        # The discovery layer should have already populated full_text_url if available
        return paper.full_text_url
    
    async def _try_semantic_scholar_pdf(self, paper: Paper) -> Optional[str]:
        """Try to get PDF URL from Semantic Scholar."""
        # Semantic Scholar doesn't directly provide PDFs
        # but may link to open access versions
        return None
    
    async def _try_doi_resolution(self, doi: str) -> Optional[str]:
        """Try to resolve DOI to an open access PDF."""
        # Try Unpaywall API for open access versions
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.unpaywall.org/v2/{doi}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('is_oa', False):
                            best_oa = data.get('best_oa_location', {})
                            if best_oa.get('url') and best_oa.get('host_type') != 'publisher':
                                pdf_url = best_oa['url']
                                if self.validator.validate_url(pdf_url):
                                    return pdf_url
        except Exception:
            pass  # Gracefully fail
        
        return None
    
    async def retrieve_pdf(self, paper: Paper, pdf_url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve PDF from URL and store locally.
        
        Returns:
            Dictionary with retrieval metadata or None on failure
        """
        if not self.validator.validate_url(pdf_url):
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    pdf_url, 
                    timeout=aiohttp.ClientTimeout(total=60),
                    allow_redirects=True
                ) as response:
                    if response.status != 200:
                        return None
                    
                    # Read PDF content
                    pdf_content = await response.read()
                    
                    # Validate it's actually a PDF
                    if not pdf_content.startswith(b'%PDF'):
                        return None
                    
                    # Generate content hash
                    content_hash = hashlib.sha256(pdf_content).hexdigest()
                    
                    # Create safe filename
                    safe_doi = paper.doi.replace('/', '_').replace(':', '_')
                    filename = f"{safe_doi}.pdf"
                    local_path = self.storage_dir / filename
                    
                    # Save to local storage
                    with open(local_path, 'wb') as f:
                        f.write(pdf_content)
                    
                    return {
                        "paper_doi": paper.doi,
                        "source_url": pdf_url,
                        "local_path": str(local_path),
                        "content_hash": content_hash,
                        "file_size": len(pdf_content),
                        "retrieved_at": datetime.utcnow().isoformat(),
                        "status": "success"
                    }
                    
        except Exception as e:
            print(f"Failed to retrieve PDF for {paper.doi}: {e}")
            return {
                "paper_doi": paper.doi,
                "source_url": pdf_url,
                "error": str(e),
                "status": "failed",
                "retrieved_at": datetime.utcnow().isoformat()
            }
    
    def get_local_pdf_path(self, paper: Paper) -> Optional[Path]:
        """Get local path for already-retrieved PDF."""
        safe_doi = paper.doi.replace('/', '_').replace(':', '_')
        filename = f"{safe_doi}.pdf"
        local_path = self.storage_dir / filename
        
        if local_path.exists():
            return local_path
        
        return None

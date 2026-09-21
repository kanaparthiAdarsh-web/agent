"""Paper discovery service with multiple provider support."""

import logging
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime
import hashlib

from ..schemas import PaperMetadata, PaperSource
from ..exceptions import DiscoveryError, ProviderUnavailableError

logger = logging.getLogger(__name__)


class DiscoveryProvider(ABC):
    """Abstract base class for paper discovery providers."""
    
    name: str = "base"
    source: PaperSource = PaperSource.UPLOADED
    
    @abstractmethod
    async def search(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        max_results: int = 50,
    ) -> List[PaperMetadata]:
        """Search for papers matching the query."""
        pass
    
    @abstractmethod
    async def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get metadata for a specific paper."""
        pass
    
    @abstractmethod
    async def get_pdf_url(self, paper: PaperMetadata) -> Optional[str]:
        """Get PDF download URL for a paper."""
        pass
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available (API keys configured, etc.)."""
        return True


class OpenAlexProvider(DiscoveryProvider):
    """OpenAlex API provider for paper discovery."""
    
    name = "openalex"
    source = PaperSource.OPENALEX
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self._available = True
    
    async def search(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        max_results: int = 50,
    ) -> List[PaperMetadata]:
        """Search OpenAlex for papers."""
        import httpx
        
        # Build filter string
        filters = []
        if year_min:
            filters.append(f"publication_year:>={year_min}")
        if year_max:
            filters.append(f"publication_year:<={year_max}")
        filter_str = ",".join(filters) if filters else None
        
        url = f"{self.BASE_URL}/works"
        params = {
            "search": query,
            "per_page": min(max_results, 200),
            "select": "id,title,authorships,publication_year,host_venue,doi,open_access,abstract_inverted_index,cited_by_count",
        }
        if filter_str:
            params["filter"] = filter_str
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                results = []
                for work in data.get("results", []):
                    paper = self._parse_work(work)
                    if paper:
                        results.append(paper)
                
                return results
                
        except httpx.HTTPError as e:
            logger.warning("OpenAlex API error: %s", e)
            raise ProviderUnavailableError(f"OpenAlex unavailable: {e}")
        except Exception as e:
            logger.error("OpenAlex search failed: %s", e)
            raise DiscoveryError(f"OpenAlex search failed: {e}")
    
    def _parse_work(self, work: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Parse OpenAlex work response to PaperMetadata."""
        try:
            # Extract authors
            authors = []
            for authorship in work.get("authorships", [])[:10]:  # Limit to 10 authors
                author_name = authorship.get("author", {}).get("display_name", "")
                if author_name:
                    authors.append(author_name)
            
            # Extract venue
            venue = None
            host_venue = work.get("host_venue")
            if host_venue:
                venue = host_venue.get("display_name")
            
            # Extract abstract (inverted index format)
            abstract = None
            abstract_index = work.get("abstract_inverted_index")
            if abstract_index:
                abstract = self._reconstruct_abstract(abstract_index)
            
            # Get PDF URL if open access
            pdf_url = None
            open_access = work.get("open_access", {})
            if open_access.get("is_oa"):
                pdf_url = open_access.get("oa_url")
            
            return PaperMetadata(
                id=self._normalize_id(work.get("id")),
                title=work.get("title", ""),
                authors=authors,
                year=work.get("publication_year"),
                venue=venue,
                doi=work.get("doi"),
                url=work.get("id"),
                pdf_url=pdf_url,
                abstract=abstract,
                source=self.source,
                citation_count=work.get("cited_by_count"),
                open_access=open_access.get("is_oa", False),
            )
        except Exception as e:
            logger.warning("Failed to parse OpenAlex work: %s", e)
            return None
    
    def _reconstruct_abstract(self, inverted_index: Dict[str, List[int]]) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        
        # Create position -> word mapping
        positions = {}
        for word, indices in inverted_index.items():
            for pos in indices:
                positions[pos] = word
        
        # Sort by position and join
        sorted_words = [positions[i] for i in sorted(positions.keys())]
        return " ".join(sorted_words)
    
    def _normalize_id(self, openalex_id: str) -> str:
        """Normalize OpenAlex ID to our internal format."""
        if not openalex_id:
            return ""
        # OpenAlex IDs look like "https://openalex.org/W1234567890"
        # We extract the numeric part
        parts = openalex_id.rstrip("/").split("/")
        return f"openalex_{parts[-1]}" if len(parts) > 1 else openalex_id
    
    async def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get specific paper from OpenAlex."""
        import httpx
        
        # Convert our ID format back to OpenAlex format
        if paper_id.startswith("openalex_"):
            openalex_id = paper_id.replace("openalex_", "https://openalex.org/")
        else:
            openalex_id = paper_id
        
        url = f"{self.BASE_URL}/works/{openalex_id}"
        params = {"select": "id,title,authorships,publication_year,host_venue,doi,open_access,abstract_inverted_index,cited_by_count"}
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                work = response.json()
                return self._parse_work(work)
        except Exception as e:
            logger.warning("Failed to get paper from OpenAlex: %s", e)
            return None
    
    async def get_pdf_url(self, paper: PaperMetadata) -> Optional[str]:
        """Get PDF URL from OpenAlex."""
        if paper.pdf_url:
            return paper.pdf_url
        
        # Try to fetch updated info
        updated = await self.get_paper(paper.id)
        if updated:
            return updated.pdf_url
        
        return None


class SemanticScholarProvider(DiscoveryProvider):
    """Semantic Scholar API provider for paper discovery."""
    
    name = "semantic_scholar"
    source = PaperSource.SEMANTIC_SCHOLAR
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
    
    async def search(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        max_results: int = 50,
    ) -> List[PaperMetadata]:
        """Search Semantic Scholar for papers."""
        import httpx
        
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,authors,year,venue,externalIds,abstract,isOpenAccess,url,citationCount,openAccessPdf",
        }
        
        if year_min:
            params["year"] = f"{year_min}-"
        if year_max:
            params["year"] = f"{params.get('year', '')}-{year_max}"
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                results = []
                for paper_data in data.get("data", []):
                    paper = self._parse_paper(paper_data)
                    if paper:
                        results.append(paper)
                
                return results
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise ProviderUnavailableError("Semantic Scholar rate limit exceeded")
            logger.warning("Semantic Scholar API error: %s", e)
            raise ProviderUnavailableError(f"Semantic Scholar unavailable: {e}")
        except Exception as e:
            logger.error("Semantic Scholar search failed: %s", e)
            raise DiscoveryError(f"Semantic Scholar search failed: {e}")
    
    def _parse_paper(self, paper_data: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Parse Semantic Scholar paper response."""
        try:
            authors = [a.get("name", "") for a in paper_data.get("authors", []) if a.get("name")]
            
            # Get DOI from external IDs
            external_ids = paper_data.get("externalIds", {})
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")
            
            # Get PDF URL
            pdf_url = None
            open_access_pdf = paper_data.get("openAccessPdf")
            if open_access_pdf:
                pdf_url = open_access_pdf.get("url")
            
            return PaperMetadata(
                id=f"s2_{paper_data.get('paperId', '')}",
                title=paper_data.get("title", ""),
                authors=authors[:10],  # Limit to 10
                year=paper_data.get("year"),
                venue=paper_data.get("venue"),
                doi=doi,
                arxiv_id=arxiv_id,
                url=paper_data.get("url"),
                pdf_url=pdf_url,
                abstract=paper_data.get("abstract"),
                source=self.source,
                citation_count=paper_data.get("citationCount"),
                open_access=paper_data.get("isOpenAccess", False),
            )
        except Exception as e:
            logger.warning("Failed to parse Semantic Scholar paper: %s", e)
            return None
    
    async def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get specific paper from Semantic Scholar."""
        import httpx
        
        s2_id = paper_id.replace("s2_", "") if paper_id.startswith("s2_") else paper_id
        
        url = f"{self.BASE_URL}/paper/{s2_id}"
        params = {"fields": "title,authors,year,venue,externalIds,abstract,isOpenAccess,url,citationCount,openAccessPdf"}
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return self._parse_paper(response.json())
        except Exception as e:
            logger.warning("Failed to get paper from Semantic Scholar: %s", e)
            return None
    
    async def get_pdf_url(self, paper: PaperMetadata) -> Optional[str]:
        """Get PDF URL from Semantic Scholar."""
        if paper.pdf_url:
            return paper.pdf_url
        return None


class ArXivProvider(DiscoveryProvider):
    """arXiv API provider for paper discovery."""
    
    name = "arxiv"
    source = PaperSource.ARXIV  # type: ignore
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    async def search(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        max_results: int = 50,
    ) -> List[PaperMetadata]:
        """Search arXiv for papers."""
        import httpx
        import xml.etree.ElementTree as ET
        
        # arXiv uses ATOM XML format
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                
                results = []
                for entry in root.findall("atom:entry", ns):
                    paper = self._parse_entry(entry, ns)
                    if paper:
                        results.append(paper)
                
                return results
                
        except Exception as e:
            logger.error("arXiv search failed: %s", e)
            raise DiscoveryError(f"arXiv search failed: {e}")
    
    def _parse_entry(self, entry: Any, ns: Dict) -> Optional[PaperMetadata]:
        """Parse arXiv ATOM entry."""
        try:
            def find_text(elem, tag):
                found = elem.find(f"atom:{tag}", ns)
                return found.text if found is not None else ""
            
            title = find_text(entry, "title").strip()
            summary = find_text(entry, "summary").strip()
            
            # Get authors
            authors = []
            for author_elem in entry.findall("atom:author", ns):
                name_elem = author_elem.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())
            
            # Get published date
            published = find_text(entry, "published")
            year = None
            if published:
                try:
                    year = int(published.split("-")[0])
                except ValueError:
                    pass
            
            # Get arXiv ID and links
            arxiv_id = ""
            pdf_url = None
            url = None
            
            for id_elem in entry.findall("atom:id", ns):
                if id_elem.text:
                    url = id_elem.text
                    # Extract arXiv ID from URL
                    if "arxiv.org/abs/" in id_elem.text:
                        arxiv_id = id_elem.text.split("arxiv.org/abs/")[-1].split("v")[0]
            
            for link_elem in entry.findall("atom:link", ns):
                rel = link_elem.get("rel", "")
                link_type = link_elem.get("type", "")
                href = link_elem.get("href", "")
                
                if link_type == "application/pdf":
                    pdf_url = href
                elif rel == "alternate" and not url:
                    url = href
            
            return PaperMetadata(
                id=f"arxiv_{arxiv_id}" if arxiv_id else "",
                title=title,
                authors=authors[:10],
                year=year,
                venue="arXiv",
                arxiv_id=arxiv_id,
                url=url,
                pdf_url=pdf_url,
                abstract=summary,
                source=self.source,
                open_access=True,  # arXiv papers are open access
            )
        except Exception as e:
            logger.warning("Failed to parse arXiv entry: %s", e)
            return None
    
    async def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get specific paper from arXiv."""
        arxiv_id = paper_id.replace("arxiv_", "") if paper_id.startswith("arxiv_") else paper_id
        
        # Search by ID
        results = await self.search(f"id_list:{arxiv_id}", max_results=1)
        return results[0] if results else None
    
    async def get_pdf_url(self, paper: PaperMetadata) -> Optional[str]:
        """Get PDF URL from arXiv."""
        if paper.pdf_url:
            return paper.pdf_url
        if paper.arxiv_id:
            return f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        return None


class CrossrefProvider(DiscoveryProvider):
    """Crossref API provider for paper discovery."""
    
    name = "crossref"
    source = PaperSource.CROSSREF
    BASE_URL = "https://api.crossref.org/works"
    
    def __init__(self, mailto: Optional[str] = None, timeout: int = 30):
        self.mailto = mailto or "research-agent@example.com"
        self.timeout = timeout
    
    async def search(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        max_results: int = 50,
    ) -> List[PaperMetadata]:
        """Search Crossref for papers."""
        import httpx
        
        params = {
            "query": query,
            "rows": min(max_results, 100),
            "mailto": self.mailto,
        }
        
        # Crossref uses from-until-date format
        if year_min:
            params["from-pub-date"] = f"{year_min}-01-01"
        if year_max:
            params["until-pub-date"] = f"{year_max}-12-31"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = []
                for item in data.get("message", {}).get("items", []):
                    paper = self._parse_item(item)
                    if paper:
                        results.append(paper)
                
                return results
                
        except Exception as e:
            logger.error("Crossref search failed: %s", e)
            raise DiscoveryError(f"Crossref search failed: {e}")
    
    def _parse_item(self, item: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Parse Crossref item response."""
        try:
            # Get authors
            authors = []
            for author in item.get("author", [])[:10]:
                given = author.get("given", "")
                family = author.get("family", "")
                if given or family:
                    authors.append(f"{given} {family}".strip())
            
            # Get DOI
            doi = item.get("DOI")
            
            # Get URLs
            url = None
            pdf_url = None
            for url_item in item.get("URL", []):
                if url_item.get("content-type") == "application/pdf":
                    pdf_url = url_item.get("URL")
                elif not url:
                    url = url_item.get("URL")
            
            if not url and doi:
                url = f"https://doi.org/{doi}"
            
            # Check open access
            open_access = item.get("open-access") == "true" or item.get("license") is not None
            
            return PaperMetadata(
                id=f"crossref_{doi}" if doi else "",
                title=item.get("title", [""])[0] if isinstance(item.get("title"), list) else item.get("title", ""),
                authors=authors,
                year=item.get("created", {}).get("date-parts", [[None]])[0][0],
                venue=item.get("container-title", [""])[0] if isinstance(item.get("container-title"), list) else item.get("container-title"),
                doi=doi,
                url=url,
                pdf_url=pdf_url,
                abstract=item.get("abstract"),
                source=self.source,
                citation_count=item.get("is-referenced-by-count"),
                open_access=open_access,
            )
        except Exception as e:
            logger.warning("Failed to parse Crossref item: %s", e)
            return None
    
    async def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get specific paper from Crossref by DOI."""
        import httpx
        
        doi = paper_id.replace("crossref_", "") if paper_id.startswith("crossref_") else paper_id
        
        url = f"{self.BASE_URL}/{doi}"
        params = {"mailto": self.mailto}
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                item = data.get("message", {})
                return self._parse_item(item)
        except Exception as e:
            logger.warning("Failed to get paper from Crossref: %s", e)
            return None
    
    async def get_pdf_url(self, paper: PaperMetadata) -> Optional[str]:
        """Get PDF URL from Crossref."""
        if paper.pdf_url:
            return paper.pdf_url
        return None


class DiscoveryService:
    """Unified paper discovery service with multiple providers."""
    
    def __init__(
        self,
        openalex_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        crossref_mailto: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize discovery service with configured providers."""
        self.providers: List[DiscoveryProvider] = []
        self.timeout = timeout
        
        # Initialize available providers
        self.providers.append(OpenAlexProvider(api_key=openalex_api_key, timeout=timeout))
        self.providers.append(SemanticScholarProvider(api_key=semantic_scholar_api_key, timeout=timeout))
        self.providers.append(ArXivProvider(timeout=timeout))
        self.providers.append(CrossrefProvider(mailto=crossref_mailto, timeout=timeout))
        
        logger.info("Initialized DiscoveryService with %d providers", len(self.providers))
    
    async def search(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        max_results: int = 50,
        sources: Optional[List[str]] = None,
    ) -> List[PaperMetadata]:
        """
        Search across all available providers.
        
        Args:
            query: Search query
            year_min: Minimum publication year
            year_max: Maximum publication year
            max_results: Maximum total results to return
            sources: Optional list of source names to use (default: all)
        
        Returns:
            Deduplicated and ranked list of papers
        """
        all_papers: List[PaperMetadata] = []
        
        # Filter providers if sources specified
        active_providers = self.providers
        if sources:
            active_providers = [p for p in self.providers if p.name in sources]
        
        # Distribute max_results across providers
        per_provider = max(10, max_results // len(active_providers)) if active_providers else max_results
        
        # Search each provider concurrently
        import asyncio
        
        async def search_provider(provider: DiscoveryProvider) -> List[PaperMetadata]:
            if not provider.is_available:
                logger.debug("Provider %s not available, skipping", provider.name)
                return []
            
            try:
                results = await provider.search(query, year_min, year_max, per_provider)
                logger.info("Provider %s returned %d results", provider.name, len(results))
                return results
            except ProviderUnavailableError as e:
                logger.warning("Provider %s unavailable: %s", provider.name, e)
                return []
            except Exception as e:
                logger.error("Provider %s search failed: %s", provider.name, e)
                return []
        
        tasks = [search_provider(p) for p in active_providers]
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in provider_results:
            if isinstance(result, list):
                all_papers.extend(result)
        
        # Deduplicate and rank
        deduplicated = self._deduplicate_papers(all_papers)
        ranked = self._rank_papers(deduplicated, query)
        
        return ranked[:max_results]
    
    def _deduplicate_papers(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Deduplicate papers using DOI-first strategy."""
        seen_dois: Dict[str, PaperMetadata] = {}
        seen_titles: Dict[str, PaperMetadata] = {}
        
        for paper in papers:
            # DOI-based deduplication (highest priority)
            if paper.doi:
                doi_normalized = paper.doi.lower().rstrip("/")
                if doi_normalized not in seen_dois:
                    seen_dois[doi_normalized] = paper
                continue
            
            # arXiv ID deduplication
            if paper.arxiv_id:
                arxiv_key = f"arxiv:{paper.arxiv_id}"
                if arxiv_key not in seen_dois:
                    seen_dois[arxiv_key] = paper
                continue
            
            # Title-based deduplication (fallback)
            title_key = self._normalize_title(paper.title)
            if title_key and title_key not in seen_titles:
                # Check if we have same authors
                author_key = tuple(sorted(a.lower() for a in paper.authors[:3]))
                existing = seen_titles.get(title_key)
                
                if not existing:
                    seen_titles[title_key] = paper
                elif author_key == tuple(sorted(a.lower() for a in existing.authors[:3])):
                    # Same title and similar authors - keep the one with more info
                    if len(paper.authors) > len(existing.authors) or paper.abstract:
                        seen_titles[title_key] = paper
        
        # Combine DOI and title results (DOI takes precedence)
        combined = list(seen_dois.values()) + list(seen_titles.values())
        
        # Remove any remaining duplicates by ID
        seen_ids = set()
        unique = []
        for paper in combined:
            if paper.id and paper.id not in seen_ids:
                seen_ids.add(paper.id)
                unique.append(paper)
        
        return unique
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        if not title:
            return ""
        # Lowercase, remove punctuation, normalize whitespace
        import re
        normalized = title.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = " ".join(normalized.split())
        return normalized
    
    def _rank_papers(
        self,
        papers: List[PaperMetadata],
        query: str,
    ) -> List[PaperMetadata]:
        """Rank papers by relevance and quality."""
        query_terms = set(query.lower().split())
        
        scored_papers = []
        for paper in papers:
            score = 0.0
            
            # Title match (highest weight)
            title_lower = paper.title.lower()
            for term in query_terms:
                if term in title_lower:
                    score += 3.0
            
            # Abstract match
            if paper.abstract:
                abstract_lower = paper.abstract.lower()
                for term in query_terms:
                    if term in abstract_lower:
                        score += 1.0
            
            # Citation count bonus
            if paper.citation_count:
                score += min(paper.citation_count / 100, 5.0)  # Cap at 5 points
            
            # Recent paper bonus
            if paper.year and paper.year >= 2022:
                score += 1.0
            
            # Open access bonus
            if paper.open_access:
                score += 0.5
            
            # Has PDF bonus
            if paper.pdf_url:
                score += 0.5
            
            scored_papers.append((score, paper))
        
        # Sort by score descending
        scored_papers.sort(key=lambda x: -x[0])
        return [paper for _, paper in scored_papers]
    
    async def get_pdf_url(self, paper: PaperMetadata) -> Optional[str]:
        """Get PDF URL for a paper from its source provider."""
        provider = self._get_provider_for_source(paper.source)
        if provider:
            return await provider.get_pdf_url(paper)
        return paper.pdf_url
    
    def _get_provider_for_source(self, source: PaperSource) -> Optional[DiscoveryProvider]:
        """Get provider instance for a paper source."""
        for provider in self.providers:
            if provider.source == source:
                return provider
        return None

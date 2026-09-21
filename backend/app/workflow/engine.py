"""
Research Workflow Engine - Orchestrates the complete research pipeline.
"""
from typing import List, Optional, Dict, Any
import asyncio
from pathlib import Path
from datetime import datetime

from app.storage.schemas import Paper, PaperCard, Evidence, Comparison, Limitation, Gap, ResearchDirection, Job, PaperStatus
from app.discovery.service import DiscoveryService
from app.acquisition.pdf_resolver import PDFResolver
from app.ingestion.adapter import PDFIngestionAdapter
from app.indexing.adapter import IndexingAdapter
from app.generation.papercard_generator import PaperCardGenerator
from app.evidence.extractor import EvidenceExtractor
from app.evidence.verifier import EvidenceVerifier
from app.comparison.engine import ComparisonEngine
from app.limitations.extractor import LimitationExtractor
from app.limitations.clustering import LimitationClustering
from app.gaps.gap_finder import GapFinder
from app.directions.generator import DirectionGenerator
from app.storage.repositories.sqlite_repo import get_repositories


class ResearchWorkflowEngine:
    """
    Main workflow engine for the research pipeline.
    
    Orchestrates:
    1. Discovery
    2. Acquisition (PDF resolution)
    3. PDF Ingestion & Chunking
    4. Indexing
    5. PaperCard Generation
    6. Evidence Extraction & Verification
    7. Comparison
    8. Limitation Extraction & Clustering
    9. Gap Analysis
    10. Research Direction Generation
    """
    
    def __init__(self, db_path: str = "literature.db", index_dir: str = "indexes"):
        self.db_path = db_path
        self.repositories = get_repositories(db_path)
        
        # Initialize components
        self.discovery = DiscoveryService()
        self.pdf_resolver = PDFResolver()
        self.pdf_ingestion = PDFIngestionAdapter()
        self.indexing = IndexingAdapter(index_dir=index_dir)
        self.papercard_gen = PaperCardGenerator()
        self.evidence_extractor = EvidenceExtractor()
        self.evidence_verifier = EvidenceVerifier()
        self.comparison_engine = ComparisonEngine()
        self.limitation_extractor = LimitationExtractor()
        self.limitation_clustering = LimitationClustering()
        self.gap_finder = GapFinder()
        self.direction_generator = DirectionGenerator()
        
        # Track processing state
        self.processed_papers: Dict[str, Dict[str, Any]] = {}
    
    async def execute_workflow(
        self, 
        research_question: str, 
        max_papers: int = 10,
        uploaded_pdfs: Optional[List[bytes]] = None
    ) -> Job:
        """
        Execute the complete research workflow.
        
        Args:
            research_question: The research question to investigate
            max_papers: Maximum number of papers to discover
            uploaded_pdfs: Optional list of uploaded PDF bytes
            
        Returns:
            Job object with final status
        """
        # Create job
        job_id = f"job_{int(datetime.now().timestamp())}"
        job = Job(
            id=job_id,
            research_question=research_question,
            status="started",
            progress=0.0,
            total_papers=max_papers,
            processed_papers=0
        )
        
        # Save job to DB
        self.repositories["jobs"].create_job(job)
        
        all_chunks = []
        all_papers = []
        all_evidence = []
        all_limitations = []
        
        try:
            # PHASE 1: Handle uploaded PDFs first
            if uploaded_pdfs:
                job = self.repositories["jobs"].update_job(job_id, status="processing_uploads")
                for i, pdf_bytes in enumerate(uploaded_pdfs):
                    try:
                        result = self.pdf_ingestion.ingest_uploaded_pdf(
                            pdf_bytes,
                            filename=f"uploaded_paper_{i}.pdf",
                            research_question=research_question
                        )
                        
                        paper = result["paper"]
                        chunks = result["chunks"]
                        
                        # Save paper
                        self.repositories["papers"].save_paper(paper)
                        
                        # Index chunks
                        self.indexing.index_chunks(chunks, paper.doi)
                        
                        # Generate PaperCard
                        papercard = self.papercard_gen.generate_papercard(paper)
                        self.repositories["papercards"].save_paper_card(papercard)
                        
                        # Extract evidence from chunks
                        chunk_texts = [c.content for c in chunks]
                        full_text = "\n\n".join(chunk_texts)
                        evidences = self.evidence_extractor.extract_evidence(paper, full_text)
                        verified_evidences = self.evidence_verifier.verify_multiple_evidence(
                            evidences, full_text
                        )
                        for ev in verified_evidences:
                            self.repositories["evidence"].save_evidence(ev)
                            all_evidence.append(ev)
                        
                        # Extract limitations
                        limitations = self.limitation_extractor.extract_limitations(paper, full_text)
                        for lim in limitations:
                            self.repositories["limitations"].save_limitation(lim)
                            all_limitations.append(lim)
                        
                        all_chunks.extend(chunks)
                        all_papers.append(paper)
                        
                        # Store processing info
                        self.processed_papers[paper.doi] = {
                            "chunks": len(chunks),
                            "evidence": len(verified_evidences),
                            "limitations": len(limitations)
                        }
                        
                    except Exception as e:
                        print(f"Error processing uploaded PDF {i}: {e}")
                        continue
                
                job = self.repositories["jobs"].update_job(
                    job_id, 
                    status="discovery",
                    progress=0.1
                )
            
            # PHASE 2: Discovery
            papers = await self.discovery.discover_papers(research_question, limit=max_papers)
            job = self.repositories["jobs"].update_job(
                job_id, 
                status="acquisition", 
                total_papers=len(papers) + len(all_papers),
                progress=0.15
            )
            
            # PHASE 3: Acquisition and Processing
            for i, paper in enumerate(papers):
                try:
                    # Update status
                    paper.status = PaperStatus.RETRIEVED
                    saved_paper = self.repositories["papers"].save_paper(paper)
                    
                    # Try to resolve PDF URL
                    pdf_url = await self.pdf_resolver.resolve_pdf_url(paper)
                    
                    if pdf_url:
                        # Retrieve PDF
                        retrieval_result = await self.pdf_resolver.retrieve_pdf(paper, pdf_url)
                        
                        if retrieval_result and retrieval_result.get("status") == "success":
                            # Process PDF
                            pdf_path = retrieval_result["local_path"]
                            ingestion_result = self.pdf_ingestion.ingest_pdf(pdf_path, paper)
                            
                            chunks = ingestion_result["chunks"]
                            structure = ingestion_result["structure"]
                            
                            # Index chunks
                            self.indexing.index_chunks(chunks, paper.doi)
                            
                            # Generate PaperCard from actual content
                            chunk_texts = [c.content for c in chunks]
                            full_text = "\n\n".join(chunk_texts)
                            
                            # Create enhanced paper with full text info
                            enhanced_paper = Paper(
                                doi=paper.doi,
                                title=paper.title,
                                authors=paper.authors,
                                abstract=full_text[:500] if len(full_text) > 500 else full_text,
                                provider=paper.provider,
                                provider_id=paper.provider_id,
                                status=PaperStatus.PROCESSED
                            )
                            
                            papercard = self.papercard_gen.generate_papercard(enhanced_paper)
                            self.repositories["papercards"].save_paper_card(papercard)
                            
                            # Extract evidence
                            evidences = self.evidence_extractor.extract_evidence(enhanced_paper, full_text)
                            verified_evidences = self.evidence_verifier.verify_multiple_evidence(
                                evidences, full_text
                            )
                            for ev in verified_evidences:
                                self.repositories["evidence"].save_evidence(ev)
                                all_evidence.append(ev)
                            
                            # Extract limitations
                            limitations = self.limitation_extractor.extract_limitations(enhanced_paper, full_text)
                            for lim in limitations:
                                self.repositories["limitations"].save_limitation(lim)
                                all_limitations.append(lim)
                            
                            all_chunks.extend(chunks)
                            all_papers.append(saved_paper)
                            
                            # Store processing info
                            self.processed_papers[saved_paper.doi] = {
                                "chunks": len(chunks),
                                "evidence": len(verified_evidences),
                                "limitations": len(limitations),
                                "pdf_source": pdf_url
                            }
                            
                            saved_paper.status = PaperStatus.PROCESSED
                            self.repositories["papers"].save_paper(saved_paper)
                        else:
                            # No PDF available, process with metadata only
                            self._process_metadata_only(paper, all_evidence, all_limitations, all_papers)
                    else:
                        # No PDF URL found, process with metadata only
                        self._process_metadata_only(paper, all_evidence, all_limitations, all_papers)
                    
                    # Update progress
                    progress = 0.15 + ((i + 1) / len(papers)) * 0.55
                    job = self.repositories["jobs"].update_job(
                        job_id,
                        processed_papers=len(all_papers),
                        progress=progress
                    )
                    
                except Exception as e:
                    print(f"Error processing paper {paper.doi}: {e}")
                    paper.status = PaperStatus.FAILED
                    self.repositories["papers"].save_paper(paper)
                    continue
            
            # PHASE 4: Analysis
            job = self.repositories["jobs"].update_job(job_id, status="analysis", progress=0.75)
            
            # Get all processed papers
            processed_paper_list = [p for p in all_papers if p.status == PaperStatus.PROCESSED]
            
            # Generate comparisons
            if len(processed_paper_list) > 1:
                comparisons = self.comparison_engine.compare_multiple_papers(processed_paper_list)
                for comp in comparisons:
                    self.repositories["comparisons"].save_comparison(comp)
            
            # Cluster limitations
            limitation_clusters = self.limitation_clustering.cluster_limitations(all_limitations)
            
            # Find gaps
            gaps = self.gap_finder.find_gaps(processed_paper_list, all_limitations)
            for gap in gaps:
                self.repositories["gaps"].save_gap(gap)
            
            # Generate directions
            directions = self.direction_generator.generate_directions(
                processed_paper_list, gaps, all_evidence
            )
            for direction in directions:
                self.repositories["directions"].save_direction(direction)
            
            # Complete workflow
            job = self.repositories["jobs"].update_job(
                job_id,
                status="completed",
                progress=1.0,
                processed_papers=len(processed_paper_list)
            )
            
            return job
            
        except Exception as e:
            print(f"Workflow failed: {e}")
            job = self.repositories["jobs"].update_job(job_id, status="failed", progress=0.0)
            raise e
    
    def _process_metadata_only(
        self, 
        paper: Paper, 
        all_evidence: List[Evidence],
        all_limitations: List[Limitation],
        all_papers: List[Paper]
    ):
        """Process a paper using only metadata (no full text)."""
        # Generate PaperCard from abstract/metadata
        papercard = self.papercard_gen.generate_papercard(paper)
        self.repositories["papercards"].save_paper_card(papercard)
        
        # Extract limited evidence from abstract
        if paper.abstract:
            evidences = self.evidence_extractor.extract_evidence(paper, paper.abstract)
            verified = self.evidence_verifier.verify_multiple_evidence(evidences, paper.abstract)
            for ev in verified:
                self.repositories["evidence"].save_evidence(ev)
                all_evidence.append(ev)
            
            # Extract limitations from abstract
            limitations = self.limitation_extractor.extract_limitations(paper, paper.abstract)
            for lim in limitations:
                self.repositories["limitations"].save_limitation(lim)
                all_limitations.append(lim)
        
        paper.status = PaperStatus.PROCESSED
        self.repositories["papers"].save_paper(paper)
        all_papers.append(paper)
        
        self.processed_papers[paper.doi] = {
            "chunks": 0,
            "evidence": len([e for e in all_evidence if e.paper_doi == paper.doi]),
            "limitations": len([l for l in all_limitations if l.paper_doi == paper.doi]),
            "note": "metadata_only"
        }
    
    def get_job_status(self, job_id: str) -> Optional[Job]:
        """Get the status of a job."""
        return self.repositories["jobs"].get_job(job_id)
    
    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get results for a completed job."""
        job = self.repositories["jobs"].get_job(job_id)
        if not job:
            return {}
        
        return {
            "job": job,
            "papers_processed": len(self.processed_papers),
            "processing_details": self.processed_papers
        }
    
    async def process_uploaded_papers(
        self,
        pdf_contents: List[bytes],
        filenames: List[str],
        research_context: str = ""
    ) -> Dict[str, Any]:
        """
        Process uploaded PDF papers without discovery.
        
        Args:
            pdf_contents: List of PDF byte arrays
            filenames: Original filenames
            research_context: Optional research context
            
        Returns:
            Processing results
        """
        results = {
            "processed": 0,
            "failed": 0,
            "papers": [],
            "chunks": 0
        }
        
        for i, (pdf_bytes, filename) in enumerate(zip(pdf_contents, filenames)):
            try:
                result = self.pdf_ingestion.ingest_uploaded_pdf(
                    pdf_bytes,
                    filename=filename,
                    research_question=research_context
                )
                
                paper = result["paper"]
                chunks = result["chunks"]
                
                # Save and index
                self.repositories["papers"].save_paper(paper)
                self.indexing.index_chunks(chunks, paper.doi)
                
                # Generate PaperCard
                papercard = self.papercard_gen.generate_papercard(paper)
                self.repositories["papercards"].save_paper_card(papercard)
                
                results["processed"] += 1
                results["chunks"] += len(chunks)
                results["papers"].append(paper.doi)
                
                # Store for later analysis
                self.processed_papers[paper.doi] = {
                    "chunks": len(chunks),
                    "filename": filename
                }
                
            except Exception as e:
                print(f"Failed to process upload {i}: {e}")
                results["failed"] += 1
        
        return results

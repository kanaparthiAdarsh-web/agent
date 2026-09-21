"""
Deterministic End-to-End Test using local PDF fixture.

This test exercises the complete pipeline WITHOUT external API dependencies:
- PDF ingestion
- Structure extraction  
- Chunking
- Indexing
- PaperCard generation
- Evidence extraction
- Verification
- Comparison
- Limitation extraction
- Clustering
- Gap generation
- Gap verification
- Research direction generation
"""
import pytest
from pathlib import Path
import tempfile
import shutil


class TestDeterministicE2E:
    """End-to-end test with deterministic PDF fixture."""
    
    @pytest.fixture
    def test_pdf_path(self):
        """Path to test PDF fixture."""
        return Path(__file__).parent / "fixtures" / "test_paper.pdf"
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for test isolation."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        index_dir = Path(temp_dir) / "indexes"
        
        yield {
            "temp_dir": temp_dir,
            "db_path": str(db_path),
            "index_dir": str(index_dir)
        }
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_pdf_ingestion(self, test_pdf_path, temp_dirs):
        """Test PDF ingestion and chunking."""
        from app.ingestion.adapter import PDFIngestionAdapter
        from app.storage.schemas import Paper
        
        adapter = PDFIngestionAdapter()
        
        paper = Paper(
            doi="10.test/fixture",
            title="Test Research Paper",
            authors=["Author A", "Author B"],
            provider="test",
            provider_id="fixture_001",
            status="discovered"
        )
        
        result = adapter.ingest_pdf(str(test_pdf_path), paper)
        
        assert "chunks" in result
        assert "structure" in result
        assert "metadata" in result
        assert len(result["chunks"]) > 0
        assert result["metadata"]["num_pages"] == 1
        
        # Verify chunk structure
        chunk = result["chunks"][0]
        assert chunk.content is not None
        assert len(chunk.content) > 0
        assert chunk.paper_doi == paper.doi
    
    def test_indexing(self, test_pdf_path, temp_dirs):
        """Test indexing of chunks."""
        from app.ingestion.adapter import PDFIngestionAdapter
        from app.indexing.adapter import IndexingAdapter
        from app.storage.schemas import Paper
        
        adapter = PDFIngestionAdapter()
        indexer = IndexingAdapter(index_dir=temp_dirs["index_dir"])
        
        paper = Paper(
            doi="10.test/index_fixture",
            title="Test Paper for Indexing",
            authors=["Author A"],
            provider="test",
            provider_id="fixture_002",
            status="discovered"
        )
        
        # Ingest
        ingestion_result = adapter.ingest_pdf(str(test_pdf_path), paper)
        chunks = ingestion_result["chunks"]
        
        # Index
        index_result = indexer.index_chunks(chunks, paper.doi)
        
        assert index_result["status"] == "success"
        assert index_result["indexed_count"] > 0
        
        # Test search
        results = indexer.search_hybrid("machine learning", top_k=5)
        assert len(results) > 0
        assert "text" in results[0]
        assert "metadata" in results[0]
    
    def test_papercard_generation(self, test_pdf_path, temp_dirs):
        """Test PaperCard generation from ingested content."""
        from app.ingestion.adapter import PDFIngestionAdapter
        from app.generation.papercard_generator import PaperCardGenerator
        from app.storage.schemas import Paper
        
        adapter = PDFIngestionAdapter()
        generator = PaperCardGenerator()
        
        paper = Paper(
            doi="10.test/papercard_fixture",
            title="Test Paper",
            authors=["Author A"],
            abstract="This paper presents machine learning research.",
            provider="test",
            provider_id="fixture_003",
            status="discovered"
        )
        
        # Ingest to get full text
        ingestion_result = adapter.ingest_pdf(str(test_pdf_path), paper)
        chunk_texts = [c.content for c in ingestion_result["chunks"]]
        full_text = "\n\n".join(chunk_texts)
        
        # Create enhanced paper with full text
        enhanced_paper = Paper(
            doi=paper.doi,
            title=paper.title,
            authors=paper.authors,
            abstract=full_text[:500],
            provider=paper.provider,
            provider_id=paper.provider_id,
            status="processed"
        )
        
        papercard = generator.generate_papercard(enhanced_paper)
        
        assert papercard.doi == paper.doi
        assert papercard.title == paper.title
        assert papercard.summary is not None
        assert isinstance(papercard.key_findings, list)
        assert isinstance(papercard.datasets, list)
        assert isinstance(papercard.metrics, list)
        assert isinstance(papercard.limitations, list)
    
    def test_evidence_extraction(self, test_pdf_path, temp_dirs):
        """Test evidence extraction from ingested PDF."""
        from app.ingestion.adapter import PDFIngestionAdapter
        from app.evidence.extractor import EvidenceExtractor
        from app.evidence.verifier import EvidenceVerifier
        from app.storage.schemas import Paper
        
        adapter = PDFIngestionAdapter()
        extractor = EvidenceExtractor()
        verifier = EvidenceVerifier()
        
        paper = Paper(
            doi="10.test/evidence_fixture",
            title="Test Paper",
            authors=["Author A"],
            provider="test",
            provider_id="fixture_004",
            status="discovered"
        )
        
        # Ingest
        ingestion_result = adapter.ingest_pdf(str(test_pdf_path), paper)
        chunk_texts = [c.content for c in ingestion_result["chunks"]]
        full_text = "\n\n".join(chunk_texts)
        
        # Extract evidence
        evidences = extractor.extract_evidence(paper, full_text)
        
        assert len(evidences) > 0
        
        # Verify evidence has required fields
        for ev in evidences:
            assert ev.paper_doi == paper.doi
            assert ev.quote is not None or ev.content is not None
            assert ev.context is not None
        
        # Verify some evidence
        if evidences:
            verified = verifier.verify_evidence(evidences[0], full_text)
            assert verified.verification_state is not None
            assert verified.confidence >= 0
    
    def test_limitation_extraction(self, test_pdf_path, temp_dirs):
        """Test limitation extraction from ingested PDF."""
        from app.ingestion.adapter import PDFIngestionAdapter
        from app.limitations.extractor import LimitationExtractor
        from app.storage.schemas import Paper
        
        adapter = PDFIngestionAdapter()
        extractor = LimitationExtractor()
        
        paper = Paper(
            doi="10.test/limitation_fixture",
            title="Test Paper",
            authors=["Author A"],
            provider="test",
            provider_id="fixture_005",
            status="discovered"
        )
        
        # Ingest
        ingestion_result = adapter.ingest_pdf(str(test_pdf_path), paper)
        chunk_texts = [c.content for c in ingestion_result["chunks"]]
        full_text = "\n\n".join(chunk_texts)
        
        # Extract limitations
        limitations = extractor.extract_limitations(paper, full_text)
        
        # Our test PDF explicitly mentions limitations
        assert len(limitations) > 0
        
        # Check that limitations are classified
        for lim in limitations:
            assert lim.category is not None
            assert lim.original_text is not None
            assert lim.text is not None
    
    def test_limitation_clustering(self, temp_dirs):
        """Test clustering of similar limitations."""
        from app.limitations.clustering import LimitationClustering
        from app.storage.schemas import Limitation
        from datetime import datetime
        
        clustering = LimitationClustering()
        
        # Create test limitations
        limitations = [
            Limitation(
                id="lim1",
                paper_doi="10.test/p1",
                text="Limited dataset size affects generalizability",
                category="data_related",
                original_text="Limited dataset size",
                created_at=datetime.now()
            ),
            Limitation(
                id="lim2",
                paper_doi="10.test/p2",
                text="Small dataset may impact results",
                category="data_related",
                original_text="Small dataset",
                created_at=datetime.now()
            ),
            Limitation(
                id="lim3",
                paper_doi="10.test/p3",
                text="High computational cost",
                category="technical",
                original_text="Computational cost",
                created_at=datetime.now()
            )
        ]
        
        clusters = clustering.cluster_limitations(limitations)
        
        assert len(clusters) > 0
        
        # Data-related limitations should be clustered together
        data_cluster_found = False
        for cluster_id, cluster in clusters.items():
            if cluster[0].category == "data_related" and len(cluster) > 1:
                data_cluster_found = True
                break
        
        assert data_cluster_found, "Similar limitations should be clustered"
    
    def test_gap_generation(self, test_pdf_path, temp_dirs):
        """Test gap generation from limitations."""
        from app.ingestion.adapter import PDFIngestionAdapter
        from app.limitations.extractor import LimitationExtractor
        from app.gaps.gap_finder import GapFinder
        from app.storage.schemas import Paper
        
        adapter = PDFIngestionAdapter()
        lim_extractor = LimitationExtractor()
        gap_finder = GapFinder()
        
        paper = Paper(
            doi="10.test/gap_fixture",
            title="Test Paper with Limitations",
            authors=["Author A"],
            provider="test",
            provider_id="fixture_006",
            status="discovered"
        )
        
        # Ingest and extract limitations
        ingestion_result = adapter.ingest_pdf(str(test_pdf_path), paper)
        chunk_texts = [c.content for c in ingestion_result["chunks"]]
        full_text = "\n\n".join(chunk_texts)
        
        limitations = lim_extractor.extract_limitations(paper, full_text)
        
        # Generate gaps
        gaps = gap_finder.find_gaps([paper], limitations)
        
        # Should find at least one potential gap
        assert len(gaps) > 0
        
        # Verify gap structure
        for gap in gaps:
            assert gap.description is not None
            assert gap.supporting_evidence is not None
            assert gap.uncertainty_expressed is True  # Critical requirement
            assert gap.verified is False  # Starts unverified
    
    def test_research_direction_generation(self, test_pdf_path, temp_dirs):
        """Test research direction generation."""
        from app.ingestion.adapter import PDFIngestionAdapter
        from app.limitations.extractor import LimitationExtractor
        from app.gaps.gap_finder import GapFinder
        from app.directions.generator import DirectionGenerator
        from app.storage.schemas import Paper, Evidence
        from datetime import datetime
        
        adapter = PDFIngestionAdapter()
        lim_extractor = LimitationExtractor()
        gap_finder = GapFinder()
        direction_gen = DirectionGenerator()
        
        paper = Paper(
            doi="10.test/direction_fixture",
            title="Test Paper",
            authors=["Author A"],
            provider="test",
            provider_id="fixture_007",
            status="discovered"
        )
        
        # Process paper
        ingestion_result = adapter.ingest_pdf(str(test_pdf_path), paper)
        chunk_texts = [c.content for c in ingestion_result["chunks"]]
        full_text = "\n\n".join(chunk_texts)
        
        limitations = lim_extractor.extract_limitations(paper, full_text)
        gaps = gap_finder.find_gaps([paper], limitations)
        
        # Create minimal evidence
        evidence = [
            Evidence(
                id="ev1",
                paper_doi=paper.doi,
                content="Test evidence",
                quote="Test quote",
                context="Test context",
                verification_state="unverified",
                confidence=0.5,
                provenance={"source": "test"},
                created_at=datetime.now()
            )
        ]
        
        # Generate directions
        directions = direction_gen.generate_directions([paper], gaps, evidence)
        
        assert len(directions) > 0
        
        # Verify direction structure
        for direction in directions:
            assert direction.description is not None
            assert direction.supporting_evidence is not None
    
    def test_complete_pipeline(self, test_pdf_path, temp_dirs):
        """
        Complete end-to-end pipeline test.
        
        Exercises: PDF -> ingestion -> chunking -> indexing -> 
                   PaperCard -> evidence -> limitations -> gaps -> directions
        """
        from app.ingestion.adapter import PDFIngestionAdapter
        from app.indexing.adapter import IndexingAdapter
        from app.generation.papercard_generator import PaperCardGenerator
        from app.evidence.extractor import EvidenceExtractor
        from app.evidence.verifier import EvidenceVerifier
        from app.limitations.extractor import LimitationExtractor
        from app.limitations.clustering import LimitationClustering
        from app.gaps.gap_finder import GapFinder
        from app.directions.generator import DirectionGenerator
        from app.storage.schemas import Paper, Evidence
        from datetime import datetime
        
        # Initialize all components
        pdf_adapter = PDFIngestionAdapter()
        indexer = IndexingAdapter(index_dir=temp_dirs["index_dir"])
        papercard_gen = PaperCardGenerator()
        evidence_ext = EvidenceExtractor()
        evidence_ver = EvidenceVerifier()
        lim_ext = LimitationExtractor()
        lim_cluster = LimitationClustering()
        gap_finder = GapFinder()
        direction_gen = DirectionGenerator()
        
        # Create test paper
        paper = Paper(
            doi="10.test/complete_pipeline",
            title="Complete Pipeline Test Paper",
            authors=["Test Author"],
            provider="test",
            provider_id="fixture_complete",
            status="discovered"
        )
        
        # Step 1: Ingest PDF
        ingestion_result = pdf_adapter.ingest_pdf(str(test_pdf_path), paper)
        assert len(ingestion_result["chunks"]) > 0
        
        # Step 2: Index chunks
        chunks = ingestion_result["chunks"]
        index_result = indexer.index_chunks(chunks, paper.doi)
        assert index_result["status"] == "success"
        
        # Step 3: Generate PaperCard
        full_text = "\n\n".join([c.content for c in chunks])
        enhanced_paper = Paper(
            doi=paper.doi,
            title=paper.title,
            authors=paper.authors,
            abstract=full_text[:500],
            provider=paper.provider,
            provider_id=paper.provider_id,
            status="processed"
        )
        papercard = papercard_gen.generate_papercard(enhanced_paper)
        assert papercard is not None
        
        # Step 4: Extract evidence
        evidences = evidence_ext.extract_evidence(enhanced_paper, full_text)
        assert len(evidences) > 0
        
        # Step 5: Verify evidence
        verified_evidences = []
        for ev in evidences:
            verified = evidence_ver.verify_evidence(ev, full_text)
            verified_evidences.append(verified)
        
        # Step 6: Extract limitations
        limitations = lim_ext.extract_limitations(enhanced_paper, full_text)
        assert len(limitations) > 0
        
        # Step 7: Cluster limitations
        clusters = lim_cluster.cluster_limitations(limitations)
        assert len(clusters) > 0
        
        # Step 8: Generate gaps
        gaps = gap_finder.find_gaps([enhanced_paper], limitations)
        assert len(gaps) > 0
        
        # Verify no absolute novelty claims
        for gap in gaps:
            assert "novel" not in gap.description.lower() or "potential" in gap.description.lower()
            assert gap.uncertainty_expressed is True
        
        # Step 9: Generate directions
        min_evidence = [
            Evidence(
                id=f"min_ev_{i}",
                paper_doi=paper.doi,
                content=ev.content,
                quote=ev.quote or "",
                context=ev.context,
                verification_state=ev.verification_state,
                confidence=ev.confidence,
                provenance={},
                created_at=datetime.now()
            )
            for i, ev in enumerate(verified_evidences[:3])
        ]
        
        directions = direction_gen.generate_directions([enhanced_paper], gaps, min_evidence)
        assert len(directions) > 0
        
        # Final assertion: pipeline completed successfully
        assert True

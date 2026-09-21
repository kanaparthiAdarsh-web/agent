"""Storage repositories for research data - using canonical app.schemas."""

import logging
import sqlite3
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import json

# Use canonical schemas from app.schemas
from app.schemas import (
    PaperMetadata, PaperCard, ChunkData, SourceClaim as Evidence,
    MethodologyComparison, DatasetComparison, ResultsComparison,
    PaperComparisonMatrix, LimitationExtraction as Limitation, LimitationCluster,
    GapCandidate, GapVerification, Counterevidence, ResearchDirection,
    GapAnalysisResult, ResearchJob, WorkflowStatus, WorkflowStep, StepResult
)
from app.exceptions import StorageError

logger = logging.getLogger(__name__)


class SQLiteConnection:
    """Manages SQLite database connection and schema."""
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
    
    def connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            self._conn.row_factory = sqlite3.Row
            self._create_tables()
        return self._conn
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                venue TEXT,
                doi TEXT,
                arxiv_id TEXT,
                url TEXT,
                pdf_url TEXT,
                abstract TEXT,
                source TEXT,
                citation_count INTEGER,
                open_access BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # PaperCards table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_cards (
                paper_id TEXT PRIMARY KEY,
                research_problem TEXT,
                research_question TEXT,
                methodology TEXT,
                models TEXT,
                datasets TEXT,
                evaluation_metrics TEXT,
                key_results TEXT,
                contributions TEXT,
                limitations TEXT,
                future_work TEXT,
                terminology TEXT,
                evidence_references TEXT,
                extraction_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """)
        
        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT,
                text TEXT,
                section_type TEXT,
                section_title TEXT,
                page_numbers TEXT,
                position_in_section INTEGER,
                is_partial BOOLEAN,
                metadata TEXT,
                embedding_semantic BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """)
        
        # Claims/Evidence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                paper_id TEXT,
                chunk_id TEXT,
                claim_type TEXT,
                content TEXT,
                exact_quote TEXT,
                page_numbers TEXT,
                section TEXT,
                confidence REAL,
                entailment_score REAL,
                verification_status TEXT,
                verification_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
            )
        """)
        
        # Comparison table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY,
                research_topic TEXT,
                paper_ids TEXT,
                methodology TEXT,
                datasets TEXT,
                results TEXT,
                limitations_summary TEXT,
                summary TEXT,
                key_insights TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Limitations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS limitations (
                limitation_id TEXT PRIMARY KEY,
                paper_id TEXT,
                chunk_id TEXT,
                original_text TEXT,
                normalized_description TEXT,
                category TEXT,
                subcategory TEXT,
                affected_methods TEXT,
                affected_datasets TEXT,
                page_numbers TEXT,
                section TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """)
        
        # Limitation clusters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS limitation_clusters (
                cluster_id TEXT PRIMARY KEY,
                theme TEXT,
                category TEXT,
                description TEXT,
                member_limitations TEXT,
                source_papers TEXT,
                representative_quotes TEXT,
                frequency INTEGER,
                severity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Gaps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gap_candidates (
                gap_id TEXT PRIMARY KEY,
                research_topic TEXT,
                description TEXT,
                category TEXT,
                supporting_papers TEXT,
                supporting_evidence TEXT,
                related_limitation_clusters TEXT,
                affected_methods TEXT,
                affected_datasets TEXT,
                initial_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Gap verifications table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gap_verifications (
                verification_id TEXT PRIMARY KEY,
                gap_candidate_id TEXT,
                coverage_status TEXT,
                supporting_evidence_count INTEGER,
                counterevidence TEXT,
                coverage_summary TEXT,
                reasoning TEXT,
                assessment_confidence REAL,
                novelty_disclaimer TEXT,
                verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (gap_candidate_id) REFERENCES gap_candidates(gap_id)
            )
        """)
        
        # Research directions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_directions (
                direction_id TEXT PRIMARY KEY,
                research_topic TEXT,
                proposed_problem TEXT,
                motivation TEXT,
                suggested_methodology TEXT,
                possible_datasets TEXT,
                candidate_models TEXT,
                evaluation_strategy TEXT,
                evaluation_metrics TEXT,
                feasibility_considerations TEXT,
                required_resources TEXT,
                assumptions TEXT,
                supporting_evidence TEXT,
                evidence_paper_ids TEXT,
                related_gap_ids TEXT,
                priority TEXT,
                novelty_potential TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Research jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_jobs (
                job_id TEXT PRIMARY KEY,
                topic TEXT,
                subquestions TEXT,
                year_min INTEGER,
                year_max INTEGER,
                conferences TEXT,
                max_papers INTEGER,
                status TEXT,
                progress_percentage REAL,
                current_step TEXT,
                step_results TEXT,
                discovered_paper_ids TEXT,
                processed_paper_ids TEXT,
                failed_paper_ids TEXT,
                paper_card_ids TEXT,
                evidence_ids TEXT,
                comparison_id TEXT,
                limitation_cluster_ids TEXT,
                gap_candidate_ids TEXT,
                verification_ids TEXT,
                direction_ids TEXT,
                errors TEXT,
                warnings TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                updated_at TIMESTAMP,
                completed_at TIMESTAMP,
                can_resume BOOLEAN,
                checkpoint_data TEXT
            )
        """)
        
        conn.commit()
        logger.info("Database tables created at %s", self.db_path)


class PaperRepository:
    """Repository for paper metadata operations."""
    
    def __init__(self, db: SQLiteConnection):
        self.db = db
    
    def save(self, paper: PaperMetadata) -> bool:
        """Save a paper to the database."""
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO papers 
                (id, title, authors, year, venue, doi, arxiv_id, url, pdf_url, 
                 abstract, source, citation_count, open_access)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper.id,
                paper.title,
                json.dumps(paper.authors) if paper.authors else None,
                paper.year,
                paper.venue,
                paper.doi,
                paper.arxiv_id,
                paper.url,
                paper.pdf_url,
                paper.abstract,
                paper.source.value if paper.source else None,
                paper.citation_count,
                paper.open_access,
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error("Failed to save paper %s: %s", paper.id, e)
            return False
    
    def get(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get a paper by ID."""
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_paper(row)
            
        except Exception as e:
            logger.error("Failed to get paper %s: %s", paper_id, e)
            return None
    
    def get_all(self) -> List[PaperMetadata]:
        """Get all papers."""
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM papers")
            rows = cursor.fetchall()
            
            return [self._row_to_paper(row) for row in rows]
            
        except Exception as e:
            logger.error("Failed to get all papers: %s", e)
            return []
    
    def _row_to_paper(self, row: sqlite3.Row) -> PaperMetadata:
        """Convert database row to PaperMetadata."""
        return PaperMetadata(
            id=row["id"],
            title=row["title"] or "",
            authors=json.loads(row["authors"]) if row["authors"] else None,
            year=row["year"],
            venue=row["venue"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            url=row["url"],
            pdf_url=row["pdf_url"],
            abstract=row["abstract"],
            source=row["source"],
            citation_count=row["citation_count"],
            open_access=row["open_access"],
        )


class PaperCardRepository:
    """Repository for PaperCard operations."""
    
    def __init__(self, db: SQLiteConnection):
        self.db = db
    
    def save(self, card: PaperCard) -> bool:
        """Save a PaperCard to the database."""
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO paper_cards 
                (paper_id, research_problem, research_question, methodology,
                 models, datasets, evaluation_metrics, key_results, contributions,
                 limitations, future_work, terminology, evidence_references,
                 extraction_confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card.paper_id,
                card.research_problem,
                card.research_question,
                card.methodology,
                json.dumps(card.models),
                json.dumps(card.datasets),
                json.dumps(card.evaluation_metrics),
                json.dumps(card.key_results),
                json.dumps(card.contributions),
                json.dumps(card.limitations),
                json.dumps(card.future_work),
                json.dumps(card.terminology),
                json.dumps(card.evidence_references),
                card.extraction_confidence,
                card.created_at.isoformat() if card.created_at else None,
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error("Failed to save PaperCard %s: %s", card.paper_id, e)
            return False
    
    def get(self, paper_id: str) -> Optional[PaperCard]:
        """Get a PaperCard by paper ID."""
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM paper_cards WHERE paper_id = ?", (paper_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_card(row)
            
        except Exception as e:
            logger.error("Failed to get PaperCard %s: %s", paper_id, e)
            return None
    
    def _row_to_card(self, row: sqlite3.Row) -> PaperCard:
        """Convert database row to PaperCard."""
        return PaperCard(
            paper_id=row["paper_id"],
            research_problem=row["research_problem"],
            research_question=row["research_question"],
            methodology=row["methodology"],
            models=json.loads(row["models"]) if row["models"] else [],
            datasets=json.loads(row["datasets"]) if row["datasets"] else [],
            evaluation_metrics=json.loads(row["evaluation_metrics"]) if row["evaluation_metrics"] else [],
            key_results=json.loads(row["key_results"]) if row["key_results"] else [],
            contributions=json.loads(row["contributions"]) if row["contributions"] else [],
            limitations=json.loads(row["limitations"]) if row["limitations"] else [],
            future_work=json.loads(row["future_work"]) if row["future_work"] else [],
            terminology=json.loads(row["terminology"]) if row["terminology"] else {},
            evidence_references=json.loads(row["evidence_references"]) if row["evidence_references"] else [],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            extraction_confidence=row["extraction_confidence"] or 0.0,
        )


class JobRepository:
    """Repository for research job operations."""
    
    def __init__(self, db: SQLiteConnection):
        self.db = db
    
    def save(self, job: ResearchJob) -> bool:
        """Save a research job to the database."""
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO research_jobs 
                (job_id, topic, subquestions, year_min, year_max, conferences,
                 max_papers, status, progress_percentage, current_step,
                 step_results, discovered_paper_ids, processed_paper_ids,
                 failed_paper_ids, paper_card_ids, evidence_ids, comparison_id,
                 limitation_cluster_ids, gap_candidate_ids, verification_ids,
                 direction_ids, errors, warnings, created_at, started_at,
                 updated_at, completed_at, can_resume, checkpoint_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id,
                job.topic,
                json.dumps(job.subquestions) if job.subquestions else None,
                job.year_min,
                job.year_max,
                json.dumps(job.conferences) if job.conferences else None,
                job.max_papers,
                job.status.value,
                job.progress_percentage,
                job.current_step.value if job.current_step else None,
                json.dumps({k: v.model_dump() for k, v in job.step_results.items()}),
                json.dumps(job.discovered_paper_ids),
                json.dumps(job.processed_paper_ids),
                json.dumps(job.failed_paper_ids),
                json.dumps(job.paper_card_ids),
                json.dumps(job.evidence_ids),
                job.comparison_id,
                json.dumps(job.limitation_cluster_ids),
                json.dumps(job.gap_candidate_ids),
                json.dumps(job.verification_ids),
                json.dumps(job.direction_ids),
                json.dumps(job.errors),
                json.dumps(job.warnings),
                job.created_at.isoformat() if job.created_at else None,
                job.started_at.isoformat() if job.started_at else None,
                job.updated_at.isoformat() if job.updated_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.can_resume,
                json.dumps(job.checkpoint_data) if job.checkpoint_data else None,
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error("Failed to save job %s: %s", job.job_id, e)
            return False
    
    def get(self, job_id: str) -> Optional[ResearchJob]:
        """Get a research job by ID."""
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM research_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_job(row)
            
        except Exception as e:
            logger.error("Failed to get job %s: %s", job_id, e)
            return None
    
    def get_all(self) -> List[ResearchJob]:
        """Get all research jobs."""
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM research_jobs ORDER BY created_at DESC")
            rows = cursor.fetchall()
            
            return [self._row_to_job(row) for row in rows]
            
        except Exception as e:
            logger.error("Failed to get all jobs: %s", e)
            return []
    
    def _row_to_job(self, row: sqlite3.Row) -> ResearchJob:
        """Convert database row to ResearchJob."""
        from app.schemas.workflow import WorkflowStep
        
        step_results = {}
        if row["step_results"]:
            sr_data = json.loads(row["step_results"])
            for k, v in sr_data.items():
                step_results[k] = v  # Could reconstruct StepResult objects here
        
        return ResearchJob(
            job_id=row["job_id"],
            topic=row["topic"],
            subquestions=json.loads(row["subquestions"]) if row["subquestions"] else None,
            year_min=row["year_min"],
            year_max=row["year_max"],
            conferences=json.loads(row["conferences"]) if row["conferences"] else None,
            max_papers=row["max_papers"],
            status=WorkflowStatus(row["status"]),
            progress_percentage=row["progress_percentage"] or 0.0,
            current_step=WorkflowStep(row["current_step"]) if row["current_step"] else None,
            step_results=step_results,
            discovered_paper_ids=json.loads(row["discovered_paper_ids"]) or [],
            processed_paper_ids=json.loads(row["processed_paper_ids"]) or [],
            failed_paper_ids=json.loads(row["failed_paper_ids"]) or [],
            paper_card_ids=json.loads(row["paper_card_ids"]) or [],
            evidence_ids=json.loads(row["evidence_ids"]) or [],
            comparison_id=row["comparison_id"],
            limitation_cluster_ids=json.loads(row["limitation_cluster_ids"]) or [],
            gap_candidate_ids=json.loads(row["gap_candidate_ids"]) or [],
            verification_ids=json.loads(row["verification_ids"]) or [],
            direction_ids=json.loads(row["direction_ids"]) or [],
            errors=json.loads(row["errors"]) or [],
            warnings=json.loads(row["warnings"]) or [],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            can_resume=row["can_resume"],
            checkpoint_data=json.loads(row["checkpoint_data"]) if row["checkpoint_data"] else None,
        )


# Convenience functions for creating repositories
_db_instance: Optional[SQLiteConnection] = None


def get_database(db_path: str = "research_agent.db") -> SQLiteConnection:
    """Get or create database connection."""
    global _db_instance
    if _db_instance is None:
        _db_instance = SQLiteConnection(db_path)
    return _db_instance


def get_repositories(db_path: str = "research_agent.db"):
    """Get all repositories with a shared database connection."""
    db = get_database(db_path)
    return {
        "papers": PaperRepository(db),
        "paper_cards": PaperCardRepository(db),
        "jobs": JobRepository(db),
    }

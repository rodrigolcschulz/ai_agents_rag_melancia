"""
Interaction Logger for RAG system.

Logs all RAG interactions: queries, responses, retrieved docs, feedback.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class InteractionLogger:
    """Logs RAG interactions for evaluation and analysis."""
    
    def __init__(self, db_path: str = "data/evaluation/interactions.db"):
        """
        Initialize logger.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                model_name TEXT,
                provider TEXT,
                latency_seconds REAL,
                user_id TEXT,
                session_id TEXT,
                metadata TEXT
            )
        """)
        
        # Retrieved documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieved_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT NOT NULL,
                doc_id TEXT,
                content TEXT NOT NULL,
                score REAL,
                rank INTEGER,
                metadata TEXT,
                FOREIGN KEY (interaction_id) REFERENCES interactions (interaction_id)
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                rating INTEGER,
                comment TEXT,
                FOREIGN KEY (interaction_id) REFERENCES interactions (interaction_id)
            )
        """)
        
        # Metrics table (for automated metrics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                FOREIGN KEY (interaction_id) REFERENCES interactions (interaction_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_interaction(
        self,
        query: str,
        response: str,
        retrieved_docs: List[Dict],
        model_name: str,
        provider: str,
        latency_seconds: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log a RAG interaction.
        
        Args:
            query: User query
            response: RAG response
            retrieved_docs: List of retrieved documents with scores
            model_name: Name of LLM used
            provider: Provider (openai, ollama, etc)
            latency_seconds: Total latency
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata
        
        Returns:
            interaction_id: Unique ID for this interaction
        """
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert interaction
        cursor.execute("""
            INSERT INTO interactions (
                interaction_id, timestamp, query, response,
                model_name, provider, latency_seconds,
                user_id, session_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction_id,
            timestamp,
            query,
            response,
            model_name,
            provider,
            latency_seconds,
            user_id,
            session_id,
            json.dumps(metadata) if metadata else None
        ))
        
        # Insert retrieved documents
        for rank, doc in enumerate(retrieved_docs, start=1):
            cursor.execute("""
                INSERT INTO retrieved_docs (
                    interaction_id, doc_id, content, score, rank, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                doc.get("id"),
                doc.get("content", ""),
                doc.get("score"),
                rank,
                json.dumps(doc.get("metadata", {}))
            ))
        
        conn.commit()
        conn.close()
        
        return interaction_id
    
    def log_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None
    ):
        """
        Log user feedback for an interaction.
        
        Args:
            interaction_id: ID of the interaction
            feedback_type: Type of feedback ("positive", "negative", "neutral")
            rating: Optional numerical rating (1-5)
            comment: Optional text comment
        """
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback (
                interaction_id, timestamp, feedback_type, rating, comment
            ) VALUES (?, ?, ?, ?, ?)
        """, (interaction_id, timestamp, feedback_type, rating, comment))
        
        conn.commit()
        conn.close()
    
    def log_metrics(self, interaction_id: str, metrics: Dict[str, float]):
        """
        Log automated metrics for an interaction.
        
        Args:
            interaction_id: ID of the interaction
            metrics: Dictionary of metric_name -> value
        """
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in metrics.items():
            cursor.execute("""
                INSERT INTO metrics (
                    interaction_id, timestamp, metric_name, metric_value
                ) VALUES (?, ?, ?, ?)
            """, (interaction_id, timestamp, metric_name, metric_value))
        
        conn.commit()
        conn.close()
    
    def get_interaction(self, interaction_id: str) -> Optional[Dict]:
        """Get full interaction details including docs and feedback."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get interaction
        cursor.execute("""
            SELECT * FROM interactions WHERE interaction_id = ?
        """, (interaction_id,))
        
        interaction = cursor.fetchone()
        if not interaction:
            conn.close()
            return None
        
        interaction_dict = dict(interaction)
        
        # Get retrieved docs
        cursor.execute("""
            SELECT * FROM retrieved_docs 
            WHERE interaction_id = ? 
            ORDER BY rank
        """, (interaction_id,))
        
        interaction_dict["retrieved_docs"] = [dict(row) for row in cursor.fetchall()]
        
        # Get feedback
        cursor.execute("""
            SELECT * FROM feedback WHERE interaction_id = ?
        """, (interaction_id,))
        
        interaction_dict["feedback"] = [dict(row) for row in cursor.fetchall()]
        
        # Get metrics
        cursor.execute("""
            SELECT * FROM metrics WHERE interaction_id = ?
        """, (interaction_id,))
        
        interaction_dict["metrics"] = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return interaction_dict
    
    def get_recent_interactions(self, limit: int = 100) -> List[Dict]:
        """Get recent interactions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM interactions 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        interactions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return interactions
    
    def get_stats(self) -> Dict:
        """Get overall statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total interactions
        cursor.execute("SELECT COUNT(*) FROM interactions")
        stats["total_interactions"] = cursor.fetchone()[0]
        
        # Average latency (geral)
        cursor.execute("SELECT AVG(latency_seconds) FROM interactions")
        stats["avg_latency"] = cursor.fetchone()[0] or 0
        
        # Latency stats por provider (SEM PERCENTILE_CONT)
        cursor.execute("""
            SELECT 
                provider,
                COUNT(*) as count,
                AVG(latency_seconds) as avg_latency,
                MIN(latency_seconds) as min_latency,
                MAX(latency_seconds) as max_latency
            FROM interactions 
            WHERE provider IS NOT NULL
            GROUP BY provider
        """)
        stats["latency_by_provider"] = {}
        for row in cursor.fetchall():
            provider = row[0]
            stats["latency_by_provider"][provider] = {
                "count": row[1],
                "avg": row[2],
                "min": row[3],
                "max": row[4]
            }
            
            # Calcular percentis manualmente para cada provider
            cursor.execute("""
                SELECT latency_seconds 
                FROM interactions 
                WHERE provider = ?
                ORDER BY latency_seconds
            """, (provider,))
            
            latencies = [r[0] for r in cursor.fetchall()]
            if latencies:
                n = len(latencies)
                stats["latency_by_provider"][provider]["p50"] = latencies[int(n * 0.50)] if n > 0 else 0
                stats["latency_by_provider"][provider]["p95"] = latencies[int(n * 0.95)] if n > 1 else latencies[0]
                stats["latency_by_provider"][provider]["p99"] = latencies[int(n * 0.99)] if n > 1 else latencies[0]
        
        # Latency últimas 24h
        cursor.execute("""
            SELECT AVG(latency_seconds) 
            FROM interactions 
            WHERE datetime(timestamp) > datetime('now', '-1 day')
        """)
        stats["avg_latency_24h"] = cursor.fetchone()[0] or 0
        
        # Slow queries (> 10s)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM interactions 
            WHERE latency_seconds > 10.0
        """)
        stats["slow_queries_count"] = cursor.fetchone()[0]
        
        # Feedback counts
        cursor.execute("""
            SELECT feedback_type, COUNT(*) 
            FROM feedback 
            GROUP BY feedback_type
        """)
        stats["feedback_counts"] = dict(cursor.fetchall())
        
        # Average rating
        cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL")
        stats["avg_rating"] = cursor.fetchone()[0]
        
        # Most used models
        cursor.execute("""
            SELECT model_name, COUNT(*) as count 
            FROM interactions 
            GROUP BY model_name 
            ORDER BY count DESC 
            LIMIT 5
        """)
        stats["top_models"] = dict(cursor.fetchall())
        
        # Provider distribution
        cursor.execute("""
            SELECT provider, COUNT(*) as count
            FROM interactions
            WHERE provider IS NOT NULL
            GROUP BY provider
        """)
        stats["provider_distribution"] = dict(cursor.fetchall())
        
        conn.close()
        return stats


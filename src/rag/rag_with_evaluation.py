"""
RAG system with integrated evaluation loops.

Wraps existing RAG to add automatic logging and evaluation.
"""

import time
from typing import Dict, List, Optional
from src.evaluation.interaction_logger import InteractionLogger
from src.evaluation.rag_evaluator import RAGEvaluator


class RAGWithEvaluation:
    """RAG wrapper that logs and evaluates all interactions."""
    
    def __init__(
        self,
        rag_system,
        logger: Optional[InteractionLogger] = None,
        evaluator: Optional[RAGEvaluator] = None,
        enable_logging: bool = True,
        enable_evaluation: bool = True
    ):
        """
        Initialize RAG with evaluation.
        
        Args:
            rag_system: Your existing RAG system (must have .query() method)
            logger: InteractionLogger instance
            evaluator: RAGEvaluator instance
            enable_logging: Whether to log interactions
            enable_evaluation: Whether to calculate metrics
        """
        self.rag = rag_system
        self.logger = logger or InteractionLogger()
        self.evaluator = evaluator or RAGEvaluator()
        self.enable_logging = enable_logging
        self.enable_evaluation = enable_evaluation
    
    def query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Query RAG system with automatic logging and evaluation.
        
        Args:
            query: User query
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata
        
        Returns:
            Dict with:
                - answer: Generated answer
                - sources: Retrieved documents
                - interaction_id: ID for feedback
                - metrics: Evaluation metrics (if enabled)
        """
        start_time = time.time()
        
        # Query RAG system
        # Adapt this to your RAG's interface
        rag_result = self.rag.query(query)
        
        latency = time.time() - start_time
        
        # Extract components (adapt to your RAG's response format)
        if isinstance(rag_result, dict):
            answer = rag_result.get("answer", str(rag_result))
            sources = rag_result.get("sources", [])
            model_name = rag_result.get("model_name", "unknown")
            provider = rag_result.get("provider", "unknown")
        else:
            answer = str(rag_result)
            sources = []
            model_name = "unknown"
            provider = "unknown"
        
        # Format retrieved docs for logging
        retrieved_docs = []
        for i, source in enumerate(sources):
            if isinstance(source, dict):
                retrieved_docs.append({
                    "id": source.get("id", f"doc_{i}"),
                    "content": source.get("content", str(source)),
                    "score": source.get("score"),
                    "metadata": source.get("metadata", {})
                })
            else:
                retrieved_docs.append({
                    "id": f"doc_{i}",
                    "content": str(source),
                    "score": None,
                    "metadata": {}
                })
        
        interaction_id = None
        metrics = {}
        
        # Log interaction
        if self.enable_logging:
            interaction_id = self.logger.log_interaction(
                query=query,
                response=answer,
                retrieved_docs=retrieved_docs,
                model_name=model_name,
                provider=provider,
                latency_seconds=latency,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata
            )
        
        # Evaluate
        if self.enable_evaluation and retrieved_docs:
            doc_contents = [doc["content"] for doc in retrieved_docs]
            
            metrics = self.evaluator.evaluate_interaction(
                query=query,
                response=answer,
                retrieved_docs=doc_contents
            )
            
            # Log metrics
            if self.enable_logging and interaction_id:
                self.logger.log_metrics(interaction_id, metrics)
        
        return {
            "answer": answer,
            "sources": sources,
            "interaction_id": interaction_id,
            "metrics": metrics,
            "latency": latency
        }
    
    def submit_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None
    ):
        """
        Submit user feedback for an interaction.
        
        Args:
            interaction_id: ID from query() response
            feedback_type: "positive", "negative", or "neutral"
            rating: Optional 1-5 rating
            comment: Optional text comment
        """
        if not self.enable_logging:
            return
        
        self.logger.log_feedback(
            interaction_id=interaction_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment
        )
    
    def get_stats(self) -> Dict:
        """Get overall statistics."""
        if not self.enable_logging:
            return {}
        
        return self.logger.get_stats()
    
    def get_recent_interactions(self, limit: int = 50) -> List[Dict]:
        """Get recent interactions."""
        if not self.enable_logging:
            return []
        
        return self.logger.get_recent_interactions(limit=limit)


# Example usage:
"""
from src.rag.your_rag_system import YourRAGSystem
from src.rag.rag_with_evaluation import RAGWithEvaluation

# Your existing RAG
rag = YourRAGSystem()

# Wrap with evaluation
rag_eval = RAGWithEvaluation(rag)

# Query (automatically logged and evaluated)
result = rag_eval.query("O que é ACOS?")
print(result["answer"])
print(f"Metrics: {result['metrics']}")

# User feedback
rag_eval.submit_feedback(
    interaction_id=result["interaction_id"],
    feedback_type="positive",
    rating=5
)

# Get stats
stats = rag_eval.get_stats()
print(f"Total interactions: {stats['total_interactions']}")
print(f"Average rating: {stats['avg_rating']}")
"""


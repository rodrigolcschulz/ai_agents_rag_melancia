"""
Evaluation module for RAG system.

Provides:
- Interaction logging
- RAG metrics (RAGAS framework)
- User feedback collection
- Analysis and reporting
"""

from .interaction_logger import InteractionLogger
from .rag_evaluator import RAGEvaluator

__all__ = ["InteractionLogger", "RAGEvaluator"]

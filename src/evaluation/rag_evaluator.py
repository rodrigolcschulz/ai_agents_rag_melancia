"""
RAG Evaluator using RAGAS metrics.

Evaluates RAG system quality automatically:
- Retrieval quality (context precision, recall, relevancy)
- Generation quality (faithfulness, answer relevancy)
"""

from typing import Dict, List, Optional
import re


class RAGEvaluator:
    """Evaluates RAG interactions using automated metrics."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_interaction(
        self,
        query: str,
        response: str,
        retrieved_docs: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a RAG interaction.
        
        Args:
            query: User query
            response: Generated response
            retrieved_docs: List of retrieved document contents
            ground_truth: Optional ground truth answer for comparison
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Answer Relevancy: Does response actually answer the query?
        metrics["answer_relevancy"] = self._answer_relevancy(query, response)
        
        # Faithfulness: Is response based on retrieved docs?
        metrics["faithfulness"] = self._faithfulness(response, retrieved_docs)
        
        # Context Relevancy: Are retrieved docs relevant to query?
        metrics["context_relevancy"] = self._context_relevancy(query, retrieved_docs)
        
        # Response length (informational)
        metrics["response_length"] = len(response)
        metrics["num_docs_retrieved"] = len(retrieved_docs)
        
        # If ground truth provided, calculate similarity
        if ground_truth:
            metrics["answer_correctness"] = self._answer_correctness(response, ground_truth)
        
        return metrics
    
    def _answer_relevancy(self, query: str, response: str) -> float:
        """
        Measure if response answers the query.
        
        Simple heuristic: keyword overlap and length check.
        TODO: Use LLM-based evaluation for better accuracy.
        """
        # Extract keywords from query (simple: words > 3 chars, excluding common words)
        common_words = {"que", "como", "onde", "quando", "quem", "por", "para", "com", "sem", "sobre", "isso", "este", "esta"}
        
        query_words = set(
            word.lower() 
            for word in re.findall(r'\w+', query) 
            if len(word) > 3 and word.lower() not in common_words
        )
        
        response_words = set(
            word.lower() 
            for word in re.findall(r'\w+', response)
        )
        
        # Keyword overlap
        if not query_words:
            return 0.5  # No keywords to match
        
        overlap = len(query_words & response_words) / len(query_words)
        
        # Penalize very short responses
        if len(response) < 50:
            overlap *= 0.5
        
        # Cap at 1.0
        return min(overlap, 1.0)
    
    def _faithfulness(self, response: str, retrieved_docs: List[str]) -> float:
        """
        Measure if response is grounded in retrieved documents.
        
        Checks if statements in response appear in docs.
        TODO: Use LLM to check claim-by-claim.
        """
        if not retrieved_docs:
            return 0.0
        
        # Combine all docs
        docs_text = " ".join(retrieved_docs).lower()
        
        # Split response into sentences
        sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 20]
        
        if not sentences:
            return 0.5
        
        # Check how many sentences have content overlap with docs
        grounded_sentences = 0
        
        for sentence in sentences:
            # Extract content words from sentence
            words = [
                word.lower() 
                for word in re.findall(r'\w+', sentence) 
                if len(word) > 4
            ]
            
            if not words:
                continue
            
            # Check if at least 50% of words appear in docs
            found_words = sum(1 for word in words if word in docs_text)
            
            if len(words) > 0 and found_words / len(words) >= 0.5:
                grounded_sentences += 1
        
        return grounded_sentences / len(sentences) if sentences else 0.5
    
    def _context_relevancy(self, query: str, retrieved_docs: List[str]) -> float:
        """
        Measure if retrieved documents are relevant to query.
        
        Checks keyword overlap between query and docs.
        """
        if not retrieved_docs:
            return 0.0
        
        # Extract keywords from query
        query_words = set(
            word.lower() 
            for word in re.findall(r'\w+', query) 
            if len(word) > 3
        )
        
        if not query_words:
            return 0.5
        
        # Check each doc for relevance
        relevant_docs = 0
        
        for doc in retrieved_docs:
            doc_words = set(
                word.lower() 
                for word in re.findall(r'\w+', doc)
            )
            
            # If doc contains >30% of query keywords, consider it relevant
            overlap = len(query_words & doc_words) / len(query_words)
            
            if overlap >= 0.3:
                relevant_docs += 1
        
        return relevant_docs / len(retrieved_docs)
    
    def _answer_correctness(self, response: str, ground_truth: str) -> float:
        """
        Compare response to ground truth.
        
        Simple word overlap metric.
        TODO: Use semantic similarity (sentence embeddings).
        """
        # Extract words
        response_words = set(
            word.lower() 
            for word in re.findall(r'\w+', response) 
            if len(word) > 3
        )
        
        truth_words = set(
            word.lower() 
            for word in re.findall(r'\w+', ground_truth) 
            if len(word) > 3
        )
        
        if not truth_words:
            return 0.5
        
        # Precision and recall
        intersection = response_words & truth_words
        
        precision = len(intersection) / len(response_words) if response_words else 0
        recall = len(intersection) / len(truth_words) if truth_words else 0
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def evaluate_batch(
        self,
        interactions: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate multiple interactions and return aggregated metrics.
        
        Args:
            interactions: List of interaction dicts with query, response, docs
        
        Returns:
            Aggregated metrics (averages)
        """
        all_metrics = []
        
        for interaction in interactions:
            metrics = self.evaluate_interaction(
                query=interaction["query"],
                response=interaction["response"],
                retrieved_docs=interaction.get("retrieved_docs", []),
                ground_truth=interaction.get("ground_truth")
            )
            all_metrics.append(metrics)
        
        # Aggregate (average)
        if not all_metrics:
            return {}
        
        aggregated = {}
        
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            aggregated[f"avg_{key}"] = sum(values) / len(values) if values else 0
        
        return aggregated


# TODO: Advanced metrics using LLM-based evaluation
class LLMBasedEvaluator:
    """
    Use LLM to evaluate RAG quality (more accurate but slower/costly).
    
    Can be implemented later for:
    - Better faithfulness checking
    - Answer relevancy with reasoning
    - Hallucination detection
    """
    
    def __init__(self, llm):
        """
        Initialize with an LLM.
        
        Args:
            llm: Language model for evaluation
        """
        self.llm = llm
    
    def evaluate_faithfulness(self, response: str, docs: List[str]) -> float:
        """
        Use LLM to check if response is faithful to documents.
        
        Prompt LLM: "Given these documents, is this response factually correct?"
        """
        # TODO: Implement
        pass
    
    def evaluate_answer_relevancy(self, query: str, response: str) -> float:
        """
        Use LLM to check if response answers the query.
        
        Prompt LLM: "Does this response adequately answer the question?"
        """
        # TODO: Implement
        pass


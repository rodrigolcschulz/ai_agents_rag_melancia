"""
Sistema de benchmark para avaliar diferentes LLMs em tarefas de RAG.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain.chains import ConversationalRetrievalChain
import pandas as pd


# Perguntas padrão para teste
DEFAULT_TEST_QUESTIONS = [
    "O que é ACOS e como calcular?",
    "Como funcionam os Product Ads no Mercado Livre?",
    "Quais são as melhores práticas para anúncios patrocinados?",
    "Como otimizar palavras-chave para aumentar vendas?",
    "Como melhorar a visibilidade dos anúncios?",
]


@dataclass
class BenchmarkResult:
    """Resultado de benchmark para um modelo."""
    provider: str
    model_name: str
    avg_latency: float
    total_time: float
    success_rate: float
    avg_quality: float = 0.0
    avg_relevance: float = 0.0
    avg_tokens: int = 0
    num_questions: int = 0
    error: Optional[str] = None
    details: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "avg_latency": self.avg_latency,
            "total_time": self.total_time,
            "success_rate": self.success_rate,
            "avg_quality": self.avg_quality,
            "avg_relevance": self.avg_relevance,
            "avg_tokens": self.avg_tokens,
            "num_questions": self.num_questions,
            "error": self.error
        }


class ModelBenchmark:
    """Sistema de benchmark para modelos LLM."""
    
    def __init__(self, retriever, memory):
        """
        Inicializa o benchmark.
        
        Args:
            retriever: Retriever do RAG
            memory: Memória conversacional
        """
        self.retriever = retriever
        self.memory = memory
        self.models = {}
        self.results = []
    
    def add_model(self, provider: str, model_name: str, llm: Any):
        """
        Adiciona um modelo para testar.
        
        Args:
            provider: Nome do provider (openai, ollama, etc)
            model_name: Nome do modelo
            llm: Instância do LLM
        """
        key = f"{provider}::{model_name}"
        self.models[key] = {
            "provider": provider,
            "model_name": model_name,
            "llm": llm
        }
    
    def run(
        self, 
        questions: List[str],
        evaluate_quality: bool = False,
        verbose: bool = False
    ) -> List[BenchmarkResult]:
        """
        Executa benchmark em todos os modelos.
        
        Args:
            questions: Lista de perguntas para testar
            evaluate_quality: Se True, avalia qualidade das respostas
            verbose: Se True, imprime progresso
        
        Returns:
            Lista de BenchmarkResults
        """
        results = []
        
        for key, model_info in self.models.items():
            if verbose:
                print(f"\n{'='*70}")
                print(f"Testando: {model_info['provider']} - {model_info['model_name']}")
                print(f"{'='*70}")
            
            result = self._test_model(
                model_info['provider'],
                model_info['model_name'],
                model_info['llm'],
                questions,
                evaluate_quality,
                verbose
            )
            
            results.append(result)
            
            if verbose:
                print(f"\nResultado: Latência média = {result.avg_latency:.2f}s")
                print(f"Taxa de sucesso: {result.success_rate:.1%}")
        
        self.results = results
        return results
    
    def _test_model(
        self,
        provider: str,
        model_name: str,
        llm: Any,
        questions: List[str],
        evaluate_quality: bool,
        verbose: bool
    ) -> BenchmarkResult:
        """Testa um modelo específico."""
        latencies = []
        successes = 0
        qualities = []
        relevances = []
        details = []
        
        # Criar chain para este modelo
        try:
            from agent.prompt import get_prompt_template
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={
                    "prompt": get_prompt_template(),
                    "document_separator": "\n\n---\n\n"
                },
                return_source_documents=True,
                verbose=False
            )
        except Exception as e:
            return BenchmarkResult(
                provider=provider,
                model_name=model_name,
                avg_latency=0,
                total_time=0,
                success_rate=0,
                num_questions=len(questions),
                error=str(e)
            )
        
        # Testar cada pergunta
        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"  [{i}/{len(questions)}] {question[:50]}...")
            
            start_time = time.time()
            try:
                # Executar query
                resultado = qa_chain.invoke({"question": question})
                latency = time.time() - start_time
                
                # Extrair resposta
                if isinstance(resultado, dict) and 'answer' in resultado:
                    resposta = resultado['answer']
                    source_docs = resultado.get('source_documents', [])
                else:
                    resposta = str(resultado)
                    source_docs = []
                
                latencies.append(latency)
                successes += 1
                
                # Avaliar qualidade (simplificado)
                if evaluate_quality:
                    quality = self._evaluate_quality(resposta)
                    relevance = self._evaluate_relevance(question, resposta)
                    qualities.append(quality)
                    relevances.append(relevance)
                
                details.append({
                    "question": question,
                    "latency": latency,
                    "success": True,
                    "response_length": len(resposta),
                    "num_sources": len(source_docs)
                })
                
                if verbose:
                    print(f"      ✓ {latency:.2f}s")
                
            except Exception as e:
                latencies.append(0)
                details.append({
                    "question": question,
                    "success": False,
                    "error": str(e)
                })
                
                if verbose:
                    print(f"      ✗ Erro: {str(e)[:50]}")
        
        # Calcular métricas
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_time = sum(latencies)
        success_rate = successes / len(questions) if questions else 0
        avg_quality = sum(qualities) / len(qualities) if qualities else 0
        avg_relevance = sum(relevances) / len(relevances) if relevances else 0
        
        return BenchmarkResult(
            provider=provider,
            model_name=model_name,
            avg_latency=avg_latency,
            total_time=total_time,
            success_rate=success_rate,
            avg_quality=avg_quality,
            avg_relevance=avg_relevance,
            num_questions=len(questions),
            details=details
        )
    
    def _evaluate_quality(self, response: str) -> float:
        """
        Avalia qualidade da resposta (métrica simplificada).
        
        Args:
            response: Resposta do modelo
        
        Returns:
            Score de qualidade (0-1)
        """
        if not response:
            return 0.0
        
        score = 0.0
        
        # Comprimento adequado (50-500 palavras)
        words = len(response.split())
        if 50 <= words <= 500:
            score += 0.3
        elif words > 20:
            score += 0.15
        
        # Tem pontuação
        if any(p in response for p in ['.', '!', '?']):
            score += 0.2
        
        # Tem formatação
        if any(f in response for f in ['\n', '•', '-', '1.', '2.']):
            score += 0.2
        
        # Não é muito curto
        if len(response) > 100:
            score += 0.15
        
        # Palavras-chave relevantes
        keywords = ['retail', 'media', 'anúncio', 'campanha', 'performance', 'produto']
        if any(kw in response.lower() for kw in keywords):
            score += 0.15
        
        return min(score, 1.0)
    
    def _evaluate_relevance(self, question: str, response: str) -> float:
        """
        Avalia relevância da resposta para a pergunta.
        
        Args:
            question: Pergunta
            response: Resposta
        
        Returns:
            Score de relevância (0-1)
        """
        if not response:
            return 0.0
        
        # Extrair palavras importantes
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Remover stopwords
        stopwords = {'o', 'a', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'uma', 'para'}
        question_words -= stopwords
        response_words -= stopwords
        
        # Calcular overlap
        if not question_words:
            return 0.5
        
        overlap = len(question_words & response_words) / len(question_words)
        return min(overlap, 1.0)
    
    def print_report(self):
        """Imprime relatório formatado dos resultados."""
        if not self.results:
            print("Nenhum resultado disponível")
            return
        
        print("\n" + "="*80)
        print("📊 RELATÓRIO DE BENCHMARK")
        print("="*80)
        
        # Ordenar por qualidade
        sorted_results = sorted(self.results, key=lambda x: x.avg_quality, reverse=True)
        
        print(f"\n{'Rank':<6}{'Modelo':<35}{'Latência':<12}{'Qualidade':<12}{'Sucesso':<10}")
        print("-"*75)
        
        for i, r in enumerate(sorted_results, 1):
            model_str = f"{r.provider}::{r.model_name}"[:34]
            latency_str = f"{r.avg_latency:.2f}s"
            quality_str = f"{r.avg_quality:.3f}"
            success_str = f"{r.success_rate:.1%}"
            
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{emoji:<6}{model_str:<35}{latency_str:<12}{quality_str:<12}{success_str:<10}")
        
        print("="*80)
    
    def generate_report(self) -> pd.DataFrame:
        """
        Gera relatório em formato DataFrame.
        
        Returns:
            DataFrame com resultados
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for r in self.results:
            data.append({
                "provider": r.provider,
                "model_name": r.model_name,
                "latency_avg": r.avg_latency,
                "quality_avg": r.avg_quality,
                "relevance_avg": r.avg_relevance,
                "success_rate": r.success_rate,
                "total_time": r.total_time,
                "num_questions": r.num_questions,
                "total_cost": r.avg_tokens * 0.00001 if r.provider == "openai" else 0
            })
        
        return pd.DataFrame(data)


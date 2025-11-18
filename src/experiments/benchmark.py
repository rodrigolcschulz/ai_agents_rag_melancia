"""
Sistema de benchmark para comparação de diferentes LLMs
Avalia performance, qualidade de resposta, latência e custo
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado de um teste de benchmark"""
    model_name: str
    provider: str
    question: str
    answer: str
    latency_seconds: float
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return asdict(self)


class ModelBenchmark:
    """
    Sistema de benchmark para avaliar diferentes LLMs
    
    Exemplo:
        benchmark = ModelBenchmark(retriever, memory)
        
        # Adicionar modelos para testar
        benchmark.add_model("openai", "gpt-4o-mini")
        benchmark.add_model("ollama", "llama3.1:8b")
        
        # Executar benchmark
        results = benchmark.run(test_questions)
        
        # Gerar relatório
        report = benchmark.generate_report()
    """
    
    def __init__(self, retriever, memory, output_dir: str = "data/experiments"):
        """
        Args:
            retriever: Retriever do RAG
            memory: Memória conversacional
            output_dir: Diretório para salvar resultados
        """
        self.retriever = retriever
        self.memory = memory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, tuple[str, Union[ChatOpenAI, OllamaLLM, HuggingFaceHub]]] = {}
        self.results: List[BenchmarkResult] = []
    
    def add_model(self, provider: str, model_name: str, llm: Union[ChatOpenAI, OllamaLLM, HuggingFaceHub]):
        """
        Adiciona um modelo para benchmark
        
        Args:
            provider: Nome do provedor (openai, ollama, huggingface)
            model_name: Nome do modelo
            llm: Instância do LLM
        """
        key = f"{provider}::{model_name}"
        self.models[key] = (provider, llm)
        logger.info(f"Modelo adicionado: {key}")
    
    def run(
        self,
        test_questions: List[str],
        evaluate_quality: bool = True,
        verbose: bool = True
    ) -> List[BenchmarkResult]:
        """
        Executa benchmark em todos os modelos
        
        Args:
            test_questions: Lista de perguntas para testar
            evaluate_quality: Se deve avaliar qualidade das respostas
            verbose: Se deve imprimir progresso
            
        Returns:
            Lista de resultados
        """
        self.results = []
        total_tests = len(self.models) * len(test_questions)
        current = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Iniciando Benchmark: {len(self.models)} modelos x {len(test_questions)} perguntas")
            print(f"{'='*60}\n")
        
        for model_key, (provider, llm) in self.models.items():
            model_name = model_key.split("::")[1]
            
            if verbose:
                print(f"\n📊 Testando: {provider.upper()} - {model_name}")
                print("-" * 60)
            
            # Criar cadeia RAG para este modelo
            try:
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=self.retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=False
                )
            except Exception as e:
                logger.error(f"Erro ao criar cadeia para {model_key}: {e}")
                continue
            
            # Testar cada pergunta
            for question in test_questions:
                current += 1
                progress = (current / total_tests) * 100
                
                if verbose:
                    print(f"  [{current}/{total_tests}] {progress:.1f}% - {question[:50]}...")
                
                result = self._test_single_question(
                    provider=provider,
                    model_name=model_name,
                    qa_chain=qa_chain,
                    question=question,
                    evaluate_quality=evaluate_quality
                )
                
                self.results.append(result)
                
                if verbose and result.error:
                    print(f"    ⚠️  Erro: {result.error}")
                elif verbose:
                    print(f"    ✓ Latência: {result.latency_seconds:.2f}s")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ Benchmark concluído! {len(self.results)} testes realizados")
            print(f"{'='*60}\n")
        
        # Salvar resultados
        self._save_results()
        
        return self.results
    
    def _test_single_question(
        self,
        provider: str,
        model_name: str,
        qa_chain: ConversationalRetrievalChain,
        question: str,
        evaluate_quality: bool
    ) -> BenchmarkResult:
        """Testa uma única pergunta em um modelo"""
        start_time = time.time()
        
        try:
            # Executar pergunta
            response = qa_chain.invoke({"question": question})
            latency = time.time() - start_time
            
            # Extrair resposta
            answer = response.get("answer", str(response))
            
            # Avaliar qualidade (se solicitado)
            quality_score = None
            relevance_score = None
            if evaluate_quality:
                quality_score = self._evaluate_answer_quality(answer)
                relevance_score = self._evaluate_relevance(question, answer)
            
            # Estimar tokens e custo
            tokens = self._estimate_tokens(question, answer)
            cost = self._estimate_cost(provider, model_name, tokens)
            
            return BenchmarkResult(
                model_name=model_name,
                provider=provider,
                question=question,
                answer=answer,
                latency_seconds=latency,
                tokens_used=tokens,
                cost_usd=cost,
                quality_score=quality_score,
                relevance_score=relevance_score
            )
            
        except Exception as e:
            logger.error(f"Erro ao testar {model_name}: {e}")
            return BenchmarkResult(
                model_name=model_name,
                provider=provider,
                question=question,
                answer="",
                latency_seconds=time.time() - start_time,
                error=str(e)
            )
    
    def _evaluate_answer_quality(self, answer: str) -> float:
        """
        Avalia qualidade da resposta (0-1)
        Critérios simples: comprimento, estrutura, completude
        """
        if not answer:
            return 0.0
        
        score = 0.0
        
        # Comprimento adequado (50-500 palavras = melhor)
        words = len(answer.split())
        if 50 <= words <= 500:
            score += 0.3
        elif words > 20:
            score += 0.15
        
        # Tem pontuação (indica estrutura)
        if any(p in answer for p in ['.', '!', '?']):
            score += 0.2
        
        # Tem formatação (indica organização)
        if any(f in answer for f in ['\n', '•', '-', '1.', '2.']):
            score += 0.2
        
        # Não é muito curto
        if len(answer) > 100:
            score += 0.15
        
        # Tem palavras-chave relevantes
        keywords = ['retail', 'media', 'anúncio', 'campanha', 'performance', 'produto']
        if any(kw in answer.lower() for kw in keywords):
            score += 0.15
        
        return min(score, 1.0)
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """
        Avalia relevância da resposta para a pergunta (0-1)
        Heurística simples baseada em overlap de palavras-chave
        """
        if not answer:
            return 0.0
        
        # Extrair palavras importantes da pergunta
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remover stopwords comuns
        stopwords = {'o', 'a', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'uma', 'para'}
        question_words -= stopwords
        answer_words -= stopwords
        
        # Calcular overlap
        if not question_words:
            return 0.5
        
        overlap = len(question_words & answer_words) / len(question_words)
        return min(overlap, 1.0)
    
    def _estimate_tokens(self, question: str, answer: str) -> int:
        """Estimativa simples de tokens (1 token ≈ 4 caracteres)"""
        total_chars = len(question) + len(answer)
        return total_chars // 4
    
    def _estimate_cost(self, provider: str, model_name: str, tokens: int) -> Optional[float]:
        """Estima custo em USD baseado no provedor e modelo"""
        if provider == "ollama":
            return 0.0  # Local, gratuito
        
        if provider == "huggingface":
            return 0.0  # API gratuita (com limites)
        
        if provider == "openai":
            # Preços aproximados (input + output combinados)
            costs_per_1k = {
                "gpt-4o-mini": 0.00015,
                "gpt-4o": 0.005,
                "gpt-3.5-turbo": 0.0005,
            }
            cost_per_token = costs_per_1k.get(model_name, 0.0001) / 1000
            return tokens * cost_per_token
        
        return None
    
    def _save_results(self):
        """Salva resultados em arquivo JSON e CSV"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # JSON detalhado
        json_path = self.output_dir / f"benchmark_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, ensure_ascii=False)
        
        # CSV para análise
        csv_path = self.output_dir / f"benchmark_{timestamp}.csv"
        df = pd.DataFrame([r.to_dict() for r in self.results])
        df.to_csv(csv_path, index=False, encoding="utf-8")
        
        logger.info(f"Resultados salvos em: {json_path} e {csv_path}")
    
    def generate_report(self) -> pd.DataFrame:
        """
        Gera relatório comparativo dos modelos
        
        Returns:
            DataFrame com métricas agregadas por modelo
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Agrupar por modelo
        report = df.groupby(["provider", "model_name"]).agg({
            "latency_seconds": ["mean", "std", "min", "max"],
            "tokens_used": "mean",
            "cost_usd": "sum",
            "quality_score": "mean",
            "relevance_score": "mean",
            "error": lambda x: x.notna().sum()  # Contagem de erros
        }).round(4)
        
        report.columns = [
            "latency_avg", "latency_std", "latency_min", "latency_max",
            "tokens_avg", "total_cost", "quality_avg", "relevance_avg", "errors"
        ]
        
        return report.reset_index()
    
    def print_report(self):
        """Imprime relatório formatado no console"""
        report = self.generate_report()
        
        if report.empty:
            print("Nenhum resultado disponível")
            return
        
        print("\n" + "="*80)
        print("📊 RELATÓRIO DE BENCHMARK")
        print("="*80 + "\n")
        
        for _, row in report.iterrows():
            print(f"🤖 {row['provider'].upper()} - {row['model_name']}")
            print("-" * 80)
            print(f"  ⚡ Latência:    {row['latency_avg']:.2f}s (±{row['latency_std']:.2f}s)")
            print(f"  📝 Tokens:      {row['tokens_avg']:.0f} (média)")
            print(f"  💰 Custo:       ${row['total_cost']:.4f}")
            print(f"  ⭐ Qualidade:   {row['quality_avg']:.2f}/1.0")
            print(f"  🎯 Relevância:  {row['relevance_avg']:.2f}/1.0")
            if row['errors'] > 0:
                print(f"  ⚠️  Erros:       {row['errors']}")
            print()
        
        print("="*80 + "\n")
        
        # Ranking
        print("🏆 RANKING")
        print("-" * 80)
        
        # Melhor qualidade
        best_quality = report.loc[report['quality_avg'].idxmax()]
        print(f"  🥇 Melhor Qualidade:  {best_quality['provider']} - {best_quality['model_name']} ({best_quality['quality_avg']:.2f})")
        
        # Mais rápido
        best_speed = report.loc[report['latency_avg'].idxmin()]
        print(f"  ⚡ Mais Rápido:       {best_speed['provider']} - {best_speed['model_name']} ({best_speed['latency_avg']:.2f}s)")
        
        # Melhor custo-benefício (qualidade / (custo + latência))
        report['cost_benefit'] = report['quality_avg'] / (report['total_cost'] + report['latency_avg'] + 0.01)
        best_value = report.loc[report['cost_benefit'].idxmax()]
        print(f"  💎 Melhor Custo-Benefício: {best_value['provider']} - {best_value['model_name']}")
        
        print("="*80 + "\n")


# Perguntas padrão para benchmark
DEFAULT_TEST_QUESTIONS = [
    "O que é Retail Media?",
    "Quais são as principais métricas de performance em campanhas de anúncios?",
    "Como funciona o ACOS no Mercado Livre?",
    "Quais estratégias para melhorar CTR em campanhas?",
    "Explique a diferença entre CPC e CPM",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Módulo de Benchmark carregado com sucesso!")
    print(f"Perguntas padrão: {len(DEFAULT_TEST_QUESTIONS)}")


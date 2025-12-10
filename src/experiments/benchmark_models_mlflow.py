#!/usr/bin/env python3
"""
Benchmark de Modelos LLM para MelancIA
Otimizado para hardware limitado (CPU only, 15GB RAM)

Testa múltiplos modelos open source + OpenAI e registra no MLflow
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import json
import mlflow
import mlflow.pyfunc

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

from agent.config import OPENAI_API_KEY, DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL
from agent.retriever import get_retriever
from agent.memory import get_memory
from llm import MultiLLMManager  # 🔄 NOVO PATH
from experiments.benchmark import ModelBenchmark
from evaluation.rag_evaluator import RAGEvaluator

# Perguntas de teste específicas para Retail Media
TEST_QUESTIONS = [
    # Métricas
    "O que é ACOS e como calcular?",
    "Qual a diferença entre ACOS e TACOS?",
    "Como melhorar o ROAS das campanhas?",
    
    # Product Ads
    "Como funcionam os Product Ads no Mercado Livre?",
    "Quais são as melhores práticas para anúncios patrocinados?",
    "Como criar uma campanha de Product Ads eficiente?",
    
    # Otimização
    "Como otimizar palavras-chave para aumentar vendas?",
    "Qual é a melhor estratégia de lances para Product Ads?",
    "Como melhorar a visibilidade dos anúncios?",
    
    # Específico (autopeças)
    "Como funcionam as compatibilidades de autopeças no Mercado Livre?",
    "Por que as compatibilidades impactam as vendas de autopeças?",
    
    # Black Friday
    "Como se preparar para Black Friday no Mercado Livre?",
    
    # Reputação
    "Como melhorar a reputação de vendedor?",
    "O que fazer quando recebo reclamações?",
]

# Modelos recomendados para o hardware específico
RECOMMENDED_MODELS = {
    # Categoria: Ultra Rápidos (< 3s resposta)
    "ultra_fast": [
        ("ollama", "gemma2:2b", "Gemma 2B - Google, ultra rápido"),
        ("ollama", "qwen2.5:3b", "Qwen 2.5 3B - Alibaba, equilibrado"),
        ("ollama", "phi3:mini", "Phi-3 Mini - Microsoft, otimizado"),
    ],
    
    # Categoria: Rápidos e Bons (3-5s resposta)
    "fast_quality": [
        ("ollama", "llama3.2:3b", "Llama 3.2 3B - Meta, ótima qualidade"),
        ("ollama", "phi3:medium", "Phi-3 Medium - Microsoft, melhor contexto"),
        ("ollama", "gemma2:9b", "Gemma 2 9B - Google, alta qualidade"),
    ],
    
    # Categoria: Qualidade Máxima (5-10s resposta)
    "max_quality": [
        ("ollama", "mistral:7b-instruct-q4_K_M", "Mistral 7B Q4 - Mistral AI"),
        ("ollama", "llama3.1:8b-instruct-q4_K_M", "Llama 3.1 8B Q4 - Meta"),
    ],
    
    # Categoria: Cloud (Baseline)
    "cloud": [
        ("openai", "gpt-4o-mini", "GPT-4o Mini - OpenAI (baseline)"),
        ("openai", "gpt-3.5-turbo", "GPT-3.5 Turbo - OpenAI (barato)"),
    ]
}


def check_ollama_models() -> List[str]:
    """Verifica quais modelos estão instalados no Ollama."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        installed = []
        for line in lines:
            if line.strip():
                model_name = line.split()[0]
                installed.append(model_name)
        
        return installed
    except Exception as e:
        print(f"⚠️  Erro ao verificar modelos Ollama: {e}")
        return []


def suggest_models_to_install(hardware_tier: str = "low") -> List[Tuple[str, str, str]]:
    """
    Sugere modelos para instalar baseado no hardware.
    
    Args:
        hardware_tier: "low" (CPU only, <16GB), "medium", "high"
    """
    if hardware_tier == "low":
        # Para hardware limitado
        return [
            ("gemma2:2b", "1.6GB", "Ultra rápido, boa qualidade"),
            ("phi3:mini", "2.3GB", "Microsoft, otimizado para CPU"),
            ("llama3.2:3b", "2GB", "Meta, excelente qualidade"),
            ("qwen2.5:3b", "2GB", "Alibaba, multilíngue"),
            ("mistral:7b-instruct-q4_K_M", "4.1GB", "Qualidade alta (mais lento)"),
        ]
    elif hardware_tier == "medium":
        return [
            ("phi3:medium", "7.9GB", "Melhor versão do Phi-3"),
            ("gemma2:9b", "5.5GB", "Google, alta qualidade"),
            ("llama3.1:8b", "4.7GB", "Meta, versão completa"),
        ]
    else:
        return []


def install_recommended_models(models: List[Tuple[str, str, str]]):
    """Helper para instalar modelos recomendados."""
    import subprocess
    
    print("\n" + "="*70)
    print("📥 MODELOS RECOMENDADOS PARA SEU HARDWARE")
    print("="*70)
    
    for i, (model, size, description) in enumerate(models, 1):
        print(f"\n{i}. {model}")
        print(f"   Tamanho: {size}")
        print(f"   {description}")
    
    print("\n" + "="*70)
    print("Para instalar, execute:")
    print("-"*70)
    for model, _, _ in models:
        print(f"ollama pull {model}")
    print("="*70 + "\n")


def run_comprehensive_benchmark(
    categories: List[str] = ["ultra_fast", "fast_quality", "cloud"],
    max_questions: int = 5,
    save_results: bool = True
):
    """
    Executa benchmark completo de modelos.
    
    Args:
        categories: Lista de categorias para testar
        max_questions: Número máximo de perguntas para testar
        save_results: Se True, salva resultados em arquivo
    """
    print("\n" + "="*70)
    print("🚀 BENCHMARK DE MODELOS LLM - MELANCIA")
    print("="*70)
    
    # Verificar modelos instalados
    print("\n📋 Verificando modelos Ollama instalados...")
    installed_models = check_ollama_models()
    print(f"✅ Encontrados {len(installed_models)} modelos: {', '.join(installed_models)}")
    
    # Setup RAG components
    print("\n🔧 Configurando componentes RAG...")
    retriever = get_retriever(str(VECTOR_DB_DIR), EMBEDDING_MODEL, k=10)
    memory = get_memory(str(DATA_DIR / "output" / "benchmark_history.pkl"))
    evaluator = RAGEvaluator()
    
    # Configurar MLflow
    mlflow.set_experiment("melancia_model_comparison")
    
    # Coletar modelos para testar
    models_to_test = []
    for category in categories:
        if category in RECOMMENDED_MODELS:
            for provider, model_name, description in RECOMMENDED_MODELS[category]:
                # Se for Ollama, verificar se está instalado
                if provider == "ollama":
                    model_key = model_name.split(":")[0]  # Remove tag
                    if not any(model_key in installed for installed in installed_models):
                        print(f"⚠️  {model_name} não instalado - pulando")
                        continue
                
                models_to_test.append((provider, model_name, description))
    
    print(f"\n🎯 Testando {len(models_to_test)} modelos")
    print("-"*70)
    
    results = []
    
    # Testar cada modelo
    for i, (provider, model_name, description) in enumerate(models_to_test, 1):
        print(f"\n{'='*70}")
        print(f"Modelo {i}/{len(models_to_test)}: {provider}::{model_name}")
        print(f"Descrição: {description}")
        print(f"{'='*70}")
        
        # Criar run no MLflow
        with mlflow.start_run(run_name=f"{provider}_{model_name}"):
            # Log parâmetros
            mlflow.log_param("provider", provider)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("description", description)
            mlflow.log_param("num_questions", min(max_questions, len(TEST_QUESTIONS)))
            
            try:
                # Criar LLM
                llm = MultiLLMManager.create_llm(provider, model_name)
                
                # Criar benchmark
                benchmark = ModelBenchmark(retriever, memory)
                benchmark.add_model(provider, model_name, llm)
                
                # Executar teste
                questions = TEST_QUESTIONS[:max_questions]
                benchmark_results = benchmark.run(questions, evaluate_quality=True)
                
                # Extrair métricas
                if benchmark_results:
                    result = benchmark_results[0]  # Primeiro (e único) modelo
                    
                    # Log métricas no MLflow
                    mlflow.log_metric("avg_latency", result.avg_latency)
                    mlflow.log_metric("total_time", result.total_time)
                    mlflow.log_metric("success_rate", result.success_rate)
                    mlflow.log_metric("avg_quality", result.avg_quality)
                    mlflow.log_metric("avg_relevance", result.avg_relevance)
                    mlflow.log_metric("questions_tested", len(questions))
                    
                    # Estimar custo (OpenAI pricing)
                    if provider == "openai":
                        # gpt-4o-mini: $0.150/1M input, $0.600/1M output
                        # Estimativa: ~1000 tokens/query (500 in, 500 out)
                        estimated_cost = (500 * 0.150 / 1_000_000) + (500 * 0.600 / 1_000_000)
                    else:
                        estimated_cost = 0.0
                    
                    mlflow.log_metric("estimated_cost_per_query", estimated_cost)
                    
                    # Log tags
                    mlflow.set_tags({
                        "hardware": "cpu_only",
                        "category": next(cat for cat in categories if any(
                            m[1] == model_name for m in RECOMMENDED_MODELS.get(cat, [])
                        )),
                        "status": "success"
                    })
                    
                    # Salvar resultado
                    results.append({
                        "provider": provider,
                        "model": model_name,
                        "description": description,
                        "avg_latency": result.avg_latency,
                        "avg_quality": result.avg_quality,
                        "avg_relevance": result.avg_relevance,
                        "success_rate": result.success_rate,
                        "cost_per_query": estimated_cost
                    })
                    
                    print(f"\n✅ Resultado:")
                    print(f"   Latência média: {result.avg_latency:.2f}s")
                    print(f"   Qualidade: {result.avg_quality:.2f}/1.0")
                    print(f"   Relevância: {result.avg_relevance:.2f}/1.0")
                    print(f"   Taxa sucesso: {result.success_rate:.1%}")
                    
            except Exception as e:
                print(f"❌ Erro ao testar {provider}::{model_name}: {e}")
                mlflow.log_param("error", str(e))
                mlflow.set_tag("status", "failed")
    
    # Gerar relatório final
    print("\n" + "="*70)
    print("📊 RELATÓRIO FINAL - COMPARAÇÃO DE MODELOS")
    print("="*70)
    
    if results:
        # Ordenar por qualidade
        results_sorted = sorted(results, key=lambda x: x['avg_quality'], reverse=True)
        
        print(f"\n{'Rank':<6}{'Modelo':<35}{'Latência':<12}{'Qualidade':<12}{'Custo':<10}")
        print("-"*75)
        
        for i, r in enumerate(results_sorted, 1):
            model_str = f"{r['provider']}::{r['model']}"[:34]
            latency_str = f"{r['avg_latency']:.2f}s"
            quality_str = f"{r['avg_quality']:.3f}"
            cost_str = f"${r['cost_per_query']:.5f}" if r['cost_per_query'] > 0 else "Grátis"
            
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{emoji:<6}{model_str:<35}{latency_str:<12}{quality_str:<12}{cost_str:<10}")
        
        # Recomendação
        print("\n" + "="*70)
        print("💡 RECOMENDAÇÕES")
        print("="*70)
        
        best_quality = results_sorted[0]
        
        # Mais rápido: IGNORAR modelos que falharam (latência = 0)
        successful_results = [r for r in results if r['avg_latency'] > 0]
        fastest = min(successful_results, key=lambda x: x['avg_latency']) if successful_results else None
        
        # Melhor gratuito
        best_free = min([r for r in results if r['cost_per_query'] == 0 and r['avg_latency'] > 0], 
                       key=lambda x: -x['avg_quality'], default=None)
        
        print(f"\n🏆 Melhor qualidade: {best_quality['provider']}::{best_quality['model']}")
        print(f"   Qualidade: {best_quality['avg_quality']:.3f} | Latência: {best_quality['avg_latency']:.2f}s")
        cost_str = f"${best_quality['cost_per_query']:.5f}" if best_quality['cost_per_query'] > 0 else "Grátis"
        print(f"   Custo: {cost_str}/query")
        
        if fastest:
            print(f"\n⚡ Mais rápido: {fastest['provider']}::{fastest['model']}")
            print(f"   Latência: {fastest['avg_latency']:.2f}s | Qualidade: {fastest['avg_quality']:.3f}")
            cost_str = f"${fastest['cost_per_query']:.5f}" if fastest['cost_per_query'] > 0 else "Grátis"
            print(f"   Custo: {cost_str}/query")
        
        if best_free:
            print(f"\n💰 Melhor gratuito (open source): {best_free['provider']}::{best_free['model']}")
            print(f"   Qualidade: {best_free['avg_quality']:.3f} | Latência: {best_free['avg_latency']:.2f}s")
        
        # Salvar resultados
        if save_results:
            output_file = Path("data/experiments/model_comparison_results.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "models_tested": len(results),
                    "results": results_sorted,
                    "recommendations": {
                        "best_quality": best_quality,
                        "fastest": fastest,
                        "best_free": best_free
                    }
                }, f, indent=2)
            
            print(f"\n💾 Resultados salvos em: {output_file}")
    
    print("\n" + "="*70)
    print("✅ Benchmark concluído!")
    print(f"📊 Ver resultados no MLflow: mlflow ui --port 5000")
    print("="*70 + "\n")
    
    return results


def quick_test(model_provider: str = "ollama", model_name: str = "phi3:mini"):
    """Teste rápido de um modelo específico."""
    print(f"\n🧪 Teste rápido: {model_provider}::{model_name}")
    print("-"*70)
    
    try:
        llm = MultiLLMManager.create_llm(model_provider, model_name)
        
        test_question = "O que é ACOS no contexto de Product Ads?"
        print(f"Pergunta: {test_question}")
        
        start = time.time()
        response = llm.invoke(test_question)
        latency = time.time() - start
        
        print(f"\nResposta ({latency:.2f}s):")
        print("-"*70)
        print(response)
        print("-"*70)
        print(f"\n✅ Modelo funcional! Latência: {latency:.2f}s")
        
    except Exception as e:
        print(f"❌ Erro: {e}")


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark de Modelos LLM para MelancIA')
    parser.add_argument(
        '--mode',
        choices=['quick', 'fast', 'full', 'test'],
        default='fast',
        help='Modo de execução'
    )
    parser.add_argument(
        '--install-guide',
        action='store_true',
        help='Mostrar guia de instalação de modelos'
    )
    parser.add_argument(
        '--test-model',
        help='Testar modelo específico (formato: provider::model, ex: ollama::phi3:mini)'
    )
    
    args = parser.parse_args()
    
    # Guia de instalação
    if args.install_guide:
        models = suggest_models_to_install("low")
        install_recommended_models(models)
        return
    
    # Teste de modelo específico
    if args.test_model:
        provider, model = args.test_model.split("::", 1)
        quick_test(provider, model)
        return
    
    # Benchmark
    if args.mode == 'quick':
        # Apenas modelos ultra rápidos
        print("🚀 Modo QUICK: Testando apenas modelos ultra rápidos")
        run_comprehensive_benchmark(
            categories=["ultra_fast", "cloud"],
            max_questions=3
        )
    
    elif args.mode == 'fast':
        # Modelos rápidos + cloud
        print("⚡ Modo FAST: Testando modelos rápidos e cloud")
        run_comprehensive_benchmark(
            categories=["ultra_fast", "fast_quality", "cloud"],
            max_questions=5
        )
    
    elif args.mode == 'full':
        # Todos os modelos
        print("🎯 Modo FULL: Testando todos os modelos disponíveis")
        run_comprehensive_benchmark(
            categories=["ultra_fast", "fast_quality", "max_quality", "cloud"],
            max_questions=len(TEST_QUESTIONS)
        )
    
    elif args.mode == 'test':
        # Apenas teste de conexão
        print("🧪 Modo TEST: Verificando setup")
        installed = check_ollama_models()
        print(f"\n✅ {len(installed)} modelos Ollama instalados")
        if installed:
            print(f"Testando primeiro modelo: {installed[0]}")
            quick_test("ollama", installed[0])


if __name__ == "__main__":
    main()


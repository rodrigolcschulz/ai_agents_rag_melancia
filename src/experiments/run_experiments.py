"""
Script principal para executar experimentos com diferentes LLMs
Compara OpenAI, Ollama e HuggingFace em tarefas de RAG
"""
import sys
import logging
from pathlib import Path
import argparse

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiments.multi_llm import MultiLLMManager, LLMFactory
from src.experiments.benchmark import ModelBenchmark, DEFAULT_TEST_QUESTIONS
from src.mlops.tracking import ExperimentTracker
from src.agent.retriever import carregar_markdowns, get_retriever, indexar_novos_markdowns
from src.agent.memory import get_memory
from src.agent import config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_rag_components():
    """
    Configura componentes do RAG (retriever e memory)
    
    Returns:
        Tupla (retriever, memory)
    """
    logger.info("Configurando componentes RAG...")
    
    # Carregar e indexar documentos
    docs = carregar_markdowns(config.INPUT_MARKDOWN)
    indexar_novos_markdowns(docs, str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL)
    
    # Criar retriever
    retriever = get_retriever(str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL)
    
    # Criar memória
    memory = get_memory(config.HISTORY_FILE)
    
    logger.info(f"✓ RAG configurado: {len(docs)} documentos indexados")
    
    return retriever, memory


def run_quick_test():
    """
    Executa teste rápido com modelos disponíveis
    """
    print("\n" + "="*80)
    print("🚀 TESTE RÁPIDO - Experimentação de LLMs")
    print("="*80 + "\n")
    
    # Setup
    retriever, memory = setup_rag_components()
    benchmark = ModelBenchmark(retriever, memory)
    tracker = ExperimentTracker("melancia-quick-test")
    
    # Perguntas de teste (resumidas)
    test_questions = DEFAULT_TEST_QUESTIONS[:3]
    
    print("📝 Perguntas de teste:")
    for i, q in enumerate(test_questions, 1):
        print(f"   {i}. {q}")
    print()
    
    # Testar modelos disponíveis
    models_to_test = []
    
    # 1. OpenAI (sempre disponível)
    try:
        print("🔍 Testando OpenAI...")
        llm_openai = MultiLLMManager.create_llm("openai", "gpt-4o-mini")
        benchmark.add_model("openai", "gpt-4o-mini", llm_openai)
        models_to_test.append(("openai", "gpt-4o-mini"))
        print("   ✓ OpenAI configurado")
    except Exception as e:
        print(f"   ✗ OpenAI não disponível: {e}")
    
    # 2. Ollama (se estiver rodando)
    try:
        print("🔍 Testando Ollama...")
        llm_ollama = MultiLLMManager.create_llm("ollama", "llama3.1:8b")
        # Teste rápido
        llm_ollama.invoke("test")
        benchmark.add_model("ollama", "llama3.1:8b", llm_ollama)
        models_to_test.append(("ollama", "llama3.1:8b"))
        print("   ✓ Ollama configurado")
    except Exception as e:
        print(f"   ✗ Ollama não disponível: {e}")
        print("   💡 Instale com: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   💡 E rode: ollama pull llama3.1:8b")
    
    if not models_to_test:
        print("\n⚠️  Nenhum modelo disponível para teste!")
        return
    
    print(f"\n{'='*80}")
    print(f"🎯 Testando {len(models_to_test)} modelo(s) em {len(test_questions)} perguntas")
    print(f"{'='*80}\n")
    
    # Executar benchmark
    with tracker.start_run("quick-test"):
        # Log parâmetros
        tracker.log_params({
            "num_models": len(models_to_test),
            "num_questions": len(test_questions),
            "models": ", ".join([f"{p}::{m}" for p, m in models_to_test])
        })
        
        # Rodar benchmark
        results = benchmark.run(test_questions, evaluate_quality=True, verbose=True)
        
        # Log resultados
        tracker.log_benchmark_results([r.to_dict() for r in results])
        
        # Gerar relatório
        benchmark.print_report()
        
        # Log métricas agregadas
        report = benchmark.generate_report()
        for _, row in report.iterrows():
            tracker.log_metrics({
                f"{row['provider']}_{row['model_name']}_quality": row['quality_avg'],
                f"{row['provider']}_{row['model_name']}_latency": row['latency_avg'],
                f"{row['provider']}_{row['model_name']}_cost": row['total_cost'],
            })
    
    print("\n✅ Teste concluído!")
    print(f"📊 Resultados salvos em: data/experiments/")
    print(f"🔬 MLflow tracking em: mlruns/")
    print("\n💡 Para visualizar no MLflow UI:")
    print("   mlflow ui --port 5000")
    print("   Abra: http://localhost:5000")


def run_full_benchmark():
    """
    Executa benchmark completo com todos os modelos
    """
    print("\n" + "="*80)
    print("🚀 BENCHMARK COMPLETO - Comparação de LLMs")
    print("="*80 + "\n")
    
    # Setup
    retriever, memory = setup_rag_components()
    benchmark = ModelBenchmark(retriever, memory)
    tracker = ExperimentTracker("melancia-full-benchmark")
    
    # Usar todas as perguntas padrão
    test_questions = DEFAULT_TEST_QUESTIONS
    
    print(f"📝 {len(test_questions)} perguntas de teste preparadas\n")
    
    # Adicionar todos os modelos disponíveis
    models_config = [
        # OpenAI
        ("openai", "gpt-4o-mini"),
        ("openai", "gpt-3.5-turbo"),
        
        # Ollama (local)
        ("ollama", "llama3.1:8b"),
        ("ollama", "mistral:7b"),
        ("ollama", "phi3:mini"),
    ]
    
    models_added = []
    for provider, model_name in models_config:
        try:
            print(f"🔍 Configurando {provider} - {model_name}...")
            llm = MultiLLMManager.create_llm(provider, model_name)
            
            # Teste rápido
            if provider == "ollama":
                llm.invoke("test")
            
            benchmark.add_model(provider, model_name, llm)
            models_added.append((provider, model_name))
            print(f"   ✓ {model_name} pronto")
            
        except Exception as e:
            print(f"   ✗ {model_name} não disponível: {e}")
    
    if not models_added:
        print("\n⚠️  Nenhum modelo disponível!")
        return
    
    print(f"\n{'='*80}")
    print(f"🎯 Benchmark: {len(models_added)} modelos x {len(test_questions)} perguntas")
    print(f"{'='*80}\n")
    
    # Executar benchmark
    with tracker.start_run("full-benchmark"):
        tracker.log_params({
            "num_models": len(models_added),
            "num_questions": len(test_questions),
            "models": ", ".join([f"{p}::{m}" for p, m in models_added])
        })
        
        results = benchmark.run(test_questions, evaluate_quality=True, verbose=True)
        tracker.log_benchmark_results([r.to_dict() for r in results])
        
        benchmark.print_report()
        
        # Log métricas
        report = benchmark.generate_report()
        for _, row in report.iterrows():
            tracker.log_metrics({
                f"{row['provider']}_{row['model_name']}_quality": row['quality_avg'],
                f"{row['provider']}_{row['model_name']}_latency": row['latency_avg'],
                f"{row['provider']}_{row['model_name']}_cost": row['total_cost'],
            })
    
    print("\n✅ Benchmark completo concluído!")
    print(f"📊 Resultados em: data/experiments/")
    print(f"🔬 MLflow tracking em: mlruns/")


def run_custom_experiment(models: list, questions: list):
    """
    Executa experimento customizado
    
    Args:
        models: Lista de tuplas (provider, model_name)
        questions: Lista de perguntas
    """
    retriever, memory = setup_rag_components()
    benchmark = ModelBenchmark(retriever, memory)
    tracker = ExperimentTracker("melancia-custom")
    
    # Adicionar modelos
    for provider, model_name in models:
        try:
            llm = MultiLLMManager.create_llm(provider, model_name)
            benchmark.add_model(provider, model_name, llm)
        except Exception as e:
            logger.error(f"Erro ao adicionar {provider} {model_name}: {e}")
    
    # Executar
    with tracker.start_run("custom-experiment"):
        results = benchmark.run(questions, evaluate_quality=True)
        tracker.log_benchmark_results([r.to_dict() for r in results])
        benchmark.print_report()


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Experimentação com diferentes LLMs para RAG"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "ui"],
        default="quick",
        help="Modo de execução (quick=teste rápido, full=benchmark completo, ui=abrir MLflow UI)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "quick":
            run_quick_test()
        elif args.mode == "full":
            run_full_benchmark()
        elif args.mode == "ui":
            print("🚀 Abrindo MLflow UI...")
            ExperimentTracker.launch_ui(port=5000)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro durante execução: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


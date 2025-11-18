#!/usr/bin/env python3
"""
Script de teste rápido dos LLMs open source
Testa conectividade e performance básica
"""
import sys
import time
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.multi_llm import MultiLLMManager

# Cores para terminal
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_header(text):
    """Imprime cabeçalho formatado"""
    print(f"\n{'='*60}")
    print(f"{BLUE}{text}{NC}")
    print(f"{'='*60}\n")

def test_model(provider, model_name, question):
    """
    Testa um modelo específico
    
    Args:
        provider: Nome do provedor (openai, ollama, huggingface)
        model_name: Nome do modelo
        question: Pergunta de teste
        
    Returns:
        Dict com resultados ou None se falhar
    """
    print(f"{YELLOW}🤖 Testando: {provider.upper()} - {model_name}{NC}")
    
    try:
        # Criar LLM
        print(f"   ⚙️  Inicializando...")
        llm = MultiLLMManager.create_llm(
            provider=provider,
            model_name=model_name,
            temperature=0.5,
            max_tokens=500
        )
        
        # Fazer pergunta
        print(f"   💬 Gerando resposta...")
        start_time = time.time()
        response = llm.invoke(question)
        latency = time.time() - start_time
        
        # Extrair texto da resposta
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        # Estatísticas
        words = len(answer.split())
        chars = len(answer)
        
        print(f"{GREEN}   ✓ Sucesso!{NC}")
        print(f"   ⏱️  Latência: {latency:.2f}s")
        print(f"   📝 Tamanho: {words} palavras, {chars} caracteres")
        print(f"\n{BLUE}   Resposta:{NC}")
        
        # Mostrar primeiras linhas
        lines = answer.split('\n')[:5]
        for line in lines:
            if line.strip():
                print(f"   {line[:100]}")
        
        if len(answer) > 500:
            print(f"   ... (resposta truncada)")
        
        print()
        
        return {
            'provider': provider,
            'model': model_name,
            'success': True,
            'latency': latency,
            'words': words,
            'answer': answer
        }
        
    except Exception as e:
        print(f"{RED}   ✗ Erro: {e}{NC}")
        print()
        return {
            'provider': provider,
            'model': model_name,
            'success': False,
            'error': str(e)
        }

def main():
    """Função principal"""
    print_header("🍉 MelâncIA - Teste Rápido de LLMs")
    
    # Pergunta de teste (contexto Retail Media)
    question = "O que é Retail Media e quais são suas principais métricas?"
    
    print(f"{BLUE}Pergunta de teste:{NC} {question}\n")
    
    # Modelos para testar (ordem de prioridade)
    models_to_test = [
        # Ollama (local - prioridade)
        ("ollama", "phi3:mini"),
        ("ollama", "llama3.2:3b"),
        ("ollama", "gemma2:2b"),
        
        # OpenAI (se configurado)
        ("openai", "gpt-4o-mini"),
    ]
    
    results = []
    
    for provider, model_name in models_to_test:
        result = test_model(provider, model_name, question)
        if result:
            results.append(result)
    
    # Resumo
    print_header("📊 Resumo dos Testes")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    if successful:
        print(f"{GREEN}✅ Modelos funcionando: {len(successful)}{NC}\n")
        
        # Ordenar por latência
        successful.sort(key=lambda x: x['latency'])
        
        print(f"{'Modelo':<25} {'Latência':<12} {'Palavras':<10} {'Status'}")
        print("-" * 60)
        
        for r in successful:
            model_full = f"{r['provider']}::{r['model']}"
            print(f"{model_full:<25} {r['latency']:.2f}s{' ':<7} {r['words']:<10} ✓")
        
        print()
        
        # Melhor modelo
        best = successful[0]
        print(f"{YELLOW}🏆 Melhor performance:{NC} {best['provider']}::{best['model']} ({best['latency']:.2f}s)")
    
    if failed:
        print(f"\n{RED}❌ Modelos com erro: {len(failed)}{NC}\n")
        for r in failed:
            print(f"  • {r['provider']}::{r['model']}: {r.get('error', 'Unknown error')}")
    
    # Recomendações
    print(f"\n{YELLOW}💡 Próximos passos:{NC}")
    
    if not successful:
        print(f"  {RED}1. Instale o Ollama:{NC}")
        print(f"     ./scripts/setup_ollama.sh")
        print(f"\n  {RED}2. Configure OpenAI API key (opcional):{NC}")
        print(f"     echo 'OPENAI_API_KEY=sk-...' >> .env")
    else:
        print(f"  {GREEN}1. Executar benchmark completo:{NC}")
        print(f"     python src/experiments/run_experiments.py --mode quick")
        print(f"\n  {GREEN}2. Usar no código:{NC}")
        print(f"     from src.experiments.multi_llm import MultiLLMManager")
        print(f"     llm = MultiLLMManager.create_llm('ollama', 'phi3:mini')")
        print(f"\n  {GREEN}3. Experimentar no Jupyter:{NC}")
        print(f"     jupyter lab")
        print(f"     # Abrir: notebooks/experimentacao_llms.ipynb")
    
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}⚠️  Interrompido pelo usuário{NC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{RED}❌ Erro fatal: {e}{NC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


#!/usr/bin/env python3
"""
Teste rápido do sistema RAG com Phi-3 Mini
Mostra a diferença entre usar o modelo puro vs com contexto dos documentos
"""
import sys
import time
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.multi_llm import MultiLLMManager
from src.agent.retriever import carregar_markdowns, get_retriever, indexar_novos_markdowns
from src.agent import config

# Cores
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
RED = '\033[0;31m'
NC = '\033[0m'

def print_header(text):
    print(f"\n{'='*70}")
    print(f"{BLUE}{text}{NC}")
    print(f"{'='*70}\n")

def test_without_rag(llm, question):
    """Testa modelo SEM contexto RAG"""
    print(f"{YELLOW}📝 Teste 1: SEM RAG (modelo puro){NC}\n")
    
    start = time.time()
    response = llm.invoke(question)
    latency = time.time() - start
    
    answer = response.content if hasattr(response, 'content') else str(response)
    
    print(f"{BLUE}Resposta:{NC}")
    print(f"{answer}\n")
    print(f"⏱️  Latência: {latency:.2f}s")
    print(f"📊 Tamanho: {len(answer.split())} palavras")
    
    return answer, latency

def test_with_rag(llm, retriever, question):
    """Testa modelo COM contexto RAG"""
    print(f"\n{YELLOW}📚 Teste 2: COM RAG (modelo + documentos){NC}\n")
    
    # Buscar contexto relevante
    print(f"🔍 Buscando contexto nos documentos...")
    docs = retriever.get_relevant_documents(question)
    
    print(f"{GREEN}✓ Encontrados {len(docs)} documentos relevantes{NC}\n")
    
    # Mostrar fontes
    print(f"{BLUE}📄 Fontes encontradas:{NC}")
    for i, doc in enumerate(docs[:3], 1):
        source = doc.metadata.get('source', 'Desconhecido')
        filename = Path(source).name if source != 'Desconhecido' else source
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"  {i}. {filename}")
        print(f"     Preview: {preview}...")
    
    # Criar prompt com contexto
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    prompt_with_context = f"""Com base nos seguintes documentos sobre Retail Media, responda a pergunta:

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""
    
    print(f"\n💬 Gerando resposta com contexto...")
    start = time.time()
    response = llm.invoke(prompt_with_context)
    latency = time.time() - start
    
    answer = response.content if hasattr(response, 'content') else str(response)
    
    print(f"\n{BLUE}Resposta:{NC}")
    print(f"{answer}\n")
    print(f"⏱️  Latência: {latency:.2f}s")
    print(f"📊 Tamanho: {len(answer.split())} palavras")
    
    return answer, latency

def main():
    print_header("🍉 MelâncIA - Teste RAG com Phi-3 Mini")
    
    # Pergunta de teste
    question = "O que é Retail Media?"
    print(f"{BLUE}❓ Pergunta:{NC} {question}\n")
    
    # Inicializar LLM
    print(f"{YELLOW}🤖 Inicializando Phi-3 Mini...{NC}")
    try:
        llm = MultiLLMManager.create_llm(
            provider="ollama",
            model_name="phi3:mini",
            temperature=0.5,
            max_tokens=500
        )
        print(f"{GREEN}✓ Phi-3 Mini carregado!{NC}\n")
    except Exception as e:
        print(f"{RED}❌ Erro ao carregar modelo: {e}{NC}")
        print(f"\n{YELLOW}💡 Certifique-se de que:{NC}")
        print(f"  1. Ollama está instalado")
        print(f"  2. Execute: ollama pull phi3:mini")
        return
    
    # ==========================================
    # TESTE 1: SEM RAG
    # ==========================================
    try:
        answer_no_rag, latency_no_rag = test_without_rag(llm, question)
    except Exception as e:
        print(f"{RED}❌ Erro no teste sem RAG: {e}{NC}")
        return
    
    # ==========================================
    # TESTE 2: COM RAG
    # ==========================================
    try:
        print(f"\n{YELLOW}📚 Preparando sistema RAG...{NC}")
        
        # Carregar e indexar documentos
        print(f"📖 Carregando documentos markdown...")
        docs = carregar_markdowns(config.INPUT_MARKDOWN)
        print(f"{GREEN}✓ {len(docs)} documentos carregados{NC}")
        
        print(f"🔍 Indexando no banco vetorial...")
        indexar_novos_markdowns(docs, str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL)
        print(f"{GREEN}✓ Documentos indexados{NC}")
        
        # Criar retriever
        print(f"⚙️  Criando retriever...")
        retriever = get_retriever(str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL)
        print(f"{GREEN}✓ RAG pronto!{NC}")
        
        # Testar com RAG
        answer_with_rag, latency_with_rag = test_with_rag(llm, retriever, question)
        
    except Exception as e:
        print(f"{RED}❌ Erro no teste com RAG: {e}{NC}")
        import traceback
        traceback.print_exc()
        return
    
    # ==========================================
    # COMPARAÇÃO
    # ==========================================
    print_header("📊 Comparação: SEM RAG vs COM RAG")
    
    print(f"{YELLOW}Latência:{NC}")
    print(f"  • Sem RAG: {latency_no_rag:.2f}s")
    print(f"  • Com RAG: {latency_with_rag:.2f}s")
    
    print(f"\n{YELLOW}Tamanho da resposta:{NC}")
    print(f"  • Sem RAG: {len(answer_no_rag.split())} palavras")
    print(f"  • Com RAG: {len(answer_with_rag.split())} palavras")
    
    print(f"\n{GREEN}✅ Conclusão:{NC}")
    print(f"  Com RAG, o modelo responde baseado nos seus {len(docs)} documentos!")
    print(f"  As respostas são mais precisas e contextualizadas.")
    
    print(f"\n{YELLOW}💡 Próximos passos:{NC}")
    print(f"  1. Adicione mais documentos markdown em: data/input/")
    print(f"  2. Execute o agente completo: python -m src.agent.main")
    print(f"  3. Use a interface web: python -m src.agent.web_interface")
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


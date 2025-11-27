#!/usr/bin/env python3
"""
Teste end-to-end do agente com as perguntas problemáticas do usuário
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from src.agent import config
from src.agent.prompt import get_prompt_template
from src.agent.memory import get_memory
from src.agent.retriever import carregar_markdowns, get_retriever, indexar_novos_markdowns


def test_agent_queries():
    """Testa o agente com queries problemáticas"""
    
    print("="*80)
    print("🤖 TESTE END-TO-END DO AGENTE")
    print("="*80 + "\n")
    
    # 1. Setup RAG
    print("⚙️  Configurando RAG...")
    docs = carregar_markdowns(config.INPUT_MARKDOWN)
    indexar_novos_markdowns(docs, str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL)
    retriever = get_retriever(str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL, k=4)
    print(f"   ✅ {len(docs)} documentos indexados\n")
    
    # 2. Setup LLM e Chain
    print("🧠 Configurando LLM...")
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
        max_tokens=1000
    )
    
    memory = get_memory(":memory:")  # Usar memória em RAM para teste
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": get_prompt_template(),
            "document_separator": "\n\n---\n\n"
        },
        return_source_documents=True,
        verbose=True  # Para ver o que está acontecendo
    )
    print("   ✅ Chain configurada\n")
    
    # 3. Testar queries problemáticas
    test_queries = [
        "como proteger a conta no mercado livre?",
        "vc não saberia responder sobre manter a conta no mercado livre segura? protegida?"
    ]
    
    print("="*80)
    print("💬 TESTANDO QUERIES")
    print("="*80 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print("="*80)
        
        # Invocar chain
        resultado = qa_chain.invoke({"question": query})
        
        # Extrair resposta e documentos
        resposta = resultado.get('answer', 'N/A')
        docs_retrieved = resultado.get('source_documents', [])
        
        print(f"\n📄 Documentos recuperados: {len(docs_retrieved)}")
        for j, doc in enumerate(docs_retrieved, 1):
            source = Path(doc.metadata.get('source', 'N/A')).name
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   {j}. {source}")
            print(f"      {content_preview}...")
        
        print(f"\n💬 Resposta do agente:")
        print(f"   {resposta}\n")
        
        # Análise
        if "não encontrei" in resposta.lower() or "não tenho" in resposta.lower():
            print("   ⚠️  PROBLEMA: Agente diz que não encontrou a informação!")
        else:
            print("   ✅ OK: Agente respondeu com informações")
    
    print("\n" + "="*80)
    print("✅ Teste concluído!")
    print("="*80)


if __name__ == "__main__":
    test_agent_queries()

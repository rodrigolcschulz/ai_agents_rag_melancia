#!/usr/bin/env python3
"""
Teste rápido com as queries que não estavam funcionando
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from src.agent import config
from src.agent.prompt import get_prompt_template
from src.agent.memory import get_memory
from src.agent.retriever import carregar_markdowns, get_retriever, indexar_novos_markdowns

def main():
    print("\n" + "="*80)
    print("🍉 MelancIA - Teste Rápido")
    print("="*80 + "\n")
    
    # Setup
    print("⚙️  Configurando sistema...")
    docs = carregar_markdowns(config.INPUT_MARKDOWN)
    indexar_novos_markdowns(docs, str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL)
    retriever = get_retriever(str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL, k=4)
    
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
        max_tokens=1000
    )
    
    memory = get_memory(":memory:")
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": get_prompt_template(),
            "document_separator": "\n\n---\n\n"
        },
        return_source_documents=True,
        verbose=True  # Ver o que está acontecendo
    )
    
    print("✅ Sistema pronto!\n")
    print("="*80)
    
    # Testar as queries problemáticas
    queries = [
        "como proteger a conta no mercado livre?",
        "vc não saberia responder sobre manter a conta no mercado livre segura? protegida?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Pergunta {i}: {query}")
        print("="*80)
        
        resultado = qa_chain.invoke({"question": query})
        resposta = resultado.get('answer', 'N/A')
        
        print(f"\n🍉 MelancIA 🤖: {resposta}\n")
    
    print("="*80)
    print("✅ Teste concluído!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

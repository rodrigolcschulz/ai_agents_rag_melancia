#!/usr/bin/env python3
"""
Script de debug para investigar por que o retriever não está encontrando informações
sobre proteção de conta no Mercado Livre
"""
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.retriever import carregar_markdowns, indexar_novos_markdowns, get_retriever
from src.agent import config
import glob


def main():
    print("="*80)
    print("🔍 DEBUG - Sistema de Retrieval")
    print("="*80)
    
    # 1. Verificar arquivos markdown
    print("\n1️⃣ ARQUIVOS MARKDOWN")
    print(f"   Padrão de busca: {config.INPUT_MARKDOWN}")
    arquivos = glob.glob(config.INPUT_MARKDOWN, recursive=True)
    print(f"   Total encontrado: {len(arquivos)}")
    
    for i, arquivo in enumerate(arquivos, 1):
        tamanho = Path(arquivo).stat().st_size
        print(f"   {i}. {Path(arquivo).name} ({tamanho:,} bytes)")
    
    # 2. Carregar documentos
    print("\n2️⃣ CARREGANDO DOCUMENTOS")
    docs = carregar_markdowns(config.INPUT_MARKDOWN)
    print(f"   Total de documentos: {len(docs)}")
    
    for i, doc in enumerate(docs, 1):
        texto_preview = doc['text'][:200].replace('\n', ' ')
        print(f"   Doc {i}: {len(doc['text'])} chars")
        print(f"         Preview: '{texto_preview}...'")
    
    # 3. Indexar documentos
    print("\n3️⃣ INDEXANDO DOCUMENTOS")
    print(f"   Vector DB dir: {config.VECTOR_DB_DIR}")
    print(f"   Embedding model: {config.EMBEDDING_MODEL}")
    
    db = indexar_novos_markdowns(docs, str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL)
    print(f"   ✅ Indexação concluída!")
    
    # 4. Testar retriever com queries específicas
    print("\n4️⃣ TESTANDO RETRIEVER")
    retriever = get_retriever(str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL, k=4)
    
    queries = [
        "como proteger a conta no mercado livre?",
        "proteger conta mercado livre",
        "segurança conta vendedor",
        "manter conta segura protegida",
        "golpes phishing mercado livre",
        "autenticação dois fatores 2FA"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        docs_retrieved = retriever.invoke(query)
        print(f"   Documentos recuperados: {len(docs_retrieved)}")
        
        for i, doc in enumerate(docs_retrieved, 1):
            source = doc.metadata.get('source', 'N/A')
            content_preview = doc.page_content[:150].replace('\n', ' ')
            print(f"      Doc {i}: {Path(source).name}")
            print(f"             {content_preview}...")
    
    # 5. Testar busca por similaridade diretamente no DB com scores
    print("\n5️⃣ TESTE DE SIMILARIDADE COM SCORES")
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(config.VECTOR_DB_DIR), embedding_function=embeddings)
    
    query = "como proteger conta no mercado livre"
    results = db.similarity_search_with_score(query, k=5)
    
    print(f"   Query: '{query}'")
    print(f"   Resultados com score:")
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'N/A')
        content_preview = doc.page_content[:100].replace('\n', ' ')
        print(f"      {i}. Score: {score:.4f}")
        print(f"         Source: {Path(source).name}")
        print(f"         Content: {content_preview}...")
    
    print("\n" + "="*80)
    print("✅ Debug concluído!")
    print("="*80)


if __name__ == "__main__":
    main()

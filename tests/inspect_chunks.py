#!/usr/bin/env python3
"""
Debug dos chunks - verificar se o conteúdo está sendo armazenado corretamente
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import config
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def inspect_chunks():
    """Inspeciona os chunks armazenados no vector DB"""
    
    print("="*80)
    print("🔍 INSPEÇÃO DE CHUNKS NO VECTOR DB")
    print("="*80 + "\n")
    
    # Carregar DB
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(config.VECTOR_DB_DIR), embedding_function=embeddings)
    
    # Buscar chunks sobre proteção de conta
    query = "como proteger conta no mercado livre"
    results = db.similarity_search_with_score(query, k=5)
    
    print(f"Query: '{query}'\n")
    print(f"Chunks recuperados: {len(results)}\n")
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'N/A')
        content = doc.page_content
        
        print(f"{'='*80}")
        print(f"CHUNK {i}")
        print(f"{'='*80}")
        print(f"Score: {score:.4f}")
        print(f"Source: {Path(source).name}")
        print(f"Content length: {len(content)} chars")
        print(f"\nContent preview (primeiros 500 chars):")
        print("-"*80)
        print(content[:500])
        print("-"*80)
        print(f"\nContent total:")
        print("-"*80)
        print(content)
        print("-"*80 + "\n")


if __name__ == "__main__":
    inspect_chunks()

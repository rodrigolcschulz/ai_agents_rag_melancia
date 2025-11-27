#!/usr/bin/env python3
"""
Testa se o filtro de relevância está bloqueando perguntas legítimas
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.utils import is_relevant

def test_queries():
    """Testa queries sobre proteção de conta"""
    
    queries = [
        "como proteger a conta no mercado livre?",
        "vc não saberia responder sobre manter a conta no mercado livre segura? protegida?",
        "como manter minha conta segura",
        "dicas de segurança para vendedor",
        "golpes no mercado livre",
        "autenticação de dois fatores",
        "phishing mercado livre"
    ]
    
    print("="*80)
    print("🧪 TESTE DO FILTRO DE RELEVÂNCIA")
    print("="*80 + "\n")
    
    for query in queries:
        is_rel = is_relevant(query)
        emoji = "✅" if is_rel else "❌"
        print(f"{emoji} Query: '{query}'")
        print(f"   Resultado: {'RELEVANTE' if is_rel else 'NÃO RELEVANTE'}\n")
    
    print("="*80)

if __name__ == "__main__":
    test_queries()

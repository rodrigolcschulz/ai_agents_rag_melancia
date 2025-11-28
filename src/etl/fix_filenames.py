#!/usr/bin/env python3
"""
Script para corrigir nomes de arquivos truncados
"""
import os
from pathlib import Path

# Mapeamento de nomes truncados para nomes completos
fixes = {
    "Como criar anúncios no Mercado Livr.md": "Como criar anúncios no Mercado Livre passo a passo.md",
    "Aprenda a modificar anúncios com Ex.md": "Aprenda a modificar anúncios com Excel.md",
    "Aprenda a preencher a planilha Exce.md": "Aprenda a preencher a planilha Excel.md",
    "Como ativar o Mercado Envios 1 (ME1.md": "Como ativar o Mercado Envios 1 ME1.md",
    "Como coloco meus anúncios em ME1.md": "Como coloco meus anúncios em Mercado Envios.md",
    "Como funcionam as taxas do Mercado.md": "Como funcionam as taxas do Mercado Livre.md",
    "Como melhorar a visibilidade e o po.md": "Como melhorar a visibilidade e poder de venda.md",
    "Como os envios no Mercado Livre fun.md": "Como os envios no Mercado Livre funcionam.md",
    "Como precificar um produto no Merca.md": "Como precificar um produto no Mercado Livre.md",
    "Como preencher a tabela de contingê.md": "Como preencher a tabela de contingência.md",
    "Como recebo o dinheiro das minhas v.md": "Como recebo o dinheiro das minhas vendas.md",
    "Como usar o Gestor de fotos para an.md": "Como usar o Gestor de fotos para anúncios.md",
    "Conheça como usar o Anunciador em m.md": "Conheça como usar o Anunciador em massa.md",
    "Consiga vendas em menos tempo com o.md": "Consiga vendas em menos tempo com otimização.md",
    "Crie vídeos e aumente suas vendas e.md": "Crie vídeos e aumente suas vendas exponencialmente.md",
    "Ferramentas para solucionar as recl.md": "Ferramentas para solucionar as reclamações.md",
    "Mensagens de pré-venda responda com.md": "Mensagens de pré-venda responda com eficiência.md",
    "Mensagens saiba quando usá-las e ec.md": "Mensagens saiba quando usá-las eficientemente.md",
    "O que é Mercado Envios 1 (ME1) e qu.md": "O que é Mercado Envios 1 ME1 e qualidade.md",
    "O que fazer quando tem uma devoluçã.md": "O que fazer quando tem uma devolução.md",
    "Por que a reputacão do vendedor é i.md": "Por que a reputação do vendedor é importante.md",
    "Quais documentos eu preciso para op.md": "Quais documentos eu preciso para operar.md",
    "Quais são os tipos de anúncios do M.md": "Quais são os tipos de anúncios do Mercado Livre.md",
}

base_dir = Path("data/input/central_vendedores")

if not base_dir.exists():
    print(f"❌ Diretório não encontrado: {base_dir}")
    exit(1)

print("🔄 Corrigindo nomes de arquivos truncados...\n")

renamed = 0
for old_name, new_name in fixes.items():
    old_path = base_dir / old_name
    new_path = base_dir / new_name
    
    if old_path.exists():
        try:
            old_path.rename(new_path)
            print(f"✅ {old_name}")
            print(f"   → {new_name}\n")
            renamed += 1
        except Exception as e:
            print(f"❌ Erro ao renomear {old_name}: {e}\n")
    else:
        print(f"⚠️  Arquivo não encontrado: {old_name}\n")

print(f"\n{'='*60}")
print(f"✅ {renamed} arquivo(s) renomeado(s) com sucesso!")
print(f"{'='*60}")
print("\n🔄 Próximos passos:")
print("1. Rode: python src/etl/populate_vector_db.py")
print("2. Reinicie o Docker (se necessário): docker compose restart melancia-ai")

#!/usr/bin/env python3
"""
Script para popular o banco vetorial com os documentos coletados pelo ETL
"""

import logging
import sys
from pathlib import Path

# Adicionar o diretório src ao path para importar módulos
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from agent.retriever import carregar_markdowns, indexar_novos_markdowns
from agent.config import VECTOR_DB_DIR

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/vector_db.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def populate_vector_database(input_dir="data/input", vector_db_dir=None):
    """
    Popula o banco vetorial com os documentos Markdown coletados
    
    Args:
        input_dir: Diretório com os arquivos Markdown
        vector_db_dir: Diretório do banco vetorial (usa o padrão se None)
    """
    
    if vector_db_dir is None:
        vector_db_dir = VECTOR_DB_DIR
    
    logger.info("🔄 Iniciando população do banco vetorial...")
    
    # Verificar se o diretório de entrada existe
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"❌ Diretório de entrada não encontrado: {input_dir}")
        return False
    
    # Contar arquivos Markdown (busca recursiva em subpastas)
    markdown_files = list(input_path.glob("**/*.md"))
    if not markdown_files:
        logger.error(f"❌ Nenhum arquivo Markdown encontrado em {input_dir}")
        return False
    
    logger.info(f"📁 Encontrados {len(markdown_files)} arquivos Markdown")
    
    try:
        # Carregar documentos (busca recursiva)
        logger.info("📖 Carregando documentos...")
        docs = carregar_markdowns(str(input_path / "**" / "*.md"))
        logger.info(f"✅ {len(docs)} documentos carregados com sucesso")
        
        # Criar diretório do banco vetorial se não existir
        vector_path = Path(vector_db_dir)
        vector_path.mkdir(parents=True, exist_ok=True)
        
        # Indexar documentos no banco vetorial
        logger.info("🔍 Criando embeddings e indexando no banco vetorial...")
        db = indexar_novos_markdowns(docs, str(vector_path))
        
        logger.info("✅ Banco vetorial populado com sucesso!")
        logger.info(f"📊 {len(docs)} documentos indexados em {vector_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao popular banco vetorial: {e}")
        return False


def main():
    """Função principal para execução via linha de comando"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Popula o banco vetorial com documentos Markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python populate_vector_db.py                    # Usa diretórios padrão
  python populate_vector_db.py --input-dir data/input  # Especifica diretório de entrada
        """
    )
    
    parser.add_argument('--input-dir', '-i',
                       default='data/input',
                       help='Diretório com arquivos Markdown')
    parser.add_argument('--vector-db-dir', '-v',
                       help='Diretório do banco vetorial')
    
    args = parser.parse_args()
    
    try:
        success = populate_vector_database(
            input_dir=args.input_dir,
            vector_db_dir=args.vector_db_dir
        )
        
        if success:
            logger.info("🎉 Processo concluído com sucesso!")
            sys.exit(0)
        else:
            logger.error("💥 Processo falhou!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Processo interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

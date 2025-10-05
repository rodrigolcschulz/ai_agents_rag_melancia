#!/usr/bin/env python3
"""
Script principal para executar o pipeline ETL completo
Coleta dados do blog e gera análises
"""

import argparse
import logging
from pathlib import Path
import sys
import time

# Adicionar o diretório pai ao path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from etl_and_scrapping.scraper import BlogScraper
from etl_and_scrapping.analyzer import ContentAnalyzer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_etl_pipeline(
    scrape: bool = True,
    analyze: bool = True,
    max_articles: int = None,
    input_dir: str = "../melanc.ia/Input/Blog",
    output_dir: str = "../melanc.ia/Output"
):
    """
    Executa o pipeline ETL completo
    
    Args:
        scrape: Se deve executar o scraping
        analyze: Se deve executar a análise
        max_articles: Número máximo de artigos para processar
        input_dir: Diretório de entrada para arquivos Markdown
        output_dir: Diretório de saída para relatórios
    """
    
    start_time = time.time()
    logger.info("🚀 Iniciando pipeline ETL...")
    
    results = {
        'scraping': None,
        'analysis': None,
        'total_time': 0
    }
    
    # Etapa 1: Scraping
    if scrape:
        logger.info("📥 Etapa 1: Scraping do blog...")
        try:
            scraper = BlogScraper(output_dir=input_dir)
            scraping_results = scraper.scrape_all(max_articles=max_articles)
            results['scraping'] = scraping_results
            
            logger.info(f"✅ Scraping concluído: {scraping_results['successful']} arquivos criados")
            
        except Exception as e:
            logger.error(f"❌ Erro no scraping: {e}")
            return results
    
    # Etapa 2: Análise
    if analyze:
        logger.info("📊 Etapa 2: Análise de conteúdo...")
        try:
            analyzer = ContentAnalyzer(input_dir=input_dir)
            analyzer.load_files()
            analysis_stats = analyzer.generate_statistics()
            
            # Gerar visualizações e relatórios
            analysis_output_dir = Path(output_dir) / "Analysis"
            analyzer.create_visualizations(str(analysis_output_dir))
            analyzer.generate_report(str(analysis_output_dir))
            analyzer.save_dataframe(str(analysis_output_dir))
            
            results['analysis'] = analysis_stats
            
            logger.info(f"✅ Análise concluída: {analysis_stats['total_files']} arquivos analisados")
            
        except Exception as e:
            logger.error(f"❌ Erro na análise: {e}")
            return results
    
    # Tempo total
    total_time = time.time() - start_time
    results['total_time'] = total_time
    
    # Relatório final
    logger.info("📋 RELATÓRIO FINAL DO PIPELINE ETL")
    logger.info("=" * 50)
    
    if results['scraping']:
        scraping = results['scraping']
        logger.info(f"📥 Scraping:")
        logger.info(f"   - URLs processadas: {scraping['total_urls']}")
        logger.info(f"   - Sucessos: {scraping['successful']}")
        logger.info(f"   - Falhas: {scraping['failed']}")
        logger.info(f"   - Total de palavras: {scraping['total_words']:,}")
    
    if results['analysis']:
        analysis = results['analysis']
        logger.info(f"📊 Análise:")
        logger.info(f"   - Arquivos analisados: {analysis['total_files']}")
        logger.info(f"   - Total de palavras: {analysis['total_words']:,}")
        logger.info(f"   - Média de palavras por arquivo: {analysis['avg_words_per_file']:.1f}")
    
    logger.info(f"⏱️ Tempo total: {total_time:.2f} segundos")
    logger.info("✅ Pipeline ETL concluído com sucesso!")
    
    return results


def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(
        description='Pipeline ETL para coleta e análise de conteúdo do blog Conecta Ads',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python run_etl.py                    # Executa scraping + análise
  python run_etl.py --no-scrape        # Apenas análise
  python run_etl.py --no-analyze       # Apenas scraping
  python run_etl.py --max-articles 50  # Limita a 50 artigos
        """
    )
    
    parser.add_argument('--no-scrape', action='store_true',
                       help='Pular etapa de scraping')
    parser.add_argument('--no-analyze', action='store_true',
                       help='Pular etapa de análise')
    parser.add_argument('--max-articles', '-m', type=int,
                       help='Número máximo de artigos para processar')
    parser.add_argument('--input-dir', '-i',
                       default='../melanc.ia/Input/Blog',
                       help='Diretório de entrada para arquivos Markdown')
    parser.add_argument('--output-dir', '-o',
                       default='../melanc.ia/Output',
                       help='Diretório de saída para relatórios')
    
    args = parser.parse_args()
    
    # Executar pipeline
    try:
        results = run_etl_pipeline(
            scrape=not args.no_scrape,
            analyze=not args.no_analyze,
            max_articles=args.max_articles,
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        
        # Salvar relatório final
        report_file = Path(args.output_dir) / "etl_pipeline_report.json"
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Relatório final salvo em: {report_file}")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal no pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

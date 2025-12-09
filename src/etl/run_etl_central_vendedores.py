#!/usr/bin/env python3
"""
Script de execução para o ETL do Central de Vendedores
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.etl.scraper_central_vendedores import CentralVendedoresScraper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Executa o scraping do Central de Vendedores"""
    
    print("\n" + "="*80)
    print("ETL - CENTRAL DE VENDEDORES DO MERCADO LIVRE")
    print("="*80 + "\n")
    
    # Configurações
    config = {
        'cursos_url': 'https://central.vendedores.mercadolivre.com.br/cursos',
        'output_dir': 'data/input/central_vendedores_new',
        'use_selenium': True,
        'max_courses': 2,  # Apenas 2 cursos para teste inicial
        'extract_details': False  # Não extrair detalhes por enquanto (mais rápido)
    }
    
    print("Configurações:")
    print(f"  - URL base: {config['cursos_url']}")
    print(f"  - Diretório de saída: {config['output_dir']}")
    print(f"  - Usar Selenium: {config['use_selenium']}")
    print(f"  - Max cursos (teste): {config['max_courses']}")
    print(f"  - Extrair detalhes: {config['extract_details']}")
    print()
    
    resposta = input("Deseja continuar? (s/n): ").strip().lower()
    if resposta != 's':
        print("Cancelado pelo usuário.")
        return
    
    # Criar scraper
    scraper = CentralVendedoresScraper(
        cursos_url=config['cursos_url'],
        output_dir=config['output_dir'],
        use_selenium=config['use_selenium']
    )
    
    try:
        # Executar scraping
        stats = scraper.scrape_all(
            max_courses=config['max_courses'],
            extract_details=config['extract_details']
        )
        
        # Mostrar resultados
        print("\n" + "="*80)
        print("RESULTADOS")
        print("="*80)
        print(f"✓ Cursos processados: {stats.get('processed_courses', 0)}")
        print(f"✓ Cursos com sucesso: {stats.get('successful_courses', 0)}")
        print(f"✗ Cursos com falha: {stats.get('failed_courses', 0)}")
        print(f"📚 Total de módulos: {stats.get('total_modules', 0)}")
        print(f"📄 Total de conteúdos: {stats.get('total_contents', 0)}")
        
        # Salvar relatório
        import json
        report_file = Path(config['output_dir']) / 'scraping_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Relatório salvo em: {report_file}")
        print(f"📁 Arquivos salvos em: {config['output_dir']}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processo interrompido pelo usuário")
    except Exception as e:
        print(f"\n\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()
        print("\n✓ Recursos liberados")


if __name__ == "__main__":
    main()


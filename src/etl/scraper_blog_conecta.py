#!/usr/bin/env python3
"""
ETL Scraper para coleta de conteúdo do blog Conecta Ads
Converte artigos do blog em arquivos Markdown para uso no RAG
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import logging
from typing import List, Set, Optional
from urllib.parse import urljoin, urlparse
import argparse
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BlogScraper:
    """Scraper para coletar conteúdo do blog Conecta Ads"""
    
    def __init__(self, base_url: str = "https://www.conectaads.com.br/conteudos/", 
                 output_dir: str = "data/input/blog_conecta"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # URLs excluídas
        self.excluded_patterns = [
            'home', 'contato', 'lgpd', 'blog', 'quem-somos', 
            '/page/', 'categoria', 'politica-de-privacidade'
        ]
    
    def sanitize_filename(self, name: str) -> str:
        """Remove caracteres inválidos do nome do arquivo"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name.strip()
    
    def collect_urls(self) -> Set[str]:
        """Coleta todas as URLs do blog paginando até a última página"""
        logger.info("Iniciando coleta de URLs do blog...")
        
        blog_urls = set()
        page = 1
        pages_accessed = 0
        
        while True:
            url = f'{self.base_url}page/{page}/' if page > 1 else self.base_url
            logger.info(f"Acessando pagina {page}: {url}")
            
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                pages_accessed += 1
                
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.find_all('a', href=True)
                
                # Filtrar URLs que parecem ser de conteúdos do blog
                new_links = [
                    link['href'] for link in links
                    if (link['href'].startswith('https://www.conectaads.com.br/') and
                        not any(pattern in link['href'] for pattern in self.excluded_patterns))
                ]
                
                blog_urls.update(new_links)
                logger.info(f"Encontrados {len(new_links)} links na pagina {page}")
                
                # Verificar se há próxima página
                next_page_link = soup.find('a', href=lambda href: href and "/page/" in href)
                if not next_page_link:
                    logger.info("Ultima pagina encontrada")
                    break
                
                page += 1
                time.sleep(1)  # Rate limiting
                
            except requests.RequestException as e:
                logger.error(f"Erro ao acessar pagina {page}: {e}")
                break
        
        logger.info(f"Total de {len(blog_urls)} URLs encontrados em {pages_accessed} paginas")
        return blog_urls
    
    def extract_content(self, url: str) -> Optional[dict]:
        """Extrai conteúdo de uma URL específica"""
        try:
            logger.info(f"Acessando URL: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extrair título
            title = soup.find('h1') or soup.find('h2')
            if not title:
                logger.warning(f"Titulo nao encontrado para {url}")
                return None
            
            title_text = title.get_text(strip=True)
            
            # Extrair conteúdo principal
            content_selectors = [
                'div.elementor-widget-theme-post-content',
                'div.elementor-text-editor',
                'div.entry-content',
                'article .content',
                'main .content'
            ]
            
            content_div = None
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    break
            
            if not content_div:
                logger.warning(f"Conteudo nao encontrado em {url}")
                return None
            
            # Extrair elementos de conteúdo
            content_elements = content_div.find_all(['p', 'h2', 'h3', 'h4', 'ul', 'ol', 'strong', 'em'])
            
            content_text = []
            for element in content_elements:
                if element.name in ['h2', 'h3', 'h4']:
                    level = int(element.name[1])
                    content_text.append(f"\n{'#' * level} {element.get_text(strip=True)}\n")
                elif element.name == 'ol':
                    items = [f"{i+1}. {li.get_text(strip=True)}" for i, li in enumerate(element.find_all('li'))]
                    content_text.append("\n" + "\n".join(items) + "\n")
                elif element.name == 'ul':
                    items = [f"- {li.get_text(strip=True)}" for li in element.find_all('li')]
                    content_text.append("\n" + "\n".join(items) + "\n")
                elif element.name == 'strong':
                    content_text.append(f" **{element.get_text(strip=True)}** ")
                elif element.name == 'em':
                    content_text.append(f" *{element.get_text(strip=True)}* ")
                else:
                    # Para parágrafos, processar conteúdo interno
                    text = element.get_text(strip=True)
                    if text:
                        content_text.append(text)
            
            full_content = " ".join(content_text)
            # Limpar espaços múltiplos
            full_content = " ".join(full_content.split())
            
            return {
                'title': title_text,
                'content': full_content,
                'url': url,
                'word_count': len(full_content.split())
            }
            
        except requests.RequestException as e:
            logger.error(f"Erro ao processar {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado ao processar {url}: {e}")
            return None
    
    def save_content(self, content_data: dict) -> bool:
        """Salva o conteúdo extraído em arquivo Markdown"""
        try:
            filename = self.sanitize_filename(content_data['title']) + '.md'
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {content_data['title']}\n\n")
                f.write(content_data['content'])
                f.write(f"\n\n---\n\n")
                f.write(f"**Fonte:** [{content_data['url']}]({content_data['url']})\n")
                f.write(f"**Palavras:** {content_data['word_count']}\n")
                f.write(f"**Extraído em:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Conteudo salvo: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo: {e}")
            return False
    
    def scrape_all(self, max_articles: Optional[int] = None) -> dict:
        """Executa o scraping completo do blog"""
        logger.info("Iniciando scraping completo do blog...")
        
        # Coletar URLs
        urls = self.collect_urls()
        
        if max_articles:
            urls = list(urls)[:max_articles]
            logger.info(f"Limitando a {max_articles} artigos")
        
        # Processar cada URL
        results = {
            'total_urls': len(urls),
            'successful': 0,
            'failed': 0,
            'total_words': 0,
            'files_created': []
        }
        
        logger.info(f"Iniciando processamento de {len(urls)} artigos...")
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processando artigo {i}/{len(urls)}")
            
            content_data = self.extract_content(url)
            if content_data:
                if self.save_content(content_data):
                    results['successful'] += 1
                    results['total_words'] += content_data['word_count']
                    results['files_created'].append(content_data['title'])
                else:
                    results['failed'] += 1
            else:
                results['failed'] += 1
            
            # Rate limiting
            time.sleep(2)
        
        # Log final
        logger.info(f"Processo concluido!")
        logger.info(f"Estatisticas:")
        logger.info(f"   - URLs processadas: {results['total_urls']}")
        logger.info(f"   - Sucessos: {results['successful']}")
        logger.info(f"   - Falhas: {results['failed']}")
        logger.info(f"   - Total de palavras: {results['total_words']:,}")
        
        return results


def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(description='Scraper do blog Conecta Ads')
    parser.add_argument('--output-dir', '-o', 
                       default='data/input/blog_conecta',
                       help='Diretório de saída para os arquivos Markdown')
    parser.add_argument('--max-articles', '-m', type=int,
                       help='Número máximo de artigos para processar')
    parser.add_argument('--base-url', '-u',
                       default='https://www.conectaads.com.br/conteudos/',
                       help='URL base do blog')
    
    args = parser.parse_args()
    
    # Criar scraper
    scraper = BlogScraper(
        base_url=args.base_url,
        output_dir=args.output_dir
    )
    
    # Executar scraping
    try:
        results = scraper.scrape_all(max_articles=args.max_articles)
        
        # Salvar relatório
        report_file = Path(args.output_dir).parent / 'scraping_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE SCRAPING\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"URLs processadas: {results['total_urls']}\n")
            f.write(f"Sucessos: {results['successful']}\n")
            f.write(f"Falhas: {results['failed']}\n")
            f.write(f"Total de palavras: {results['total_words']:,}\n\n")
            f.write("Arquivos criados:\n")
            for title in results['files_created']:
                f.write(f"- {title}\n")
        
        logger.info(f"Relatorio salvo em: {report_file}")
        
    except KeyboardInterrupt:
        logger.info("Processo interrompido pelo usuario")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")


if __name__ == "__main__":
    main()
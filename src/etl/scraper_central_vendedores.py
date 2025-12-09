#!/usr/bin/env python3
"""
ETL Scraper para Central de Vendedores do Mercado Livre
Estrutura hierárquica: Cursos → Módulos → Conteúdos
Converte conteúdos educacionais em arquivos Markdown para uso no RAG
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import logging
import json
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
import argparse
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper_central_vendedores.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CentralVendedoresScraper:
    """Scraper para coletar conteúdo educacional do Central de Vendedores"""
    
    def __init__(self, 
                 cursos_url: str = "https://central.vendedores.mercadolivre.com.br/cursos",
                 output_dir: str = "data/input/central_vendedores",
                 use_selenium: bool = True):
        """
        Args:
            cursos_url: URL da página principal de cursos
            output_dir: Diretório de saída para os arquivos Markdown
            use_selenium: Se True, usa Selenium para páginas dinâmicas; se False, usa requests
        """
        self.cursos_url = cursos_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_selenium = use_selenium
        
        # Setup para requests (fallback)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Setup para Selenium
        self.driver = None
        if use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Configura o driver do Selenium"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Rodar em background
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver configurado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao configurar Selenium: {e}")
            logger.info("Usando requests como fallback")
            self.use_selenium = False
            self.driver = None
    
    def sanitize_filename(self, name: str, max_length: int = 100) -> str:
        """Remove caracteres inválidos do nome do arquivo e limita o tamanho"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        
        # Limpar espaços e caracteres especiais
        name = name.strip().replace('\n', ' ').replace('\r', '')
        name = ' '.join(name.split())  # Remove espaços múltiplos
        
        # Limitar tamanho
        if len(name) > max_length:
            name = name[:max_length].rsplit(' ', 1)[0]  # Corta na última palavra completa
        
        return name
    
    def get_page_content(self, url: str, wait_for_element: Optional[str] = None) -> Optional[BeautifulSoup]:
        """
        Obtém o conteúdo de uma página usando Selenium ou requests
        
        Args:
            url: URL da página
            wait_for_element: Seletor CSS do elemento a aguardar (apenas para Selenium)
        
        Returns:
            BeautifulSoup object ou None em caso de erro
        """
        if self.use_selenium and self.driver:
            try:
                logger.info(f"Acessando com Selenium: {url}")
                self.driver.get(url)
                
                # Aguardar carregamento se especificado
                if wait_for_element:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                    )
                else:
                    time.sleep(2)  # Aguardar carregamento básico
                
                # Scroll para garantir que conteúdo dinâmico seja carregado
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                html = self.driver.page_source
                return BeautifulSoup(html, 'html.parser')
                
            except TimeoutException:
                logger.error(f"Timeout ao carregar {url}")
                return None
            except Exception as e:
                logger.error(f"Erro ao acessar {url} com Selenium: {e}")
                return None
        else:
            # Fallback para requests
            try:
                logger.info(f"Acessando com requests: {url}")
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except requests.RequestException as e:
                logger.error(f"Erro ao acessar {url} com requests: {e}")
                return None
    
    def collect_course_urls(self) -> List[Dict[str, str]]:
        """
        Coleta URLs de todos os cursos da página principal
        
        Returns:
            Lista de dicts com 'title' e 'url' de cada curso
        """
        logger.info("Coletando URLs dos cursos...")
        
        soup = self.get_page_content(self.cursos_url, wait_for_element=".sc-card-content-image")
        if not soup:
            logger.error("Não foi possível acessar a página de cursos")
            return []
        
        courses = []
        
        # Procurar por cards de cursos (ajustar seletores conforme necessário)
        # Baseado no HTML fornecido, procuramos por imagens de cards e seus links pai
        card_images = soup.find_all('img', class_='sc-card-content-image')
        
        for img in card_images:
            # Encontrar o link pai do card
            card_link = img.find_parent('a')
            if card_link and card_link.get('href'):
                course_url = card_link['href']
                
                # Completar URL se for relativa
                if not course_url.startswith('http'):
                    course_url = urljoin(self.cursos_url, course_url)
                
                # Obter título do curso (pode estar no alt da imagem ou próximo)
                course_title = img.get('alt', '') or 'Curso sem título'
                
                courses.append({
                    'title': course_title.strip(),
                    'url': course_url
                })
                logger.info(f"Curso encontrado: {course_title}")
        
        logger.info(f"Total de {len(courses)} cursos encontrados")
        return courses
    
    def extract_modules_from_course(self, course_url: str) -> List[Dict]:
        """
        Extrai módulos de uma página de curso
        
        Returns:
            Lista de dicts com informações dos módulos
        """
        logger.info(f"Extraindo módulos do curso: {course_url}")
        
        soup = self.get_page_content(course_url, wait_for_element=".content-expandable-guie")
        if not soup:
            return []
        
        modules = []
        
        # Procurar por módulos expansíveis
        module_elements = soup.find_all('div', class_='content-expandable-guie')
        
        for idx, module_elem in enumerate(module_elements, 1):
            try:
                # Extrair título do módulo
                title_elem = module_elem.find('h1', class_='row-guie-title')
                module_title = title_elem.get_text(strip=True) if title_elem else f"Módulo {idx}"
                
                # Extrair descrição
                desc_elem = module_elem.find('p', class_='row-guie-description')
                module_description = desc_elem.get_text(strip=True) if desc_elem else ""
                
                # Extrair número do módulo
                number_elem = module_elem.find('h1', class_='andes-typography--weight-regular')
                module_number = number_elem.get_text(strip=True) if number_elem else str(idx)
                
                modules.append({
                    'number': module_number,
                    'title': module_title,
                    'description': module_description,
                    'element': module_elem  # Guardar referência para processar depois
                })
                
                logger.info(f"  Módulo {module_number}: {module_title}")
                
            except Exception as e:
                logger.error(f"Erro ao processar módulo {idx}: {e}")
                continue
        
        return modules
    
    def extract_contents_from_module(self, module_element) -> List[Dict]:
        """
        Extrai conteúdos de um módulo
        
        Returns:
            Lista de dicts com informações dos conteúdos
        """
        contents = []
        
        # Procurar por conteúdos simples dentro do módulo
        # Nota: Como o módulo é expansível, pode precisar de clique para abrir
        # Por enquanto, vamos tentar extrair os conteúdos que estão visíveis
        content_elements = module_element.find_next_siblings('div', class_='content-simple-row-guie', limit=20)
        
        # Se não encontrar como siblings, procurar dentro de um container
        if not content_elements:
            content_elements = module_element.find_all('div', class_='content-simple-row-guie')
        
        for content_elem in content_elements:
            try:
                # Extrair título do conteúdo
                title_elem = content_elem.find('h1', class_='row-guie-title')
                if not title_elem:
                    continue
                
                content_title = title_elem.get_text(strip=True)
                
                # Extrair descrição
                desc_elem = content_elem.find('p', class_='row-guie-description')
                content_description = desc_elem.get_text(strip=True) if desc_elem else ""
                
                # Extrair duração/tipo
                duration_elem = content_elem.find('p', class_='andes-typography--type-body')
                duration = duration_elem.get_text(strip=True) if duration_elem else ""
                
                # Extrair link (se houver)
                link_elem = content_elem.find('a') or content_elem.find_parent('a')
                content_url = link_elem.get('href') if link_elem else None
                
                # Detectar se é vídeo (pelo ícone de play ou thumbnail do YouTube)
                is_video = bool(content_elem.find('img', src=lambda x: x and 'youtube' in x)) or \
                          bool(content_elem.find('svg', {'width': '20', 'height': '20'}))
                
                contents.append({
                    'title': content_title,
                    'description': content_description,
                    'duration': duration,
                    'url': content_url,
                    'is_video': is_video
                })
                
            except Exception as e:
                logger.error(f"Erro ao processar conteúdo: {e}")
                continue
        
        return contents
    
    def extract_content_detail(self, content_url: str) -> Optional[Dict]:
        """
        Acessa uma URL de conteúdo específico e extrai informações detalhadas
        
        Returns:
            Dict com conteúdo extraído ou None
        """
        if not content_url:
            return None
        
        # Completar URL se for relativa
        if not content_url.startswith('http'):
            content_url = urljoin(self.cursos_url, content_url)
        
        soup = self.get_page_content(content_url)
        if not soup:
            return None
        
        try:
            # Extrair título principal
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else ""
            
            # Extrair conteúdo textual
            # Procurar por diferentes containers de conteúdo
            content_selectors = [
                'div.content-detail',
                'div.article-content',
                'article',
                'main'
            ]
            
            content_text = []
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    # Extrair parágrafos, listas, etc.
                    for elem in content_div.find_all(['p', 'h2', 'h3', 'h4', 'ul', 'ol', 'li']):
                        text = elem.get_text(strip=True)
                        if text:
                            content_text.append(text)
                    break
            
            return {
                'title': title_text,
                'content': '\n\n'.join(content_text),
                'url': content_url
            }
            
        except Exception as e:
            logger.error(f"Erro ao extrair detalhes de {content_url}: {e}")
            return None
    
    def save_course_content(self, course_data: Dict, modules_data: List[Dict]) -> bool:
        """
        Salva o conteúdo de um curso completo em arquivo Markdown
        
        Args:
            course_data: Dados do curso
            modules_data: Lista de módulos com seus conteúdos
        
        Returns:
            True se salvou com sucesso
        """
        try:
            # Criar nome do arquivo
            filename = self.sanitize_filename(course_data['title']) + '.md'
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Cabeçalho do curso
                f.write(f"# {course_data['title']}\n\n")
                f.write(f"**Fonte:** [{course_data['url']}]({course_data['url']})\n\n")
                f.write("---\n\n")
                
                # Escrever cada módulo
                for module in modules_data:
                    f.write(f"## Módulo {module['number']}: {module['title']}\n\n")
                    
                    if module['description']:
                        f.write(f"{module['description']}\n\n")
                    
                    # Escrever conteúdos do módulo
                    if module.get('contents'):
                        for content in module['contents']:
                            f.write(f"### {content['title']}\n\n")
                            
                            if content['description']:
                                f.write(f"{content['description']}\n\n")
                            
                            if content['duration']:
                                f.write(f"**Duração:** {content['duration']}\n\n")
                            
                            if content['is_video']:
                                f.write("**Tipo:** Vídeo\n\n")
                            
                            # Se tiver conteúdo detalhado extraído
                            if content.get('detailed_content'):
                                f.write(f"{content['detailed_content']}\n\n")
                            
                            if content.get('url'):
                                f.write(f"**Link:** [{content['url']}]({content['url']})\n\n")
                            
                            f.write("---\n\n")
                    
                    f.write("\n")
                
                # Rodapé
                f.write(f"**Extraído em:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Curso salvo: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar curso: {e}")
            return False
    
    def scrape_course(self, course_data: Dict, extract_details: bool = False) -> Dict:
        """
        Faz scraping completo de um curso (módulos e conteúdos)
        
        Args:
            course_data: Dict com 'title' e 'url' do curso
            extract_details: Se True, acessa cada URL de conteúdo para extrair mais detalhes
        
        Returns:
            Dict com resultados do scraping
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando curso: {course_data['title']}")
        logger.info(f"{'='*60}")
        
        result = {
            'course': course_data['title'],
            'url': course_data['url'],
            'modules_count': 0,
            'contents_count': 0,
            'success': False
        }
        
        # Extrair módulos
        modules = self.extract_modules_from_course(course_data['url'])
        if not modules:
            logger.warning(f"Nenhum módulo encontrado para o curso: {course_data['title']}")
            return result
        
        result['modules_count'] = len(modules)
        
        # Para cada módulo, extrair conteúdos
        for module in modules:
            contents = self.extract_contents_from_module(module['element'])
            module['contents'] = contents
            result['contents_count'] += len(contents)
            
            logger.info(f"    → {len(contents)} conteúdos encontrados no módulo {module['number']}")
            
            # Se solicitado, extrair detalhes de cada conteúdo
            if extract_details:
                for content in contents:
                    if content.get('url'):
                        logger.info(f"      Extraindo detalhes: {content['title']}")
                        details = self.extract_content_detail(content['url'])
                        if details:
                            content['detailed_content'] = details.get('content', '')
                        time.sleep(1)  # Rate limiting
            
            # Remover elemento BeautifulSoup antes de salvar (não é serializável)
            del module['element']
        
        # Salvar conteúdo
        if self.save_course_content(course_data, modules):
            result['success'] = True
        
        return result
    
    def scrape_all(self, max_courses: Optional[int] = None, extract_details: bool = False) -> Dict:
        """
        Executa o scraping completo de todos os cursos
        
        Args:
            max_courses: Limitar número de cursos (para testes)
            extract_details: Se True, extrai conteúdo detalhado de cada página
        
        Returns:
            Dict com estatísticas do scraping
        """
        logger.info("\n" + "="*80)
        logger.info("INICIANDO SCRAPING DO CENTRAL DE VENDEDORES")
        logger.info("="*80 + "\n")
        
        # Coletar URLs dos cursos
        courses = self.collect_course_urls()
        
        if not courses:
            logger.error("Nenhum curso encontrado!")
            return {'error': 'Nenhum curso encontrado'}
        
        if max_courses:
            courses = courses[:max_courses]
            logger.info(f"Limitando a {max_courses} cursos para teste")
        
        # Estatísticas
        stats = {
            'total_courses': len(courses),
            'processed_courses': 0,
            'successful_courses': 0,
            'failed_courses': 0,
            'total_modules': 0,
            'total_contents': 0,
            'courses_details': []
        }
        
        # Processar cada curso
        for i, course in enumerate(courses, 1):
            logger.info(f"\n[{i}/{len(courses)}] Processando curso...")
            
            try:
                result = self.scrape_course(course, extract_details=extract_details)
                
                stats['processed_courses'] += 1
                if result['success']:
                    stats['successful_courses'] += 1
                else:
                    stats['failed_courses'] += 1
                
                stats['total_modules'] += result['modules_count']
                stats['total_contents'] += result['contents_count']
                stats['courses_details'].append(result)
                
                # Rate limiting entre cursos
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Erro ao processar curso {course['title']}: {e}")
                stats['failed_courses'] += 1
                continue
        
        # Log final
        logger.info("\n" + "="*80)
        logger.info("SCRAPING CONCLUÍDO!")
        logger.info("="*80)
        logger.info(f"Cursos processados: {stats['processed_courses']}/{stats['total_courses']}")
        logger.info(f"Sucessos: {stats['successful_courses']}")
        logger.info(f"Falhas: {stats['failed_courses']}")
        logger.info(f"Total de módulos: {stats['total_modules']}")
        logger.info(f"Total de conteúdos: {stats['total_contents']}")
        
        return stats
    
    def close(self):
        """Fecha recursos (Selenium driver)"""
        if self.driver:
            self.driver.quit()
            logger.info("Selenium driver fechado")


def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(
        description='Scraper do Central de Vendedores do Mercado Livre',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output-dir', '-o', 
                       default='data/input/central_vendedores',
                       help='Diretório de saída para os arquivos Markdown')
    parser.add_argument('--max-courses', '-m', type=int,
                       help='Número máximo de cursos para processar (para testes)')
    parser.add_argument('--cursos-url', '-u',
                       default='https://central.vendedores.mercadolivre.com.br/cursos',
                       help='URL da página de cursos')
    parser.add_argument('--extract-details', '-d', action='store_true',
                       help='Extrair conteúdo detalhado de cada página (mais lento)')
    parser.add_argument('--no-selenium', action='store_true',
                       help='Usar apenas requests (sem Selenium)')
    
    args = parser.parse_args()
    
    # Criar diretório de logs
    Path('logs').mkdir(exist_ok=True)
    
    # Criar scraper
    scraper = CentralVendedoresScraper(
        cursos_url=args.cursos_url,
        output_dir=args.output_dir,
        use_selenium=not args.no_selenium
    )
    
    try:
        # Executar scraping
        stats = scraper.scrape_all(
            max_courses=args.max_courses,
            extract_details=args.extract_details
        )
        
        # Salvar relatório
        report_file = Path(args.output_dir) / 'scraping_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nRelatório salvo em: {report_file}")
        
    except KeyboardInterrupt:
        logger.info("\nProcesso interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
    finally:
        scraper.close()


if __name__ == "__main__":
    main()


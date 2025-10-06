"""
ETL and Scrapping Module
Módulo para coleta e processamento de dados do blog Conecta Ads
"""

from .scraper import BlogScraper
from .analyzer import ContentAnalyzer

__version__ = "1.0.0"
__author__ = "Conecta Ads"
__email__ = "contato@conectaads.com.br"

__all__ = ["BlogScraper", "ContentAnalyzer"]

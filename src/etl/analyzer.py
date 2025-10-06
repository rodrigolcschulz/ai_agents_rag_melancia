#!/usr/bin/env python3
"""
Analisador de conteúdo para arquivos Markdown
Gera estatísticas e insights sobre o conteúdo coletado
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any
import re
from collections import Counter
import json

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar matplotlib para português
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)


class ContentAnalyzer:
    """Analisador de conteúdo para arquivos Markdown"""
    
    def __init__(self, input_dir: str = "../melanc.ia/Input/Blog"):
        self.input_dir = Path(input_dir)
        self.data = []
        self.stats = {}
    
    def load_files(self) -> List[Dict[str, Any]]:
        """Carrega e analisa todos os arquivos Markdown"""
        logger.info("🔎 Carregando arquivos Markdown...")
        
        md_files = list(self.input_dir.glob("*.md"))
        logger.info(f"📁 Encontrados {len(md_files)} arquivos")
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extrair metadados do arquivo
                file_data = self._analyze_file(file_path, content)
                self.data.append(file_data)
                
            except Exception as e:
                logger.error(f"❌ Erro ao processar {file_path}: {e}")
        
        logger.info(f"✅ {len(self.data)} arquivos processados com sucesso")
        return self.data
    
    def _analyze_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Analisa um arquivo individual"""
        # Contar palavras
        words = content.split()
        word_count = len(words)
        
        # Contar caracteres
        char_count = len(content)
        char_count_no_spaces = len(content.replace(' ', ''))
        
        # Contar linhas
        line_count = len(content.split('\n'))
        
        # Extrair título (primeira linha que começa com #)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem
        
        # Extrair URL de origem se disponível
        url_match = re.search(r'\*\*Fonte:\*\*\s*\[([^\]]+)\]\(([^)]+)\)', content)
        source_url = url_match.group(2) if url_match else None
        
        # Contar cabeçalhos
        headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        header_count = len(headers)
        
        # Contar listas
        list_items = re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE)
        numbered_lists = re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE)
        list_count = len(list_items) + len(numbered_lists)
        
        # Análise de sentenças
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Palavras mais comuns (excluindo stop words básicas)
        stop_words = {
            'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'da', 'do', 'das', 'dos',
            'em', 'na', 'no', 'nas', 'nos', 'para', 'por', 'com', 'sem', 'sobre', 'entre',
            'que', 'quem', 'qual', 'quais', 'onde', 'quando', 'como', 'porque', 'se', 'mas',
            'e', 'ou', 'nem', 'também', 'já', 'ainda', 'sempre', 'nunca', 'muito', 'pouco',
            'mais', 'menos', 'bem', 'mal', 'sim', 'não', 'talvez', 'é', 'são', 'foi', 'foram',
            'ser', 'estar', 'ter', 'haver', 'fazer', 'dizer', 'ver', 'saber', 'querer', 'poder',
            'dever', 'vir', 'ir', 'dar', 'falar', 'trabalhar', 'viver', 'pensar', 'conhecer',
            'encontrar', 'começar', 'acabar', 'continuar', 'voltar', 'passar', 'chegar', 'sair',
            'entrar', 'ficar', 'deixar', 'levar', 'trazer', 'mostrar', 'escrever', 'ler',
            'ouvir', 'sentir', 'gostar', 'amar', 'odiar', 'precisar', 'usar', 'precisar',
            'tentar', 'conseguir', 'perder', 'ganhar', 'comprar', 'vender', 'pagar', 'cobrar',
            'abrir', 'fechar', 'ligar', 'desligar', 'acender', 'apagar', 'subir', 'descer',
            'andar', 'correr', 'parar', 'esperar', 'buscar', 'procurar', 'encontrar', 'perder',
            'ganhar', 'vencer', 'perder', 'jogar', 'brincar', 'estudar', 'aprender', 'ensinar',
            'explicar', 'entender', 'compreender', 'lembrar', 'esquecer', 'imaginar', 'sonhar',
            'dormir', 'acordar', 'comer', 'beber', 'cozinhar', 'lavar', 'limpar', 'organizar',
            'arrumar', 'construir', 'destruir', 'criar', 'inventar', 'descobrir', 'explorar',
            'viajar', 'visitar', 'conhecer', 'apresentar', 'cumprimentar', 'agradecer',
            'pedir', 'oferecer', 'aceitar', 'recusar', 'concordar', 'discordar', 'discutir',
            'conversar', 'perguntar', 'responder', 'contar', 'narrar', 'descrever', 'explicar',
            'definir', 'classificar', 'comparar', 'diferenciar', 'relacionar', 'conectar',
            'separar', 'dividir', 'juntar', 'unir', 'misturar', 'combinar', 'escolher',
            'selecionar', 'preferir', 'gostar', 'amar', 'odiar', 'detestar', 'suportar',
            'tolerar', 'aceitar', 'recusar', 'rejeitar', 'aprovar', 'desaprovar', 'criticar',
            'elogiar', 'parabenizar', 'felicitar', 'consolar', 'ajudar', 'auxiliar', 'apoiar',
            'sustentar', 'defender', 'proteger', 'cuidar', 'tratar', 'curar', 'salvar',
            'resgatar', 'libertar', 'prender', 'capturar', 'pegar', 'soltar', 'largar',
            'abandonar', 'deixar', 'perder', 'encontrar', 'achar', 'descobrir', 'revelar',
            'mostrar', 'esconder', 'ocultar', 'disfarçar', 'mascarar', 'fingir', 'simular',
            'imitar', 'copiar', 'reproduzir', 'duplicar', 'multiplicar', 'aumentar', 'diminuir',
            'reduzir', 'crescer', 'desenvolver', 'evoluir', 'melhorar', 'piorar', 'deteriorar',
            'danificar', 'quebrar', 'consertar', 'reparar', 'restaurar', 'renovar', 'atualizar',
            'modernizar', 'antiquar', 'envelhecer', 'jovem', 'velho', 'novo', 'antigo',
            'recente', 'passado', 'presente', 'futuro', 'hoje', 'ontem', 'amanhã', 'agora',
            'antes', 'depois', 'durante', 'enquanto', 'quando', 'onde', 'como', 'porque',
            'para', 'por', 'com', 'sem', 'sobre', 'sob', 'entre', 'dentro', 'fora', 'perto',
            'longe', 'aqui', 'ali', 'lá', 'cá', 'acima', 'abaixo', 'em', 'cima', 'embaixo',
            'na', 'frente', 'atrás', 'lado', 'direita', 'esquerda', 'centro', 'meio', 'fim',
            'início', 'começo', 'final', 'primeiro', 'último', 'segundo', 'terceiro', 'quarto',
            'quinto', 'sexto', 'sétimo', 'oitavo', 'nono', 'décimo', 'um', 'dois', 'três',
            'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove', 'dez', 'cem', 'mil', 'milhão',
            'bilhão', 'trilhão', 'zero', 'nada', 'tudo', 'algo', 'algum', 'alguma', 'alguns',
            'algumas', 'nenhum', 'nenhuma', 'nenhuns', 'nenhumas', 'outro', 'outra', 'outros',
            'outras', 'mesmo', 'mesma', 'mesmos', 'mesmas', 'diferente', 'diferentes', 'igual',
            'iguais', 'similar', 'similares', 'parecido', 'parecidos', 'distinto', 'distintos',
            'único', 'única', 'únicos', 'únicas', 'especial', 'especiais', 'comum', 'comuns',
            'normal', 'normais', 'raro', 'raros', 'frequente', 'frequentes', 'ocasional',
            'ocasionais', 'sempre', 'nunca', 'jamais', 'às', 'vezes', 'geralmente', 'normalmente',
            'habitualmente', 'costumeiramente', 'frequentemente', 'raramente', 'ocasionalmente',
            'eventualmente', 'possivelmente', 'provavelmente', 'certamente', 'definitivamente',
            'absolutamente', 'completamente', 'totalmente', 'parcialmente', 'ligeiramente',
            'levemente', 'bastante', 'muito', 'pouco', 'demais', 'suficiente', 'insuficiente',
            'adequado', 'inadequado', 'apropriado', 'inapropriado', 'correto', 'incorreto',
            'certo', 'errado', 'verdadeiro', 'falso', 'real', 'irreal', 'verdadeiro', 'mentiroso',
            'honesto', 'desonesto', 'sincero', 'insincero', 'direto', 'indireto', 'claro',
            'obscuro', 'evidente', 'evidente', 'óbvio', 'claro', 'confuso', 'complicado',
            'simples', 'fácil', 'difícil', 'possível', 'impossível', 'provável', 'improvável',
            'certo', 'incerto', 'seguro', 'inseguro', 'perigoso', 'seguro', 'risco', 'segurança',
            'perigo', 'ameaça', 'proteção', 'defesa', 'ataque', 'luta', 'guerra', 'paz',
            'conflito', 'harmonia', 'discordância', 'concordância', 'acordo', 'desacordo',
            'consenso', 'divergência', 'convergência', 'união', 'separação', 'divisão',
            'integração', 'desintegração', 'organização', 'desorganização', 'ordem', 'desordem',
            'caos', 'estrutura', 'destrutura', 'construção', 'destruição', 'criação', 'aniquilação',
            'nascimento', 'morte', 'vida', 'existência', 'inexistência', 'realidade', 'ilusão',
            'sonho', 'pesadelo', 'fantasia', 'imaginação', 'criatividade', 'originalidade',
            'inovação', 'tradição', 'modernidade', 'antiguidade', 'passado', 'presente', 'futuro'
        }
        
        # Contar palavras (excluindo stop words)
        word_freq = Counter()
        for word in words:
            word_lower = word.lower().strip('.,!?;:"()[]{}')
            if len(word_lower) > 2 and word_lower not in stop_words:
                word_freq[word_lower] += 1
        
        # Palavras mais comuns
        common_words = dict(word_freq.most_common(10))
        
        return {
            'filename': file_path.name,
            'title': title,
            'source_url': source_url,
            'word_count': word_count,
            'char_count': char_count,
            'char_count_no_spaces': char_count_no_spaces,
            'line_count': line_count,
            'header_count': header_count,
            'list_count': list_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0,
            'common_words': common_words,
            'file_size_kb': file_path.stat().st_size / 1024
        }
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Gera estatísticas gerais dos dados"""
        if not self.data:
            logger.warning("⚠️ Nenhum dado disponível para análise")
            return {}
        
        df = pd.DataFrame(self.data)
        
        stats = {
            'total_files': len(df),
            'total_words': df['word_count'].sum(),
            'total_characters': df['char_count'].sum(),
            'total_sentences': df['sentence_count'].sum(),
            'avg_words_per_file': df['word_count'].mean(),
            'avg_chars_per_file': df['char_count'].mean(),
            'avg_sentences_per_file': df['sentence_count'].mean(),
            'avg_words_per_sentence': df['avg_words_per_sentence'].mean(),
            'word_count_stats': {
                'min': df['word_count'].min(),
                'max': df['word_count'].max(),
                'mean': df['word_count'].mean(),
                'median': df['word_count'].median(),
                'std': df['word_count'].std(),
                'q25': df['word_count'].quantile(0.25),
                'q75': df['word_count'].quantile(0.75)
            },
            'file_size_stats': {
                'min_kb': df['file_size_kb'].min(),
                'max_kb': df['file_size_kb'].max(),
                'mean_kb': df['file_size_kb'].mean(),
                'total_kb': df['file_size_kb'].sum()
            }
        }
        
        self.stats = stats
        return stats
    
    def create_visualizations(self, output_dir: str = "../melanc.ia/Output/Analysis"):
        """Cria visualizações dos dados"""
        if not self.data:
            logger.warning("⚠️ Nenhum dado disponível para visualização")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.data)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Distribuição de palavras por arquivo
        plt.figure(figsize=(12, 8))
        plt.hist(df['word_count'], bins=20, edgecolor='black', alpha=0.7)
        plt.title('Distribuição do Número de Palavras por Arquivo', fontsize=16, fontweight='bold')
        plt.xlabel('Número de Palavras', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'word_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top 20 arquivos com mais palavras
        plt.figure(figsize=(14, 10))
        top_files = df.nlargest(20, 'word_count')
        bars = plt.barh(range(len(top_files)), top_files['word_count'])
        plt.yticks(range(len(top_files)), [title[:50] + '...' if len(title) > 50 else title 
                                          for title in top_files['title']])
        plt.xlabel('Número de Palavras', fontsize=12)
        plt.title('Top 20 Arquivos com Mais Palavras', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 10, bar.get_y() + bar.get_height()/2, 
                    f'{int(width):,}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / 'top_files_by_words.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Relação entre palavras e caracteres
        plt.figure(figsize=(10, 8))
        plt.scatter(df['word_count'], df['char_count'], alpha=0.6, s=50)
        plt.xlabel('Número de Palavras', fontsize=12)
        plt.ylabel('Número de Caracteres', fontsize=12)
        plt.title('Relação entre Palavras e Caracteres', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Adicionar linha de tendência
        z = np.polyfit(df['word_count'], df['char_count'], 1)
        p = np.poly1d(z)
        plt.plot(df['word_count'], p(df['word_count']), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(output_path / 'words_vs_chars.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Distribuição de tamanho de arquivos
        plt.figure(figsize=(10, 6))
        plt.hist(df['file_size_kb'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('Tamanho do Arquivo (KB)', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title('Distribuição do Tamanho dos Arquivos', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'file_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Palavras mais comuns (agregadas)
        all_common_words = {}
        for file_data in self.data:
            for word, count in file_data['common_words'].items():
                all_common_words[word] = all_common_words.get(word, 0) + count
        
        top_words = dict(Counter(all_common_words).most_common(30))
        
        plt.figure(figsize=(12, 10))
        words = list(top_words.keys())
        counts = list(top_words.values())
        
        bars = plt.barh(range(len(words)), counts)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Frequência Total', fontsize=12)
        plt.title('Top 30 Palavras Mais Comuns (Agregadas)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / 'most_common_words.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Visualizações salvas em: {output_path}")
    
    def generate_report(self, output_dir: str = "../melanc.ia/Output/Analysis"):
        """Gera relatório completo em texto"""
        if not self.stats:
            self.generate_statistics()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / 'content_analysis_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DE CONTEÚDO\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("ESTATÍSTICAS GERAIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total de arquivos: {self.stats['total_files']}\n")
            f.write(f"Total de palavras: {self.stats['total_words']:,}\n")
            f.write(f"Total de caracteres: {self.stats['total_characters']:,}\n")
            f.write(f"Total de sentenças: {self.stats['total_sentences']:,}\n\n")
            
            f.write("ESTATÍSTICAS POR ARQUIVO\n")
            f.write("-" * 25 + "\n")
            f.write(f"Média de palavras por arquivo: {self.stats['avg_words_per_file']:.1f}\n")
            f.write(f"Média de caracteres por arquivo: {self.stats['avg_chars_per_file']:.1f}\n")
            f.write(f"Média de sentenças por arquivo: {self.stats['avg_sentences_per_file']:.1f}\n")
            f.write(f"Média de palavras por sentença: {self.stats['avg_words_per_sentence']:.1f}\n\n")
            
            f.write("DISTRIBUIÇÃO DE PALAVRAS\n")
            f.write("-" * 25 + "\n")
            word_stats = self.stats['word_count_stats']
            f.write(f"Mínimo: {word_stats['min']:,} palavras\n")
            f.write(f"Máximo: {word_stats['max']:,} palavras\n")
            f.write(f"Média: {word_stats['mean']:.1f} palavras\n")
            f.write(f"Mediana: {word_stats['median']:.1f} palavras\n")
            f.write(f"Desvio padrão: {word_stats['std']:.1f}\n")
            f.write(f"Q1 (25%): {word_stats['q25']:.1f} palavras\n")
            f.write(f"Q3 (75%): {word_stats['q75']:.1f} palavras\n\n")
            
            f.write("TAMANHO DOS ARQUIVOS\n")
            f.write("-" * 20 + "\n")
            size_stats = self.stats['file_size_stats']
            f.write(f"Tamanho mínimo: {size_stats['min_kb']:.1f} KB\n")
            f.write(f"Tamanho máximo: {size_stats['max_kb']:.1f} KB\n")
            f.write(f"Tamanho médio: {size_stats['mean_kb']:.1f} KB\n")
            f.write(f"Tamanho total: {size_stats['total_kb']:.1f} KB\n\n")
            
            f.write("TOP 10 ARQUIVOS COM MAIS PALAVRAS\n")
            f.write("-" * 35 + "\n")
            df = pd.DataFrame(self.data)
            top_files = df.nlargest(10, 'word_count')
            for i, (_, row) in enumerate(top_files.iterrows(), 1):
                f.write(f"{i:2d}. {row['title'][:60]}{'...' if len(row['title']) > 60 else ''}\n")
                f.write(f"    {row['word_count']:,} palavras, {row['file_size_kb']:.1f} KB\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Relatório gerado automaticamente pelo ContentAnalyzer\n")
        
        logger.info(f"📋 Relatório salvo em: {report_file}")
    
    def save_dataframe(self, output_dir: str = "../melanc.ia/Output/Analysis"):
        """Salva DataFrame completo em CSV"""
        if not self.data:
            logger.warning("⚠️ Nenhum dado disponível para salvar")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.data)
        csv_file = output_path / 'content_analysis_data.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"💾 Dados salvos em CSV: {csv_file}")


def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(description='Analisador de conteúdo Markdown')
    parser.add_argument('--input-dir', '-i', 
                       default='../melanc.ia/Input/Blog',
                       help='Diretório com arquivos Markdown')
    parser.add_argument('--output-dir', '-o',
                       default='../melanc.ia/Output/Analysis',
                       help='Diretório de saída para relatórios e gráficos')
    
    args = parser.parse_args()
    
    # Criar analisador
    analyzer = ContentAnalyzer(input_dir=args.input_dir)
    
    try:
        # Carregar e analisar arquivos
        logger.info("🚀 Iniciando análise de conteúdo...")
        analyzer.load_files()
        
        # Gerar estatísticas
        stats = analyzer.generate_statistics()
        
        # Criar visualizações
        analyzer.create_visualizations(args.output_dir)
        
        # Gerar relatório
        analyzer.generate_report(args.output_dir)
        
        # Salvar dados
        analyzer.save_dataframe(args.output_dir)
        
        logger.info("✅ Análise concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro durante a análise: {e}")


if __name__ == "__main__":
    import numpy as np  # Import necessário para polyfit
    main()

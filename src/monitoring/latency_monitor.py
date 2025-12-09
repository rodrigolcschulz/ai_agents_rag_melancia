#!/usr/bin/env python3
"""
Monitoramento de Latência e Performance do MelancIA

Boas práticas de Engenharia de IA:
- Monitorar P50, P95, P99 (não apenas média)
- Acompanhar latência por provider/modelo
- Detectar degradação de performance
- Alertas para slow queries
- Análise de tendências ao longo do tempo
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.interaction_logger import InteractionLogger


class LatencyMonitor:
    """Monitor de latência e performance para sistemas RAG."""
    
    def __init__(self, db_path: str = "data/evaluation/interactions.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            print(f"⚠️  Database não encontrado: {self.db_path}")
            print("Execute o sistema primeiro para gerar dados de interação.")
            sys.exit(1)
    
    def get_latency_percentiles(self, provider: str = None, hours: int = 24) -> Dict:
        """
        Calcula percentis de latência.
        
        Boas práticas:
        - P50 (mediana): experiência típica do usuário
        - P95: 95% dos usuários tem latência menor que isso
        - P99: detecta outliers/problemas
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                COUNT(*) as total,
                AVG(latency_seconds) as avg,
                MIN(latency_seconds) as min,
                MAX(latency_seconds) as max
            FROM interactions
            WHERE datetime(timestamp) > datetime('now', '-{} hours')
        """.format(hours)
        
        if provider:
            query += f" AND provider = '{provider}'"
        
        cursor.execute(query)
        row = cursor.fetchone()
        
        stats = {
            "total": row[0] or 0,
            "avg": row[1] or 0,
            "min": row[2] or 0,
            "max": row[3] or 0
        }
        
        # Calcular percentis manualmente (SQLite não tem PERCENTILE_CONT em todas versões)
        query_percentiles = """
            SELECT latency_seconds 
            FROM interactions
            WHERE datetime(timestamp) > datetime('now', '-{} hours')
        """.format(hours)
        
        if provider:
            query_percentiles += f" AND provider = '{provider}'"
        
        query_percentiles += " ORDER BY latency_seconds"
        
        cursor.execute(query_percentiles)
        latencies = [row[0] for row in cursor.fetchall()]
        
        if latencies:
            import statistics
            stats["p50"] = statistics.median(latencies)
            stats["p95"] = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
            stats["p99"] = latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0]
        else:
            stats["p50"] = stats["p95"] = stats["p99"] = 0
        
        conn.close()
        return stats
    
    def get_slow_queries(self, threshold: float = 10.0, limit: int = 10) -> List[Dict]:
        """Identifica queries lentas (SLO violation)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                timestamp,
                query,
                latency_seconds,
                provider,
                model_name
            FROM interactions
            WHERE latency_seconds > ?
            ORDER BY latency_seconds DESC
            LIMIT ?
        """, (threshold, limit))
        
        slow_queries = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return slow_queries
    
    def get_latency_trend(self, hours: int = 24, bucket_hours: int = 1) -> List[Tuple]:
        """Analisa tendência de latência ao longo do tempo."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as time_bucket,
                COUNT(*) as count,
                AVG(latency_seconds) as avg_latency,
                MAX(latency_seconds) as max_latency
            FROM interactions
            WHERE datetime(timestamp) > datetime('now', '-{} hours')
            GROUP BY time_bucket
            ORDER BY time_bucket
        """.format(hours))
        
        trend = cursor.fetchall()
        conn.close()
        return trend
    
    def get_provider_comparison(self, hours: int = 24) -> Dict[str, Dict]:
        """Compara performance entre providers."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                provider,
                COUNT(*) as count,
                AVG(latency_seconds) as avg_latency,
                MIN(latency_seconds) as min_latency,
                MAX(latency_seconds) as max_latency
            FROM interactions
            WHERE datetime(timestamp) > datetime('now', '-{} hours')
                AND provider IS NOT NULL
            GROUP BY provider
        """.format(hours))
        
        comparison = {}
        for row in cursor.fetchall():
            provider = row[0]
            comparison[provider] = {
                "count": row[1],
                "avg": row[2],
                "min": row[3],
                "max": row[4]
            }
        
        conn.close()
        return comparison
    
    def print_report(self, hours: int = 24):
        """Gera relatório completo de performance."""
        print("\n" + "="*80)
        print(f"📊 RELATÓRIO DE PERFORMANCE - Últimas {hours}h")
        print("="*80)
        
        # Overview geral
        print("\n### 🎯 OVERVIEW GERAL")
        stats = self.get_latency_percentiles(hours=hours)
        print(f"Total de queries: {stats['total']}")
        print(f"Latência média: {stats['avg']:.2f}s")
        print(f"P50 (mediana): {stats['p50']:.2f}s")
        print(f"P95: {stats['p95']:.2f}s")
        print(f"P99: {stats['p99']:.2f}s")
        print(f"Min/Max: {stats['min']:.2f}s / {stats['max']:.2f}s")
        
        # SLO Health Check
        print("\n### ⚡ SLO HEALTH CHECK")
        p95_slo = 5.0  # SLO: 95% das queries < 5s
        p99_slo = 10.0  # SLO: 99% das queries < 10s
        
        p95_ok = stats['p95'] < p95_slo
        p99_ok = stats['p99'] < p99_slo
        
        print(f"{'✅' if p95_ok else '❌'} P95 < {p95_slo}s: {stats['p95']:.2f}s {'(OK)' if p95_ok else '(VIOLATION)'}")
        print(f"{'✅' if p99_ok else '❌'} P99 < {p99_slo}s: {stats['p99']:.2f}s {'(OK)' if p99_ok else '(VIOLATION)'}")
        
        # Comparação por provider
        print("\n### 🤖 PERFORMANCE POR PROVIDER")
        comparison = self.get_provider_comparison(hours=hours)
        for provider, provider_stats in comparison.items():
            emoji = "🦙" if provider == "ollama" else "🤖"
            print(f"\n{emoji} **{provider.upper()}**")
            print(f"  Queries: {provider_stats['count']}")
            print(f"  Média: {provider_stats['avg']:.2f}s")
            print(f"  Min/Max: {provider_stats['min']:.2f}s / {provider_stats['max']:.2f}s")
            
            # Calcular percentis por provider
            provider_percentiles = self.get_latency_percentiles(provider=provider, hours=hours)
            print(f"  P50: {provider_percentiles['p50']:.2f}s | P95: {provider_percentiles['p95']:.2f}s | P99: {provider_percentiles['p99']:.2f}s")
        
        # Slow queries
        print("\n### 🐌 TOP 10 SLOW QUERIES")
        slow_queries = self.get_slow_queries(threshold=10.0, limit=10)
        if slow_queries:
            for i, sq in enumerate(slow_queries, 1):
                query_preview = sq['query'][:60] + "..." if len(sq['query']) > 60 else sq['query']
                print(f"{i}. {sq['latency_seconds']:.2f}s - {sq['provider']} - {query_preview}")
        else:
            print("✅ Nenhuma query lenta detectada!")
        
        # Tendência
        print("\n### 📈 TENDÊNCIA (por hora)")
        trend = self.get_latency_trend(hours=hours)
        if trend:
            print(f"{'Hora':<20} {'Queries':<10} {'Média (s)':<12} {'Max (s)':<10}")
            print("-" * 52)
            for time_bucket, count, avg_lat, max_lat in trend:
                print(f"{time_bucket:<20} {count:<10} {avg_lat:<12.2f} {max_lat:<10.2f}")
        
        # Recomendações
        print("\n### 💡 RECOMENDAÇÕES")
        if stats['p95'] > p95_slo:
            print("⚠️  P95 acima do SLO - Considere:")
            print("   • Aumentar porcentagem de OpenAI (mais rápido)")
            print("   • Reduzir timeout do Ollama")
            print("   • Investigar queries lentas")
        
        if stats['p99'] > p99_slo:
            print("⚠️  P99 acima do SLO - Problemas de outliers:")
            print("   • Verificar se Ollama está travando")
            print("   • Analisar queries complexas")
        
        if stats['avg'] > 5.0:
            print("⚠️  Latência média alta - Ações:")
            print("   • Otimizar retriever (reduzir k de 15 para 10)")
            print("   • Usar embeddings menores")
            print("   • Cachear queries frequentes")
        
        if comparison.get('ollama', {}).get('avg', 0) > 10.0:
            print("⚠️  Ollama muito lento:")
            print("   • Verificar recursos do servidor")
            print("   • Considerar modelo menor (llama3.1:8b -> phi3:mini)")
            print("   • Aumentar fallback para OpenAI")
        
        print("\n" + "="*80)
        print("✅ Relatório concluído!\n")


def main():
    """Função principal do monitor."""
    parser = argparse.ArgumentParser(description='Monitor de Latência do MelancIA')
    parser.add_argument(
        '--hours', 
        type=int, 
        default=24, 
        help='Período de análise em horas (padrão: 24)'
    )
    parser.add_argument(
        '--db-path',
        default='data/evaluation/interactions.db',
        help='Caminho do banco de dados'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Modo watch (atualiza a cada 30s)'
    )
    
    args = parser.parse_args()
    
    monitor = LatencyMonitor(db_path=args.db_path)
    
    if args.watch:
        import time
        try:
            while True:
                print("\033[2J\033[H")  # Limpar terminal
                monitor.print_report(hours=args.hours)
                print(f"\n🔄 Atualizando em 30s... (Ctrl+C para sair)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n👋 Monitor encerrado.")
    else:
        monitor.print_report(hours=args.hours)


if __name__ == "__main__":
    main()


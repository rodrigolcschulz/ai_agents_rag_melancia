# 📊 Monitoramento de Latência e Performance - Boas Práticas

## 🎯 Visão Geral

Este documento descreve as melhores práticas para monitorar a performance do MelancIA RAG Agent.

---

## 📈 Métricas Essenciais

### 1. **Latência**

**Por que monitorar:**
- Experiência do usuário depende diretamente do tempo de resposta
- Identificar degradação de performance
- Comparar providers (Ollama vs OpenAI)

**Métricas chave:**

| Métrica | O que é | SLO Recomendado | Por que é importante |
|---------|---------|-----------------|---------------------|
| **P50** (Mediana) | 50% dos usuários têm latência menor que isso | < 3s | Experiência típica |
| **P95** | 95% dos usuários têm latência menor que isso | < 5s | Experiência da maioria |
| **P99** | 99% dos usuários têm latência menor que isso | < 10s | Detecta outliers |
| **Média** | Média aritmética | < 4s | Tendência geral |
| **Max** | Pior caso | < 30s | Detecta travamentos |

**⚠️ Armadilha comum:** Monitorar apenas a média esconde problemas!
- Se P50 = 2s, P95 = 3s, P99 = 60s → Média = 3s (parece OK)
- Mas 1% dos usuários espera 1 minuto! (experiência ruim)

---

### 2. **Throughput**

- **Queries por segundo** (QPS)
- **Queries por hora** por provider
- **Taxa de sucesso** vs erros

---

### 3. **Qualidade**

- **Feedback positivo/negativo** (👍/👎)
- **Rating médio** (1-5 estrelas)
- **Métricas automáticas**: Faithfulness, Answer Relevancy

---

## 🛠️ Como Monitorar

### 1. **Interface Web - Estatísticas em Tempo Real**

Ao rodar `web_interface_with_eval.py`, clique em "🔄 Atualizar" no painel lateral para ver:
- Latência média e por modelo
- Distribuição de providers
- Feedback dos usuários
- Queries lentas

### 2. **Script de Monitoramento Standalone**

```bash
# Relatório das últimas 24h
python src/monitoring/latency_monitor.py

# Últimas 12h
python src/monitoring/latency_monitor.py --hours 12

# Modo watch (atualiza a cada 30s)
python src/monitoring/latency_monitor.py --watch
```

**Output do relatório:**
```
📊 RELATÓRIO DE PERFORMANCE - Últimas 24h
================================================================================

### 🎯 OVERVIEW GERAL
Total de queries: 150
Latência média: 4.23s
P50 (mediana): 3.12s
P95: 6.45s
P99: 12.34s
Min/Max: 1.23s / 15.67s

### ⚡ SLO HEALTH CHECK
❌ P95 < 5.0s: 6.45s (VIOLATION)
❌ P99 < 10.0s: 12.34s (VIOLATION)

### 🤖 PERFORMANCE POR PROVIDER
🦙 OLLAMA
  Queries: 120
  Média: 5.12s
  P50: 4.23s | P95: 8.12s | P99: 14.23s

🤖 OPENAI
  Queries: 30
  Média: 1.89s
  P50: 1.45s | P95: 2.34s | P99: 3.12s
```

### 3. **Acesso Direto ao Banco SQLite**

```bash
# Entrar no SQLite
sqlite3 data/evaluation/interactions.db

# Queries úteis:

# Ver últimas 10 interações
SELECT timestamp, query, latency_seconds, provider 
FROM interactions 
ORDER BY timestamp DESC 
LIMIT 10;

# Queries mais lentas
SELECT query, latency_seconds, provider, timestamp
FROM interactions
ORDER BY latency_seconds DESC
LIMIT 10;

# Latência média por provider
SELECT provider, 
       COUNT(*) as total,
       AVG(latency_seconds) as avg_latency,
       MAX(latency_seconds) as max_latency
FROM interactions
GROUP BY provider;
```

---

## 🚨 Alertas e SLOs (Service Level Objectives)

### SLOs Recomendados para Produção:

| Métrica | Target | Crítico |
|---------|--------|---------|
| P50 | < 3s | > 5s |
| P95 | < 5s | > 10s |
| P99 | < 10s | > 20s |
| Taxa de sucesso | > 99% | < 95% |
| Uptime | > 99.9% | < 99% |

### Como Implementar Alertas:

**Exemplo com script simples:**

```python
# Em produção, use Prometheus/Grafana
from src.monitoring import LatencyMonitor

monitor = LatencyMonitor()
stats = monitor.get_latency_percentiles(hours=1)

# Alerta se P95 > 5s na última hora
if stats['p95'] > 5.0:
    send_alert(f"⚠️ P95 latency too high: {stats['p95']:.2f}s")

# Alerta se Ollama está muito lento
comparison = monitor.get_provider_comparison(hours=1)
if comparison.get('ollama', {}).get('avg', 0) > 10.0:
    send_alert("⚠️ Ollama averaging > 10s - consider increasing OpenAI fallback")
```

---

## 🔧 Otimizações Baseadas em Métricas

### Se P95 > 5s:

1. **Aumentar % de OpenAI:**
   ```bash
   export OLLAMA_PERCENTAGE=0.5  # 50% Ollama, 50% OpenAI
   ```

2. **Reduzir timeout do Ollama:**
   ```python
   # web_interface_with_eval.py linha ~220
   timeout_seconds = 10.0  # De 15s para 10s
   ```

3. **Otimizar retriever:**
   ```python
   # config.py
   RETRIEVER_K = 10  # De 15 para 10 documentos
   ```

### Se Ollama >> OpenAI em latência:

1. **Verificar recursos do servidor:**
   ```bash
   # CPU/RAM disponível?
   htop
   
   # Ollama rodando?
   ollama ps
   ```

2. **Trocar para modelo menor:**
   ```bash
   # De llama3.1:8b (8B parâmetros) para:
   ollama pull phi3:mini  # 3.8B parâmetros, mais rápido
   ```

3. **Aumentar fallback para OpenAI:**
   ```bash
   export OLLAMA_PERCENTAGE=0.3  # 30% Ollama, 70% OpenAI
   ```

### Se queries específicas são lentas:

1. **Implementar cache:**
   ```python
   # Cachear perguntas frequentes
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def get_cached_response(query_hash):
       return qa_chain.invoke({"question": query})
   ```

2. **Otimizar chunking:**
   ```python
   # retriever.py
   chunk_size=800,  # De 1000 para 800
   ```

---

## 📊 Ferramentas de Monitoramento em Produção

### **Nível 1: Built-in** (Atual)
✅ SQLite + InteractionLogger  
✅ Gradio UI com stats  
✅ Script de relatório CLI  

**Bom para:** MVP, testes, poucos usuários

---

### **Nível 2: Básico** (Recomendado para produção inicial)
- **Prometheus + Grafana**
  - Métricas em tempo real
  - Dashboards customizáveis
  - Alertas automáticos

```python
# Adicionar métricas Prometheus
from prometheus_client import Counter, Histogram, start_http_server

query_latency = Histogram('query_latency_seconds', 'Query latency')
query_counter = Counter('queries_total', 'Total queries', ['provider'])

@query_latency.time()
def process_query(message):
    # ... código existente ...
    query_counter.labels(provider=provider).inc()
```

---

### **Nível 3: Avançado** (Escala)
- **DataDog / New Relic** - APM completo
- **Sentry** - Error tracking
- **PostHog / Mixpanel** - Analytics de produto
- **LangSmith / LangFuse** - LLM observability específica

---

## 📋 Checklist de Monitoramento

### Diário:
- [ ] Verificar P95 < 5s
- [ ] Verificar taxa de erro < 1%
- [ ] Revisar feedback negativo

### Semanal:
- [ ] Analisar tendência de latência
- [ ] Comparar Ollama vs OpenAI
- [ ] Identificar queries problemáticas
- [ ] Revisar custos (API OpenAI)

### Mensal:
- [ ] Otimizar modelos baseado em uso
- [ ] Ajustar SLOs conforme necessário
- [ ] Análise de ROI (custo vs qualidade)

---

## 🎓 Referências e Leitura Adicional

- [Google SRE Book - Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [AWS - Best Practices for Monitoring LLM Applications](https://aws.amazon.com/blogs/machine-learning/)
- [LangChain - Production Monitoring](https://python.langchain.com/docs/guides/productionization/monitoring/)
- [OpenAI - Best Practices for Production](https://platform.openai.com/docs/guides/production-best-practices)

---

## 💡 Resumo: Por que P50/P95/P99?

**P50 (Mediana)** = Experiência típica  
**P95** = Garantia de qualidade para a maioria  
**P99** = Detecta problemas raros mas críticos  

**Média sozinha esconde problemas!**

Exemplo real:
- 99 queries em 2s, 1 query em 200s
- **Média**: 4s ❌ (parece OK, mas enganoso)
- **P99**: 200s ✅ (mostra o problema real)

---

**🍉 MelancIA** - Monitoramento proativo para experiência excepcional!


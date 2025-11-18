# 🧪 Guia de Experimentação com LLMs

Este guia detalha como experimentar com diferentes modelos de linguagem no MelâncIA.

## 📋 Índice

1. [Setup Inicial](#setup-inicial)
2. [Provedores de LLM](#provedores-de-llm)
3. [Executando Benchmarks](#executando-benchmarks)
4. [Análise de Resultados](#análise-de-resultados)
5. [MLflow Tracking](#mlflow-tracking)
6. [Próximos Passos](#próximos-passos)

## 🚀 Setup Inicial

### 1. Instalar Dependências

```bash
# Instalar todas as dependências
pip install -r requirements.txt

# Ou apenas experimentação
pip install mlflow transformers torch ollama langchain-ollama
```

### 2. Configurar Provedores

#### OpenAI (Padrão)

```bash
# Adicionar API key no .env
echo "OPENAI_API_KEY=sk-..." >> .env
```

#### Ollama (Local - Recomendado)

```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Baixe de: https://ollama.ai/download

# Baixar modelos
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull phi3:mini

# Verificar instalação
ollama list
```

#### HuggingFace (Opcional)

```bash
# Criar token em: https://huggingface.co/settings/tokens
echo "HUGGINGFACE_API_TOKEN=hf_..." >> .env
```

## 🤖 Provedores de LLM

### OpenAI

**Prós:**
- ✅ Alta qualidade
- ✅ Rápido
- ✅ Sempre disponível

**Contras:**
- ❌ Custo por request
- ❌ Requer internet
- ❌ Limites de rate

**Modelos:**
- `gpt-4o-mini` - Melhor custo-benefício (~$0.15/1M tokens)
- `gpt-4o` - Máxima qualidade (~$5/1M tokens)
- `gpt-3.5-turbo` - Mais barato (~$0.50/1M tokens)

### Ollama (Local)

**Prós:**
- ✅ Gratuito
- ✅ Privacidade total
- ✅ Sem limites
- ✅ Offline

**Contras:**
- ❌ Requer GPU/RAM
- ❌ Setup inicial
- ❌ Qualidade varia

**Modelos Recomendados:**

| Modelo | Tamanho | RAM | Qualidade | Velocidade |
|--------|---------|-----|-----------|------------|
| `phi3:mini` | 3.8GB | 8GB | Boa | Rápido |
| `llama3.1:8b` | 4.7GB | 8GB | Muito Boa | Médio |
| `mistral:7b` | 4.1GB | 8GB | Muito Boa | Médio |
| `gemma2:9b` | 5.5GB | 16GB | Excelente | Lento |
| `llama3.1:70b` | 40GB | 64GB | Excelente | Muito Lento |

### HuggingFace

**Prós:**
- ✅ Gratuito (com limites)
- ✅ Muitos modelos
- ✅ Fácil uso

**Contras:**
- ❌ Rate limits
- ❌ Mais lento
- ❌ Requer internet

## 🏃 Executando Benchmarks

### Teste Rápido

```bash
# Testa modelos disponíveis com 3 perguntas
python src/experiments/run_experiments.py --mode quick
```

**Saída esperada:**
```
🚀 TESTE RÁPIDO - Experimentação de LLMs
================================================================================

📝 Perguntas de teste:
   1. O que é Retail Media?
   2. Quais são as principais métricas de performance?
   3. Como funciona o ACOS no Mercado Livre?

🔍 Testando OpenAI...
   ✓ OpenAI configurado
🔍 Testando Ollama...
   ✓ Ollama configurado

🎯 Testando 2 modelo(s) em 3 perguntas
================================================================================

📊 RELATÓRIO DE BENCHMARK
================================================================================

🤖 OPENAI - gpt-4o-mini
--------------------------------------------------------------------------------
  ⚡ Latência:    1.25s (±0.15s)
  📝 Tokens:      450 (média)
  💰 Custo:       $0.0023
  ⭐ Qualidade:   0.87/1.0
  🎯 Relevância:  0.92/1.0

🤖 OLLAMA - llama3.1:8b
--------------------------------------------------------------------------------
  ⚡ Latência:    2.10s (±0.30s)
  📝 Tokens:      520 (média)
  💰 Custo:       $0.0000
  ⭐ Qualidade:   0.82/1.0
  🎯 Relevância:  0.88/1.0

================================================================================
🏆 RANKING
--------------------------------------------------------------------------------
  🥇 Melhor Qualidade:        openai - gpt-4o-mini (0.87)
  ⚡ Mais Rápido:             openai - gpt-4o-mini (1.25s)
  💎 Melhor Custo-Benefício:  ollama - llama3.1:8b
================================================================================
```

### Benchmark Completo

```bash
# Testa todos os modelos com todas as perguntas
python src/experiments/run_experiments.py --mode full
```

Testa:
- OpenAI: gpt-4o-mini, gpt-3.5-turbo
- Ollama: llama3.1:8b, mistral:7b, phi3:mini (se disponíveis)

### Uso Programático

```python
from src.experiments.multi_llm import MultiLLMManager
from src.experiments.benchmark import ModelBenchmark
from src.agent.retriever import get_retriever
from src.agent.memory import get_memory
from src.agent import config

# Setup RAG
retriever = get_retriever(str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL)
memory = get_memory(config.HISTORY_FILE)

# Criar benchmark
benchmark = ModelBenchmark(retriever, memory)

# Adicionar modelos
llm1 = MultiLLMManager.create_llm("openai", "gpt-4o-mini")
benchmark.add_model("openai", "gpt-4o-mini", llm1)

llm2 = MultiLLMManager.create_llm("ollama", "llama3.1:8b")
benchmark.add_model("ollama", "llama3.1:8b", llm2)

# Executar
perguntas = [
    "O que é Retail Media?",
    "Como otimizar campanhas?",
]

results = benchmark.run(perguntas)
benchmark.print_report()

# Salvar resultados
report = benchmark.generate_report()
report.to_csv("meu_benchmark.csv")
```

## 📊 Análise de Resultados

### Jupyter Notebook

```bash
jupyter lab
# Abrir: notebooks/experimentacao_llms.ipynb
```

O notebook inclui:
- 📊 Gráficos comparativos
- 📈 Análise de trade-offs
- 🧪 Testes interativos
- 💾 Export de resultados

### Resultados Salvos

Os benchmarks são salvos automaticamente em:
- `data/experiments/benchmark_YYYYMMDD_HHMMSS.json` - Dados completos
- `data/experiments/benchmark_YYYYMMDD_HHMMSS.csv` - Formato tabular

### Visualização Pandas

```python
import pandas as pd

# Carregar resultados
df = pd.read_csv("data/experiments/benchmark_20241118_153045.csv")

# Análises
print(df.groupby("model_name")["quality_score"].mean())
print(df.groupby("model_name")["latency_seconds"].mean())

# Gráfico
df.groupby("model_name")["quality_score"].mean().plot(kind="bar")
```

## 🔬 MLflow Tracking

### Iniciar MLflow UI

```bash
# Iniciar servidor
mlflow ui --port 5000

# Ou via script
python src/experiments/run_experiments.py --mode ui

# Abrir: http://localhost:5000
```

### Recursos do MLflow

**Experiments:**
- Compare múltiplos runs
- Visualize métricas
- Filtre e ordene resultados

**Metrics:**
- Latência média
- Qualidade média
- Custo total
- Tokens usados

**Parameters:**
- Modelo usado
- Temperatura
- Max tokens
- Provider

**Artifacts:**
- Resultados JSON
- Logs de execução
- Configurações

### Acessar via Python

```python
from src.mlops.tracking import ExperimentTracker

tracker = ExperimentTracker("melancia-retail-media")

# Ver resumo
tracker.print_summary()

# Buscar melhor modelo
best = tracker.get_best_run(metric="quality_avg")
print(f"Melhor modelo: {best['params.model']}")
```

## 🎯 Próximos Passos

### Fase 2: Fine-Tuning

Após escolher o melhor modelo, você pode:
1. Coletar dataset específico de Retail Media
2. Fine-tuning com LoRA/QLoRA
3. Avaliar modelo personalizado
4. Deploy em produção

### Fase 3: MLOps Avançado

- Continuous training
- A/B testing de modelos
- Monitoramento de drift
- Feedback loop de usuários

## 💡 Dicas e Boas Práticas

### Performance

- **Ollama**: Use GPU se possível (`nvidia-smi` para verificar)
- **OpenAI**: Use batch de perguntas para economizar
- **Cache**: Respostas repetidas são automáticas (ChromaDB)

### Qualidade

- **Temperature**: 0.3-0.5 para respostas consistentes
- **Max Tokens**: 500-1000 para respostas completas
- **Retriever**: k=3-5 documentos é ideal

### Custo

- **Desenvolvimento**: Use Ollama local
- **Produção**: Compare custo vs qualidade
- **Híbrido**: Ollama para preview, OpenAI para final

## 🆘 Troubleshooting

### Ollama não conecta

```bash
# Verificar se está rodando
curl http://localhost:11434/api/tags

# Reiniciar
ollama serve

# Ver logs
journalctl -u ollama -f
```

### Erro de memória

```bash
# Usar modelo menor
ollama pull phi3:mini  # 3.8GB vs 4.7GB

# Ou aumentar swap (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### HuggingFace rate limit

- Use Ollama local ao invés
- Ou aguarde 1 hora
- Ou crie nova conta/token

## 📚 Recursos Adicionais

- [Ollama Documentation](https://github.com/ollama/ollama)
- [HuggingFace Models](https://huggingface.co/models)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [LangChain Guide](https://python.langchain.com/docs/get_started/introduction)

---

**Questões?** Abra uma issue no GitHub ou consulte a documentação!


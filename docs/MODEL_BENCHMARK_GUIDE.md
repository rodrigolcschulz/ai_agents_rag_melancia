# 🚀 Guia de Benchmark de Modelos LLM

## 🎯 Modelos Recomendados para Seu Hardware

Baseado no seu AMD Ryzen 5 3600X com 15GB RAM (CPU only):

### ⚡ Categoria: Ultra Rápidos (<3s resposta)

| Modelo | Tamanho | RAM | Descrição | Install |
|--------|---------|-----|-----------|---------|
| **gemma2:2b** | 1.6GB | 3GB | Google, ultra rápido e eficiente | `ollama pull gemma2:2b` |
| **qwen2.5:3b** | 2GB | 4GB | Alibaba, multilíngue, ótimo | `ollama pull qwen2.5:3b` |
| **phi3:mini** | 2.3GB | 4GB | Microsoft, otimizado para CPU | `ollama pull phi3:mini` |

**Melhor escolha**: `qwen2.5:3b` ou `phi3:mini`

---

### 🎯 Categoria: Rápidos e com Boa Qualidade (3-5s)

| Modelo | Tamanho | RAM | Descrição | Install |
|--------|---------|-----|-----------|---------|
| **llama3.2:3b** | 2GB | 4GB | Meta, excelente qualidade/velocidade | `ollama pull llama3.2:3b` |
| **phi3:medium** | 7.9GB | 8GB | Microsoft, melhor contexto | `ollama pull phi3:medium` |
| **gemma2:9b** | 5.5GB | 8GB | Google, alta qualidade | `ollama pull gemma2:9b` |

**Melhor escolha**: `llama3.2:3b` (equilíbrio perfeito)

---

### 🏆 Categoria: Qualidade Máxima (5-10s)

| Modelo | Tamanho | RAM | Descrição | Install |
|--------|---------|-----|-----------|---------|
| **mistral:7b-instruct-q4_K_M** | 4.1GB | 6GB | Mistral AI, excelente raciocínio | `ollama pull mistral:7b-instruct-q4_K_M` |
| **llama3.1:8b-instruct-q4_K_M** | 4.7GB | 8GB | Meta, versão avançada | `ollama pull llama3.1:8b-instruct-q4_K_M` |

**Melhor escolha**: `mistral:7b-instruct-q4_K_M` (melhor custo-benefício)

---

## 🆕 Modelos Novos Recomendados (Não estavam no doc original)

### 1. **Qwen 2.5 3B** ⭐⭐⭐⭐⭐
```bash
ollama pull qwen2.5:3b
```
- **Por quê**: Alibaba lançou recentemente, multilíngue excelente
- **Vantagens**: Rápido, contexto longo (32k tokens), ótimo em português
- **Latência esperada**: 2-4s
- **Qualidade**: 8/10

### 2. **Phi-3 Medium** ⭐⭐⭐⭐
```bash
ollama pull phi3:medium
```
- **Por quê**: Versão maior do Phi-3, mantém eficiência
- **Vantagens**: Melhor raciocínio que o mini, ainda rápido
- **Latência esperada**: 4-6s
- **Qualidade**: 8.5/10

### 3. **Gemma 2 9B** ⭐⭐⭐⭐
```bash
ollama pull gemma2:9b
```
- **Por quê**: Google, qualidade próxima a modelos maiores
- **Vantagens**: Excelente em tarefas específicas, bem treinado
- **Latência esperada**: 5-8s
- **Qualidade**: 8.5/10

---

## 📊 Como Executar o Benchmark

### Passo 1: Instalar Modelos Recomendados

```bash
# Modelos essenciais (rápido, ~10 min)
ollama pull gemma2:2b
ollama pull qwen2.5:3b
ollama pull phi3:mini
ollama pull llama3.2:3b

# Modelos adicionais de qualidade (opcional, ~15 min)
ollama pull mistral:7b-instruct-q4_K_M
ollama pull gemma2:9b
```

**Tempo total de download**: 15-30 minutos (depende da conexão)

---

### Passo 2: Ver Guia de Instalação

```bash
cd /home/coneta/ai_agents_rag_melancia
python src/experiments/benchmark_models_mlflow.py --install-guide
```

**Output:**
```
📥 MODELOS RECOMENDADOS PARA SEU HARDWARE
======================================================================

1. gemma2:2b
   Tamanho: 1.6GB
   Ultra rápido, boa qualidade

2. phi3:mini
   Tamanho: 2.3GB
   Microsoft, otimizado para CPU

3. llama3.2:3b
   Tamanho: 2GB
   Meta, excelente qualidade
...
```

---

### Passo 3: Teste Rápido (Verificar se funciona)

```bash
# Testar modelo específico
python src/experiments/benchmark_models_mlflow.py --test-model ollama::phi3:mini
```

**Output esperado:**
```
🧪 Teste rápido: ollama::phi3:mini
----------------------------------------------------------------------
Pergunta: O que é ACOS no contexto de Product Ads?

Resposta (2.34s):
----------------------------------------------------------------------
ACOS (Advertising Cost of Sale) é uma métrica que mede...
----------------------------------------------------------------------

✅ Modelo funcional! Latência: 2.34s
```

---

### Passo 4: Benchmark Completo

#### Opção A: Modo QUICK (3-5 minutos)
```bash
# Testa apenas modelos ultra rápidos + OpenAI
python src/experiments/benchmark_models_mlflow.py --mode quick
```

#### Opção B: Modo FAST (10-15 minutos) ⭐ RECOMENDADO
```bash
# Testa modelos rápidos + qualidade + OpenAI
python src/experiments/benchmark_models_mlflow.py --mode fast
```

#### Opção C: Modo FULL (30-60 minutos)
```bash
# Testa TODOS os modelos com todas as perguntas
python src/experiments/benchmark_models_mlflow.py --mode full
```

---

### Passo 5: Visualizar Resultados no MLflow

```bash
# Em um terminal separado
mlflow ui --port 5000

# Abrir navegador em:
# http://localhost:5000
```

**No MLflow você verá:**
- ✅ Comparação lado a lado de todos os modelos
- ✅ Gráficos de latência vs qualidade
- ✅ Métricas detalhadas (P50, P95, P99)
- ✅ Custo por query (OpenAI vs grátis)

---

## 📊 Resultados Esperados

### Previsão de Performance no Seu Hardware:

| Modelo | Latência | Qualidade | Custo/mês | Recomendação |
|--------|----------|-----------|-----------|--------------|
| gemma2:2b | ~1.5s | 6.5/10 | $0 | ⭐⭐⭐ Protótipos rápidos |
| qwen2.5:3b | ~2.5s | 7.5/10 | $0 | ⭐⭐⭐⭐⭐ **MELHOR GERAL** |
| phi3:mini | ~2s | 7/10 | $0 | ⭐⭐⭐⭐ Ótimo equilíbrio |
| llama3.2:3b | ~3s | 7.8/10 | $0 | ⭐⭐⭐⭐⭐ **ALTA QUALIDADE** |
| mistral:7b-q4 | ~6s | 8.2/10 | $0 | ⭐⭐⭐ Qualidade máxima |
| gemma2:9b | ~7s | 8.3/10 | $0 | ⭐⭐⭐ Alternativa qualidade |
| gpt-4o-mini | ~1.5s | 9.5/10 | ~$15 | ⭐⭐⭐⭐⭐ **BASELINE** |

---

## 💡 Recomendações Finais

### Para Desenvolvimento:
```bash
# Use o mais rápido para iteração
qwen2.5:3b  # 2.5s, grátis, boa qualidade
```

### Para Produção (Híbrido):
```bash
# 70% llama3.2:3b (grátis, 3s)
# 30% gpt-4o-mini (pago, 1.5s, melhor qualidade)

export OLLAMA_PERCENTAGE=0.7
export OLLAMA_MODEL=llama3.2:3b
```

### Para Qualidade Máxima:
```bash
# Use 100% OpenAI
export USE_MODEL_ROUTER=false
# Custo: ~$20-50/mês
```

---

## 🎯 Próximos Passos

1. **Instalar modelos base**:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull llama3.2:3b
   ollama pull phi3:mini
   ```

2. **Executar benchmark**:
   ```bash
   python src/experiments/benchmark_models_mlflow.py --mode fast
   ```

3. **Analisar resultados no MLflow**:
   ```bash
   mlflow ui --port 5000
   ```

4. **Escolher modelo** baseado em:
   - Latência aceitável
   - Qualidade necessária
   - Orçamento disponível

5. **Atualizar configuração**:
   ```python
   # src/mlops/model_router.py
   DEFAULT_OLLAMA_MODEL = "llama3.2:3b"  # Ou o escolhido
   ```

---

## 📁 Onde Encontrar Resultados

```
data/experiments/
├── model_comparison_results.json  # Resultados em JSON
└── mlruns/                         # MLflow tracking data
    └── melancia_model_comparison/  # Experimento específico
```

---

## 🐛 Troubleshooting

### Erro: "ollama: command not found"
```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

### Erro: "model not found"
```bash
# Ver modelos instalados
ollama list

# Instalar modelo faltante
ollama pull <modelo>
```

### Erro: "CUDA not available"
✅ **Normal!** Seu sistema não tem GPU NVIDIA. O código usa CPU automaticamente.

### Latência muito alta (>20s)
- Verificar se outros processos estão usando CPU
- Considerar modelo menor (gemma2:2b)
- Aumentar % de OpenAI

---

**🍉 MelancIA** - Pronto para comparar modelos e escolher o melhor para seu projeto!


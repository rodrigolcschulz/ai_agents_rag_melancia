# 🔍 Análise de Resultados do Benchmark

## 📊 Resultados Obtidos

### ✅ Modelos que Funcionaram

| Modelo | Latência | Qualidade | Relevância | Custo/query |
|--------|----------|-----------|------------|-------------|
| **gemma2:2b** | 63.18s 🔴 | 1.00 | 0.50 | $0 |
| **qwen2.5:3b** | 104.67s 🔴 | 1.00 | 0.52 | $0 |
| **phi3:mini** | 266.60s 🔴🔴 | 1.00 | 0.58 | $0 |
| **llama3.2:3b** | 127.38s 🔴 | 1.00 | 0.63 | $0 |
| **gpt-4o-mini** | 7.89s ✅ | 1.00 | 0.58 | ~$0.000375 |
| **gpt-3.5-turbo** | 5.34s ✅ | 0.80 | 0.42 | ~$0.000125 |

### ❌ Modelos que Falharam (Não Instalados)

- phi3:medium → `ollama pull phi3:medium`
- gemma2:9b → `ollama pull gemma2:9b`

---

## 🚨 PROBLEMA CRÍTICO: Ollama Extremamente Lento

### Latências Esperadas vs Reais

| Modelo | Esperado | Real | Diferença |
|--------|----------|------|-----------|
| gemma2:2b | ~2s | 63s | **31x mais lento** 🔴 |
| qwen2.5:3b | ~3s | 105s | **35x mais lento** 🔴 |
| phi3:mini | ~2s | 267s | **133x mais lento** 🔴🔴 |
| llama3.2:3b | ~3s | 127s | **42x mais lento** 🔴 |

### 🔍 Possíveis Causas

#### 1. **CPU Sobrecarregado**
```bash
# Verificar uso de CPU
htop

# Ver processos do Ollama
ps aux | grep ollama
```

**Sintomas:**
- Outros processos pesados rodando
- CPU usage perto de 100%
- Sistema travando

**Solução:**
- Fechar outros processos
- Não rodar múltiplos modelos simultaneamente

---

#### 2. **RAM Insuficiente → Usando SWAP**
```bash
# Verificar uso de memória
free -h

# Ver swap
swapon --show
```

**Sintomas:**
- Modelos maiores que RAM disponível
- Sistema lento em geral
- Disco trabalhando muito (LED piscando)

**Seu hardware:**
- RAM: 15GB total
- gemma2:2b: ~4GB RAM necessária
- qwen2.5:3b: ~6GB RAM necessária
- phi3:mini: ~4GB RAM necessária
- llama3.2:3b: ~5GB RAM necessária

**Se rodar tudo ao mesmo tempo = 19GB > 15GB = SWAP!**

**Solução:**
```bash
# Configurar Ollama para carregar apenas 1 modelo por vez
export OLLAMA_MAX_LOADED_MODELS=1
```

---

#### 3. **Ollama Configurado Incorretamente**
```bash
# Ver configuração atual
ollama ps

# Ver logs
journalctl -u ollama -f
```

**Possíveis problemas:**
- Ollama limitando threads
- Configuração de memória errada

**Solução:**
```bash
# Configurar threads otimizados para Ryzen 5 3600X (12 threads)
export OLLAMA_NUM_PARALLEL=2  # Máximo 2 requests paralelos
export OLLAMA_NUM_THREADS=12  # Usar todos os threads do CPU
```

---

#### 4. **Benchmark Rodando Modelos em Paralelo**

**PROBLEMA**: O código pode estar tentando carregar TODOS os modelos na RAM!

**Solução**: Verificar se o benchmark limpa memória entre testes.

---

## ⚡ Soluções Imediatas

### 1. Otimizar Configuração do Ollama

```bash
# Criar/editar ~/.ollama/config
cat > ~/.ollama/config << 'EOF'
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_THREADS=12
OLLAMA_NUM_PARALLEL=1
EOF

# Reiniciar Ollama
sudo systemctl restart ollama
# Ou se rodando manualmente:
pkill ollama && ollama serve &
```

### 2. Limpar RAM Entre Testes

```bash
# Antes de executar benchmark
sync && sudo sysctl -w vm.drop_caches=3

# Executar benchmark
python src/experiments/benchmark_models_mlflow.py --mode quick
```

### 3. Testar Modelos Individualmente

```bash
# Testar um por vez
python src/experiments/benchmark_models_mlflow.py --test-model ollama::gemma2:2b

# Se rápido (2-3s), problema é memória/concorrência
# Se lento (60s+), problema é CPU/configuração
```

---

## 🎯 Modelos Adicionais Recomendados

### Prioridade ALTA (Vale a Pena Instalar)

#### 1. **mistral:7b-instruct-q4_K_M** ⭐⭐⭐⭐⭐
```bash
ollama pull mistral:7b-instruct-q4_K_M
```
**Por quê:**
- Qualidade EXCELENTE (melhor que llama3.2:3b)
- Raciocínio superior
- Bom em português
- ~4.1GB (cabe na RAM)

**Latência esperada:** 6-10s (se Ollama otimizado)

---

#### 2. **gemma2:9b** ⭐⭐⭐⭐
```bash
ollama pull gemma2:9b
```
**Por quê:**
- Google, qualidade alta
- Melhor que gemma2:2b
- ~5.5GB

**Latência esperada:** 8-12s

---

### Prioridade MÉDIA (Testes Específicos)

#### 3. **llama3.1:8b-instruct-q4_K_M** ⭐⭐⭐
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```
**Por quê:**
- Versão avançada do llama3.2
- Mais capabilities
- ~4.7GB

**Latência esperada:** 8-15s

---

### Prioridade BAIXA (Só se Sobrar Tempo/Espaço)

#### 4. **phi3:medium** ⭐⭐
```bash
ollama pull phi3:medium
```
**Por quê:**
- Melhor contexto que phi3:mini
- Mas phi3:mini já está MUITO lento (~267s)
- ~7.9GB (pode causar swap)

**Não recomendo** até resolver o problema de latência.

---

## 💡 Recomendação Final

### Ação Imediata:

**1. Otimizar Ollama:**
```bash
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_THREADS=12
sudo systemctl restart ollama
```

**2. Testar individualmente:**
```bash
python src/experiments/benchmark_models_mlflow.py --test-model ollama::gemma2:2b
```

**Expectativa:** Deveria cair de 63s para ~2-3s

---

**3. Se ainda lento, rodar com apenas 1 modelo:**
```bash
# Remover modelos temporariamente
ollama rm qwen2.5:3b
ollama rm phi3:mini
ollama rm llama3.2:3b

# Testar apenas gemma2:2b
python src/experiments/benchmark_models_mlflow.py --test-model ollama::gemma2:2b

# Se rápido agora, problema ERA memória!
```

---

### Modelos para Instalar (Depois de Otimizar):

**Se latência cair para normal (~2-5s):**
```bash
# 1. ESSENCIAL - Melhor raciocínio
ollama pull mistral:7b-instruct-q4_K_M

# 2. RECOMENDADO - Alta qualidade
ollama pull gemma2:9b

# 3. OPCIONAL - Versão avançada
ollama pull llama3.1:8b-instruct-q4_K_M
```

**Comparação depois do benchmark:**
- mistral vs llama3.2 (qual tem melhor qualidade?)
- gemma2:9b vs gemma2:2b (vale a pena o modelo maior?)

---

## 📈 Próximos Passos

1. ✅ **Otimizar Ollama** (configurações de memória/threads)
2. ✅ **Testar individualmente** (verificar se latência normaliza)
3. ✅ **Instalar mistral:7b** (se latência OK)
4. ✅ **Benchmark novamente** com configurações otimizadas
5. ✅ **Comparar no MLflow** (qualidade vs latência vs custo)

---

**🍉 Resultado Esperado Após Otimização:**

| Modelo | Latência Atual | Latência Esperada | Status |
|--------|----------------|-------------------|--------|
| gemma2:2b | 63s 🔴 | ~2s ✅ | -97% |
| qwen2.5:3b | 105s 🔴 | ~3s ✅ | -97% |
| llama3.2:3b | 127s 🔴 | ~3s ✅ | -98% |
| **mistral:7b** | N/A | ~6s ✅ | **NOVO** |
| **gemma2:9b** | 0s ❌ | ~8s ✅ | **INSTALAR** |

**Decisão Final:**
- Se Ollama rápido: Usar modelo local (mistral ou llama3.2)
- Se Ollama lento: Usar OpenAI (gpt-4o-mini = 7.89s, ~$0.0004/query)


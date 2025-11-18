# 🖥️ Guia de Setup - Hardware Específico

## 📊 Especificações do Sistema

- **CPU**: AMD Ryzen 5 3600X (6 cores, 12 threads, 2.2-3.8 GHz)
  - ✅ Processador razoável para LLMs pequenos/médios
  
- **RAM**: 15 GB total (~10 GB disponível)
  - ⚠️ Limitado para LLMs grandes (>8B)
  - ✅ Suficiente para modelos 2-7B quantizados
  
- **GPU**: AMD Radeon RX 570/580 (Ellesmere)
  - ❌ Sem suporte CUDA (NVIDIA)
  - ⚠️ ROCm disponível mas complexo de configurar
  - 💡 Recomendação: Usar CPU para inferência
  
- **Armazenamento**: 439 GB total
  - Usado: 71 GB (17%)
  - Disponível: 346 GB (79%)
  - ✅ Espaço suficiente para:
    - Múltiplos modelos LLM (2-5 GB cada)
    - Datasets de treinamento (1-10 GB)
    - Vector databases (ChromaDB: 100-500 MB)
    - Logs e experimentos MLflow (1-5 GB)
    - Modelos fine-tunados (2-10 GB cada)

### 💾 Estimativa de Uso de Disco

| Item | Tamanho Estimado | Prioridade |
|------|------------------|------------|
| Ollama + 3 modelos (phi3, llama3.2, gemma2) | ~6 GB | Alta |
| ChromaDB vector database | 200-500 MB | Alta |
| MLflow experiments (6 meses) | 2-3 GB | Média |
| Datasets RAG (markdown) | 50-200 MB | Alta |
| Fine-tuning datasets | 500 MB - 2 GB | Baixa |
| Modelos fine-tunados | 2-5 GB cada | Baixa |
| **Total estimado (uso completo)** | **15-25 GB** | - |

**✅ Conclusão**: Você tem **346 GB disponíveis**, mais que suficiente para todas as fases do projeto, incluindo múltiplos modelos e experimentos.

### 📊 Monitorar Uso de Disco

```bash
# Ver uso geral
df -h /

# Ver uso do projeto
du -sh /home/coneta/ai_agents_rag_melancia

# Ver tamanho dos modelos Ollama
du -sh ~/.ollama/models

# Ver tamanho do ChromaDB
du -sh /home/coneta/ai_agents_rag_melancia/data/vector_db

# Ver tamanho dos experimentos MLflow
du -sh /home/coneta/ai_agents_rag_melancia/mlruns
```

**Limpeza periódica:**
```bash
# Remover modelos não utilizados do Ollama
ollama list
ollama rm <modelo_não_usado>

# Limpar cache do Python
find . -type d -name __pycache__ -exec rm -rf {} +

# Limpar experimentos antigos do MLflow (opcional)
# Manter apenas últimos 3 meses
```

## ✅ O Que É Viável

### Modelos Locais Recomendados (Ollama)

| Modelo | Tamanho | RAM | Velocidade | Recomendação |
|--------|---------|-----|------------|--------------|
| **phi3:mini** | 2.3GB | 4GB | Rápido | ⭐⭐⭐⭐⭐ MELHOR OPÇÃO |
| **llama3.2:3b** | 2GB | 4GB | Rápido | ⭐⭐⭐⭐⭐ ÓTIMO |
| **gemma2:2b** | 1.6GB | 3GB | Muito Rápido | ⭐⭐⭐⭐ BOM |
| **mistral:7b-q4** | 4.1GB | 6GB | Médio | ⭐⭐⭐ OK |
| **llama3.1:8b-q4** | 4.7GB | 8GB | Lento | ⭐⭐ Limite Superior |

### ❌ NÃO Recomendado

- Modelos > 8B parâmetros (ficará muito lento)
- Modelos sem quantização
- Usar GPU AMD (ROCm é complexo e não vale o esforço)
- Modelos FP16/FP32 (use Q4 ou Q5)

## 🚀 Plano de Implementação Recomendado

### **Fase 1: Experimentação (Atual) - 100% Viável**

**Objetivo**: Testar diferentes LLMs e escolher o melhor

**Setup**:
```bash
# 1. Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Baixar modelos LEVES (ordem de prioridade)
ollama pull phi3:mini        # 2.3GB - Prioridade 1
ollama pull llama3.2:3b      # 2GB - Prioridade 2
ollama pull gemma2:2b        # 1.6GB - Prioridade 3

# 3. Testar
python src/experiments/run_experiments.py --mode quick
```

**O que você consegue fazer**:
- ✅ Comparar OpenAI vs modelos locais pequenos
- ✅ Avaliar qualidade vs custo
- ✅ Benchmark com 3-5 modelos simultaneamente
- ✅ MLflow tracking de todos os experimentos

**Tempo estimado**: 1-2 horas de setup + experimentação

**Custo**: $0 (modelos locais) + ~$0.10 (testes OpenAI)

---

### **Fase 2: Fine-Tuning (Futuro) - Parcialmente Viável**

**Restrições do seu hardware**:
- ❌ Fine-tuning completo de 7B+ = NÃO VIÁVEL (RAM insuficiente)
- ⚠️ LoRA/QLoRA de 3B = VIÁVEL mas lento (CPU only)
- ✅ Fine-tuning via cloud = VIÁVEL (Google Colab, AWS, Azure)

**Recomendações**:

#### Opção A: Fine-Tuning Local (3B apenas)
```python
# Use Phi-3 Mini (3.8B) com LoRA
# Configuração otimizada para sua RAM
from peft import LoraConfig

config = LoraConfig(
    r=8,              # Rank baixo (menos memória)
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Treinar em batch pequeno
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Batch size mínimo
    gradient_accumulation_steps=4,   # Simular batch maior
    fp16=False,                      # CPU não tem FP16
    max_steps=100,                   # Poucas iterações
)
```

**Tempo**: 4-8 horas para 100 steps (muito lento)
**Viabilidade**: 3/10 - Possível mas não prático

#### Opção B: Fine-Tuning em Cloud (RECOMENDADO)
```bash
# Google Colab (Gratuito com GPU T4)
# - 15 GB RAM
# - GPU NVIDIA T4
# - Pode treinar modelos 7B com LoRA

# Ou AWS SageMaker / Azure ML
# - Paga por hora de uso
# - GPU potentes
```

**Tempo**: 30 min - 2 horas
**Custo**: $0 (Colab) ou $1-5/hora (Cloud)
**Viabilidade**: 10/10 - Totalmente prático

---

### **Fase 3: Produção - Híbrido Recomendado**

**Arquitetura Recomendada**:

```
┌─────────────────────────────────────────┐
│         Sua Máquina (Local)             │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │  Phi-3 Mini  │  │   Ollama API    │ │
│  │   (Preview)  │  │  localhost:11434│ │
│  │   2.3 GB     │  └─────────────────┘ │
│  └──────────────┘                      │
│         ↓ (Se aprovado)                │
│  ┌──────────────┐                      │
│  │  OpenAI API  │ ← Resposta Final    │
│  │ (Produção)   │                      │
│  └──────────────┘                      │
└─────────────────────────────────────────┘
```

**Vantagens**:
- ✅ Preview rápido e gratuito (Phi-3)
- ✅ Qualidade alta no resultado final (OpenAI)
- ✅ Custo otimizado (paga só respostas aprovadas)
- ✅ Não precisa de servidor potente

**Implementação**:
```python
# src/agent/hybrid_llm.py
class HybridLLM:
    def __init__(self):
        self.preview_llm = MultiLLMManager.create_llm("ollama", "phi3:mini")
        self.production_llm = MultiLLMManager.create_llm("openai", "gpt-4o-mini")
    
    def generate(self, question, use_preview=True):
        if use_preview:
            # Preview rápido (grátis)
            preview = self.preview_llm.invoke(question)
            # Usuário pode aprovar ou regenerar
            return preview
        else:
            # Produção (pago, melhor qualidade)
            return self.production_llm.invoke(question)
```

---

## 💡 Recomendação Final por Caso de Uso

### 1. **Desenvolvimento e Testes** (Agora)
```bash
# Setup mínimo
ollama pull phi3:mini
python src/experiments/run_experiments.py --mode quick
```
**Viabilidade**: ⭐⭐⭐⭐⭐ (100%)

### 2. **Experimentação e Benchmark** (Semana 1-2)
```bash
# Testar 3-4 modelos locais
ollama pull phi3:mini
ollama pull llama3.2:3b
ollama pull gemma2:2b

# Comparar com OpenAI
python src/experiments/run_experiments.py --mode full
```
**Viabilidade**: ⭐⭐⭐⭐⭐ (100%)
**Custo**: ~$0.50 (OpenAI) + $0 (locais)

### 3. **Fine-Tuning de Modelos** (Futuro)
```bash
# Use Google Colab (gratuito)
# Notebook: notebooks/fine_tuning_colab.ipynb
# - Upload dataset
# - Fine-tune Llama 3.2 3B ou Mistral 7B
# - Download modelo
# - Carregar no Ollama local
```
**Viabilidade**: ⭐⭐⭐⭐ (80% - depende de cloud)
**Custo**: $0 (Colab) ou $5-20 (se usar cloud pago)

### 4. **Produção** (Quando estável)

**Opção A: Híbrido (RECOMENDADO para você)**
- Local: Preview/desenvolvimento (Phi-3)
- Cloud: Produção/respostas finais (OpenAI)
- **Viabilidade**: ⭐⭐⭐⭐⭐ (100%)
- **Custo**: ~$10-50/mês (depende do volume)

**Opção B: 100% Local**
- Apenas modelos pequenos (Phi-3, Llama 3.2 3B)
- Qualidade inferior mas zero custo
- **Viabilidade**: ⭐⭐⭐ (60% - qualidade limitada)
- **Custo**: $0

**Opção C: 100% Cloud**
- OpenAI para tudo
- Melhor qualidade
- **Viabilidade**: ⭐⭐⭐⭐⭐ (100%)
- **Custo**: $50-200/mês (depende do volume)

---

## 🎯 Próximos Passos Práticos (Ordem Recomendada)

### Semana 1: Setup e Experimentação
```bash
# Dia 1-2: Setup básico
1. Instalar Ollama
2. Baixar phi3:mini (mais leve e rápido)
3. Testar manualmente

# Dia 3-4: Benchmark
4. Executar run_experiments.py --mode quick
5. Comparar Phi-3 vs OpenAI
6. Analisar resultados no MLflow

# Dia 5: Decisão
7. Escolher: modelo local OU OpenAI OU híbrido
8. Documentar decisão e métricas
```

### Semana 2: Otimização
```bash
# Se escolheu local:
1. Testar llama3.2:3b e gemma2:2b
2. Otimizar prompts para modelo escolhido
3. Configurar cache de respostas

# Se escolheu híbrido:
4. Implementar sistema de preview+produção
5. Definir regras de quando usar cada um

# Se escolheu OpenAI:
6. Otimizar custos (cache, batch)
7. Monitorar usage e custos
```

### Mês 1-2: Curadoria de Dados
```bash
# Preparar para fine-tuning futuro
1. Coletar perguntas reais de usuários
2. Revisar e melhorar respostas
3. Criar dataset de 500-1000 pares Q&A
4. Validar qualidade dos dados
```

### Mês 2-3: Fine-Tuning (Opcional)
```bash
# Se decidir fine-tunar:
1. Setup Google Colab com GPU
2. Fine-tune Llama 3.2 3B ou Phi-3
3. Avaliar modelo customizado
4. Comparar com base model
5. Deploy se melhorar métricas
```

---

## 📈 Benchmark Esperado no Seu Hardware

Com base nas especificações:

| Modelo | Latência | Qualidade | Custo | Recomendação |
|--------|----------|-----------|-------|--------------|
| Phi-3 Mini | ~2-3s | 7/10 | $0 | ⭐⭐⭐⭐⭐ Use para dev |
| Llama 3.2 3B | ~2-4s | 7.5/10 | $0 | ⭐⭐⭐⭐ Alternativa |
| Gemma 2B | ~1-2s | 6.5/10 | $0 | ⭐⭐⭐ Rápido mas básico |
| Mistral 7B-Q4 | ~5-8s | 8/10 | $0 | ⭐⭐ Lento demais |
| GPT-4o-mini | ~1s | 9.5/10 | $0.0001/resp | ⭐⭐⭐⭐⭐ Produção |

**Veredicto**: 
- **Desenvolvimento**: Phi-3 Mini local
- **Produção**: GPT-4o-mini (OpenAI) ou híbrido
- **Fine-tuning**: Google Colab com GPU

---

## 🔧 Comandos Otimizados para Seu Sistema

### Setup Inicial Otimizado
```bash
# 1. Instalar apenas o essencial
pip install mlflow langchain-ollama transformers --no-cache-dir

# 2. Ollama com modelo leve
ollama pull phi3:mini

# 3. Configurar para usar CPU eficientemente
export OLLAMA_NUM_PARALLEL=2  # Apenas 2 requests paralelos
export OLLAMA_MAX_LOADED_MODELS=1  # Apenas 1 modelo na RAM
```

### Testar Performance
```bash
# Benchmark apenas modelos viáveis
python << EOF
from src.experiments.multi_llm import MultiLLMManager
import time

# Teste de latência
llm = MultiLLMManager.create_llm("ollama", "phi3:mini")

start = time.time()
response = llm.invoke("O que é Retail Media?")
latency = time.time() - start

print(f"Latência: {latency:.2f}s")
print(f"Resposta: {response[:100]}...")
EOF
```

---

## 💰 Análise de Custos

### Cenário 1: 100% Local (Phi-3)
- Setup: $0
- Mensal: $0
- Energia: ~$2-5/mês
- **Total**: $2-5/mês

### Cenário 2: Híbrido (Phi-3 + OpenAI)
- Setup: $0
- Preview (local): $0
- Produção (1000 resp/mês): ~$15
- **Total**: $15-20/mês

### Cenário 3: 100% OpenAI
- Setup: $0
- 1000 respostas/mês: ~$15
- 5000 respostas/mês: ~$75
- **Total**: $15-75/mês

**Recomendação**: Cenário 2 (Híbrido) - melhor custo-benefício

---

## ✅ Checklist de Viabilidade

- ✅ Experimentação com 3-4 LLMs locais
- ✅ Benchmark automatizado
- ✅ MLflow tracking
- ✅ Jupyter notebooks interativos
- ✅ Comparação com OpenAI
- ⚠️ Fine-tuning local (3B apenas, lento)
- ✅ Fine-tuning em cloud (Colab/AWS)
- ✅ Deploy híbrido (local + cloud)
- ❌ Modelos grandes (13B+)
- ❌ GPU training (sem CUDA)

**Score de Viabilidade**: 8/10 ⭐⭐⭐⭐

Você pode fazer **quase tudo**, exceto rodar modelos grandes localmente. A solução híbrida é perfeita para suas necessidades!


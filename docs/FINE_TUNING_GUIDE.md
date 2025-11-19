# 🎯 Guia Completo de Fine-Tuning com QLoRA

Este guia ensina como fazer fine-tuning de LLMs usando QLoRA no Google Colab (GPU gratuita).

## 📋 Índice

1. [O que é QLoRA?](#o-que-é-qlora)
2. [Quando fazer fine-tuning?](#quando-fazer-fine-tuning)
3. [Preparação de dados](#preparação-de-dados)
4. [Fine-tuning no Colab](#fine-tuning-no-colab)
5. [Usando o modelo treinado](#usando-o-modelo-treinado)
6. [Troubleshooting](#troubleshooting)
7. [Melhores práticas](#melhores-práticas)

---

## O que é QLoRA?

**QLoRA (Quantized Low-Rank Adaptation)** é uma técnica eficiente de fine-tuning que permite treinar modelos LLM grandes com memória GPU limitada.

### Como funciona:

```
┌──────────────────────────────────────────┐
│  Modelo Base (7B params)                 │
│  Quantizado em 4-bit (~4 GB)             │
└──────────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────┐
│  + LoRA Adapters (~50-200 MB)            │
│  Apenas 0.1-1% dos parâmetros treináveis │
└──────────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────┐
│  Modelo Fine-Tunado                      │
│  Especializado no seu domínio            │
└──────────────────────────────────────────┘
```

### Vantagens:

- ✅ **Eficiente**: Treina em GPU T4 gratuita (Colab)
- ✅ **Rápido**: 30 min - 2 horas de treino
- ✅ **Leve**: Apenas adaptadores (~50-200 MB) vs modelo completo (4-14 GB)
- ✅ **Qualidade**: Resultados comparáveis a fine-tuning completo

### Paper Original:

[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)

---

## Quando fazer fine-tuning?

### ✅ Faça fine-tuning quando:

1. **Domínio específico**: Seu caso usa terminologia/conceitos únicos
   - Ex: Retail Media, jurídico, médico, técnico

2. **Estilo específico**: Precisa de tom/formato particular
   - Ex: Formal, casual, técnico, didático

3. **Performance insuficiente**: Modelos base não atendem qualidade
   - Respostas genéricas, imprecisas, ou irrelevantes

4. **Dados proprietários**: Você tem dados únicos valiosos
   - Logs de interações, documentação interna, Q&A curado

5. **Custo**: Quer reduzir custos de API
   - Substituir GPT-4 por modelo local customizado

### ❌ NÃO faça fine-tuning quando:

1. **Falta de dados**: Menos de 50-100 exemplos de qualidade
2. **RAG resolve**: Problema é falta de contexto (use RAG primeiro)
3. **Prompts funcionam**: Engenharia de prompt já resolve
4. **Tempo/recursos limitados**: Fine-tuning requer iteração

### 🎯 Abordagem Recomendada:

```
1. Prompt Engineering
   ↓ (não resolveu?)
2. RAG (Retrieval)
   ↓ (ainda insuficiente?)
3. Fine-Tuning
```

---

## Preparação de Dados

### 1. Coleta de Dados

Você precisa de **pares pergunta-resposta** de qualidade.

#### Fontes de dados:

- **Logs do sistema RAG**: Interações reais dos usuários
- **Documentação**: Convertida em Q&A
- **FAQs**: Perguntas frequentes curadas
- **Especialistas**: Respostas revisadas por humanos

#### Quantidade mínima:

| Dataset | Mínimo | Recomendado | Ideal |
|---------|--------|-------------|-------|
| **Testes** | 10-20 | 50 | 100 |
| **Produção** | 100-200 | 500-1000 | 2000+ |

### 2. Formato dos Dados

QLoRA usa templates específicos. Os mais comuns:

#### Formato Alpaca (Recomendado):

```text
### Instruction:
O que é Retail Media?

### Response:
Retail Media é uma forma de publicidade digital onde varejistas monetizam...
```

#### Formato ChatML (Phi-3, GPT):

```text
<|system|>
Você é um assistente especializado em Retail Media.
<|end|>
<|user|>
O que é Retail Media?
<|end|>
<|assistant|>
Retail Media é uma forma de publicidade...
<|end|>
```

#### Formato Llama 2:

```text
[INST] <<SYS>>
Você é um assistente especializado em Retail Media.
<</SYS>>

O que é Retail Media? [/INST] Retail Media é...
```

### 3. Preparar Dataset (Local)

Use o notebook: `notebooks/prepare_finetuning_data.ipynb`

```python
from src.finetuning.data_prep import DatasetPreparator

# Criar preparator
preparator = DatasetPreparator(template="alpaca")

# Opção 1: De JSON
dataset = preparator.prepare_from_json("data.json")

# Opção 2: De CSV
dataset = preparator.prepare_from_csv("data.csv")

# Opção 3: De logs RAG
dataset = preparator.prepare_from_rag_logs("logs/", min_quality_score=4.0)

# Dividir treino/validação
split_dataset = preparator.create_train_test_split(dataset, test_size=0.1)

# Salvar
preparator.save_dataset(split_dataset, "data/finetuning/processed/my_dataset")
```

### 4. Qualidade dos Dados

**CRITICAL**: Qualidade > Quantidade

✅ **Boas práticas:**

- ✅ Respostas completas e precisas
- ✅ Diversidade de perguntas
- ✅ Cobertura de diferentes tópicos
- ✅ Revisar manualmente alguns exemplos
- ✅ Remover exemplos ruins/ambíguos

❌ **Evite:**

- ❌ Respostas genéricas/óbvias
- ❌ Erros factuais
- ❌ Viés ou linguagem problemática
- ❌ Duplicatas
- ❌ Respostas muito curtas (<50 chars) ou longas (>2000 chars)

---

## Fine-Tuning no Colab

### 1. Setup Google Colab

1. **Acessar**: [Google Colab](https://colab.research.google.com/)
2. **Ativar GPU**: 
   - Runtime → Change runtime type
   - Hardware accelerator → **T4 GPU**
3. **Upload notebook**: `notebooks/fine_tuning_qlora_colab.ipynb`

### 2. Fazer Upload do Dataset

**Opção A: Google Drive (Recomendado)**

```python
# No Colab
from google.colab import drive
drive.mount('/content/drive')

# Dataset em: /content/drive/MyDrive/training_dataset
```

**Opção B: Upload direto**

```python
from google.colab import files
uploaded = files.upload()
```

### 3. Escolher Modelo Base

Modelos recomendados para Colab (GPU T4):

| Modelo | Tamanho | Velocidade | Qualidade | RAM GPU |
|--------|---------|------------|-----------|---------|
| **microsoft/phi-2** | 2.7B | ⚡⚡⚡ Rápido | ⭐⭐⭐ Boa | ~8 GB |
| **meta-llama/Llama-2-7b-hf** | 7B | ⚡⚡ Médio | ⭐⭐⭐⭐ Ótima | ~12 GB |
| **mistralai/Mistral-7B-v0.1** | 7B | ⚡⚡ Médio | ⭐⭐⭐⭐⭐ Excelente | ~12 GB |

**Recomendação:** Comece com **Phi-2** para testes rápidos, depois use **Mistral 7B** para produção.

### 4. Configurar Parâmetros

```python
# No notebook Colab, ajuste:

MODEL_NAME = "microsoft/phi-2"
DATASET_PATH = "/content/drive/MyDrive/training_dataset"
OUTPUT_MODEL_NAME = "phi2-retail-media"

# Hiperparâmetros
EPOCHS = 3              # Mais epochs = mais aprendizado (cuidado com overfitting)
BATCH_SIZE = 4          # Reduza se ficar sem memória
LEARNING_RATE = 2e-4    # Taxa de aprendizado (geralmente 1e-4 a 5e-4)
MAX_SEQ_LENGTH = 2048   # Tamanho máximo de contexto

# LoRA
LORA_R = 16             # Rank (8-64, maior = mais capacidade)
LORA_ALPHA = 32         # Geralmente 2x o rank
```

### 5. Executar Treinamento

Execute as células do notebook sequencialmente:

1. ✅ Verificar GPU
2. ✅ Instalar dependências
3. ✅ Carregar dataset
4. ✅ Carregar modelo com quantização
5. ✅ Aplicar LoRA
6. ✅ Tokenizar dataset
7. ✅ **Treinar** ⏱️ (30 min - 2 horas)
8. ✅ Salvar modelo
9. ✅ Testar

### 6. Monitorar Treinamento

Durante o treino, observe:

- **Loss**: Deve diminuir gradualmente
  - Inicial: ~2-3
  - Final: ~0.5-1.5
  - Se não diminui: aumente learning rate
  - Se diminui muito rápido: reduza learning rate

- **GPU Memory**: 
  - Uso: ~80-90% é ideal
  - Se 100%: reduza BATCH_SIZE ou MAX_SEQ_LENGTH
  - Se <50%: pode aumentar BATCH_SIZE

- **Tempo por step**:
  - Phi-2: ~1-2s/step
  - Llama/Mistral 7B: ~3-5s/step

### 7. Salvar e Download

Modelo salvo automaticamente em:
- **Local**: `/content/models/{OUTPUT_MODEL_NAME}/`
- **Google Drive**: `/content/drive/MyDrive/finetuned_models/{OUTPUT_MODEL_NAME}/`
- **ZIP**: Para download direto

---

## Usando o Modelo Treinado

### Opção 1: Testar no Colab

Já incluído no notebook:

```python
def generate_response(prompt):
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Testar
response = generate_response("O que é Retail Media?")
print(response)
```

### Opção 2: Usar Localmente com Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Carregar modelo base
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Carregar adaptadores LoRA
model = PeftModel.from_pretrained(base_model, "./phi2-retail-media")

# Carregar tokenizer
tokenizer = AutoTokenizer.from_pretrained("./phi2-retail-media")

# Usar
def ask(question):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = ask("O que é programmatic advertising?")
print(response)
```

### Opção 3: Converter para Ollama (Recomendado)

Para usar localmente sem GPU potente:

#### Passo 1: Merge LoRA com modelo base

```python
from src.finetuning.export_to_ollama import ModelExporter

exporter = ModelExporter("./phi2-retail-media")
merged_path = exporter.merge_lora_adapters(
    base_model_name="microsoft/phi-2",
    output_path="./phi2-retail-media-merged"
)
```

#### Passo 2: Converter para GGUF (opcional)

Requer [llama.cpp](https://github.com/ggerganov/llama.cpp) instalado:

```bash
# Instalar llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Converter
python convert.py /path/to/phi2-retail-media-merged --outfile phi2-retail.fp16.gguf
./quantize phi2-retail.fp16.gguf phi2-retail.q4_k_m.gguf q4_k_m
```

#### Passo 3: Criar Modelfile Ollama

```python
from src.finetuning.export_to_ollama import ModelExporter

exporter.create_ollama_modelfile(
    gguf_path="./phi2-retail.q4_k_m.gguf",
    output_path="./Modelfile",
    model_name="phi2-retail"
)
```

#### Passo 4: Usar com Ollama

```bash
# Criar modelo no Ollama
ollama create phi2-retail -f Modelfile

# Usar
ollama run phi2-retail "O que é Retail Media?"
```

#### Passo 5: Integrar no seu código

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="phi2-retail", temperature=0.7)
response = llm.invoke("O que é programmatic advertising?")
print(response.content)
```

---

## Troubleshooting

### Problema: Out of Memory (OOM)

**Sintoma**: `CUDA out of memory` durante treinamento

**Soluções**:

1. **Reduzir batch size**:
   ```python
   BATCH_SIZE = 2  # ou 1
   GRADIENT_ACCUMULATION = 8  # compensar batch menor
   ```

2. **Reduzir sequence length**:
   ```python
   MAX_SEQ_LENGTH = 1024  # ao invés de 2048
   ```

3. **Reduzir LoRA rank**:
   ```python
   LORA_R = 8  # ao invés de 16
   ```

4. **Usar modelo menor**:
   ```python
   MODEL_NAME = "microsoft/phi-2"  # ao invés de Llama 7B
   ```

### Problema: Loss não diminui

**Sintoma**: Loss fica estável ou aumenta

**Soluções**:

1. **Aumentar learning rate**:
   ```python
   LEARNING_RATE = 5e-4  # ao invés de 2e-4
   ```

2. **Verificar dados**:
   - Dataset está no formato correto?
   - Exemplos fazem sentido?

3. **Treinar mais**:
   ```python
   EPOCHS = 5  # ao invés de 3
   ```

### Problema: Overfitting

**Sintoma**: Loss de treino baixa, mas modelo não generaliza

**Soluções**:

1. **Mais dados**: Adicionar mais exemplos diversos

2. **Menos epochs**:
   ```python
   EPOCHS = 2
   ```

3. **Aumentar dropout**:
   ```python
   LORA_DROPOUT = 0.1  # ao invés de 0.05
   ```

4. **Validação regular**: Sempre use split treino/validação

### Problema: Modelo não segue instruções

**Sintoma**: Respostas não fazem sentido ou ignoram pergunta

**Soluções**:

1. **Verificar template**: Usar formato correto (Alpaca, ChatML, etc)

2. **Exemplos de qualidade**: Revisar se respostas no dataset são boas

3. **Mais dados**: Adicionar mais exemplos variados

4. **Testar modelo base**: Verificar se modelo base já segue instruções

### Problema: Colab desconecta

**Sintoma**: Sessão Colab desconecta durante treino longo

**Soluções**:

1. **Salvar checkpoints**:
   ```python
   save_steps = 50  # salva a cada 50 steps
   ```

2. **Backup no Drive**: Já configurado no notebook

3. **Manter aba ativa**: Deixar aba do Colab aberta

4. **Colab Pro**: Considerar upgrade ($10/mês)

---

## Melhores Práticas

### 1. Curadoria de Dados

- ✅ **Qualidade primeiro**: 100 exemplos excelentes > 1000 medianos
- ✅ **Diversidade**: Cubra diferentes tipos de perguntas
- ✅ **Revisão humana**: Sempre revisar alguns exemplos
- ✅ **Limpeza**: Remover duplicatas, erros, exemplos ruins
- ✅ **Validação**: Sempre usar split treino/validação (90/10)

### 2. Escolha de Modelo

| Use Case | Modelo Recomendado | Justificativa |
|----------|-------------------|---------------|
| **Testes rápidos** | Phi-2 (2.7B) | Rápido, roda em qualquer GPU |
| **Produção balanceada** | Mistral 7B | Melhor custo-benefício |
| **Máxima qualidade** | Llama 2 7B | Excelente, bem suportado |
| **Muito específico** | Qualquer + mais dados | Dataset é mais importante |

### 3. Hiperparâmetros

**Configuração conservadora (recomendada)**:

```python
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

**Para dataset pequeno (<200 exemplos)**:

```python
EPOCHS = 5  # Treinar mais
LORA_R = 8  # Rank menor (evitar overfit)
```

**Para dataset grande (>1000 exemplos)**:

```python
EPOCHS = 2  # Menos epochs
LORA_R = 32  # Rank maior (mais capacidade)
```

### 4. Avaliação

Sempre avaliar em múltiplas dimensões:

1. **Loss quantitativo**: Training/validation loss
2. **Testes qualitativos**: Perguntas específicas do domínio
3. **Comparação**: vs modelo base, vs OpenAI
4. **Usuários reais**: Beta testing com usuários

### 5. Iteração

Fine-tuning é um processo iterativo:

```
1. Baseline (modelo base)
   ↓
2. Fine-tune v1 (dataset inicial)
   ↓
3. Avaliar e coletar feedback
   ↓
4. Melhorar dataset
   ↓
5. Fine-tune v2
   ↓
6. Repetir até satisfatório
```

### 6. Deployment

**Opção A: Local (CPU)**
- ✅ Zero custo
- ✅ Privacidade total
- ❌ Latência alta (2-5s)
- **Ideal para**: Desenvolvimento, baixo volume

**Opção B: Local (GPU)**
- ✅ Latência baixa (<1s)
- ✅ Privacidade
- ❌ Requer GPU NVIDIA
- **Ideal para**: Produção, alto volume

**Opção C: Cloud (RunPod, AWS)**
- ✅ Escalável
- ✅ GPU potentes
- ❌ Custo mensal
- **Ideal para**: Produção, muitos usuários

**Opção D: Híbrido**
- Modelo local para preview
- OpenAI para respostas finais
- **Ideal para**: Otimizar custos

---

## Recursos Adicionais

### Papers

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

### Documentação

- [PEFT (HuggingFace)](https://huggingface.co/docs/peft)
- [Transformers](https://huggingface.co/docs/transformers)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [TRL](https://huggingface.co/docs/trl)

### Ferramentas

- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Framework de fine-tuning
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) - UI para fine-tuning
- [Unsloth](https://github.com/unslothai/unsloth) - Fine-tuning otimizado

### Comunidades

- [HuggingFace Discord](https://hf.co/join/discord)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)

---

## FAQ

### Quanto custa fazer fine-tuning?

- **Google Colab (gratuito)**: $0
- **Google Colab Pro**: $10/mês (GPU melhor, mais tempo)
- **RunPod**: $0.30-0.80/hora (pay-as-you-go)
- **AWS SageMaker**: $1-5/hora

Para 1 fine-tuning: ~$0-2

### Quanto tempo leva?

- **Phi-2 (2.7B)**: 20-40 min (100 exemplos, 3 epochs)
- **Llama/Mistral 7B**: 1-2 horas (100 exemplos, 3 epochs)

### Quantos dados preciso?

- **Mínimo viável**: 50-100 exemplos
- **Recomendado**: 500-1000 exemplos
- **Ideal**: 2000+ exemplos

### Posso fine-tunar GPT-4?

Não. GPT-4 é closed-source. Mas você pode:
- Fine-tunar GPT-3.5 Turbo via OpenAI API
- Fine-tunar modelos open-source (Llama, Mistral, Phi)

### Fine-tuning é melhor que RAG?

**Não é "ou"**, é **"e"**:

- **RAG**: Adiciona conhecimento/contexto
- **Fine-tuning**: Melhora estilo/comportamento

**Melhor**: RAG + Fine-tuning juntos

### Posso comercializar modelo fine-tunado?

Depende da licença do modelo base:

- ✅ **Llama 2**: Sim (comercial até 700M users)
- ✅ **Mistral**: Sim (Apache 2.0)
- ✅ **Phi-2**: Sim (MIT license)
- ❌ **Llama 1**: Não (apenas pesquisa)

Sempre verifique a licença!

---

## Próximos Passos

1. ✅ Preparar seu dataset (`prepare_finetuning_data.ipynb`)
2. ✅ Fazer fine-tuning no Colab (`fine_tuning_qlora_colab.ipynb`)
3. ✅ Testar e avaliar modelo
4. ✅ Iterar e melhorar
5. ✅ Deploy em produção

**Boa sorte com seu fine-tuning! 🚀**

---

**Dúvidas?** Abra uma issue no repositório ou consulte a documentação.


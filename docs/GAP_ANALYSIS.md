# 🎯 Análise de Gap - MelancIA Fine-Tuning & MLOps

**Data**: 19 de Novembro de 2025  
**Status**: 📊 Análise Completa de Requisitos vs Implementação Atual

---

## 📋 SUMÁRIO EXECUTIVO

### ✅ O que JÁ está implementado (Score: 45%)

| Categoria | Status | Nota |
|-----------|--------|------|
| **RAG Base** | ✅ Completo | 100% - RAG funcional com ChromaDB |
| **MLflow Tracking** | ✅ Completo | 100% - Tracking de experimentos |
| **Multi-LLM Support** | ✅ Completo | 100% - OpenAI, Ollama, HuggingFace |
| **Model Router** | ✅ Completo | 90% - Roteamento inteligente + A/B test |
| **Pipeline ETL** | ✅ Completo | 100% - Scraping + curadoria de dados |
| **Docker** | ✅ Parcial | 70% - Docker Compose básico |
| **Fine-Tuning** | ❌ Não iniciado | 0% - **CRÍTICO** |
| **Evaluation Loops** | ❌ Não iniciado | 0% - **CRÍTICO** |
| **CI/CD** | ❌ Não iniciado | 0% - **IMPORTANTE** |
| **Monitoramento Produção** | ❌ Não iniciado | 0% - **IMPORTANTE** |
| **API REST (FastAPI)** | ❌ Não iniciado | 0% - **CRÍTICO** |
| **Testes Automatizados** | ❌ Não iniciado | 0% - **IMPORTANTE** |

### 🎯 Prioridades

**🔴 CRÍTICO (Sem isso, não atende o escopo)**
1. Fine-Tuning de LLMs (LoRA/QLoRA)
2. API REST com FastAPI
3. Dataset proprietário para fine-tuning
4. Evaluation loops automatizados

**🟡 IMPORTANTE (MLOps completo)**
5. CI/CD Pipeline
6. Testes automatizados
7. Monitoramento em produção
8. Model Registry completo
9. Drift Detection

**🟢 DESEJÁVEL (Polimento)**
10. Integração Vertex AI / Azure ML
11. Data versioning (DVC)
12. Model Cards formais
13. Dashboard de analytics

---

## 🔴 1. GAPS CRÍTICOS (Bloqueadores)

### 1.1 Fine-Tuning de LLMs ❌

**Status**: Não implementado  
**Impacto**: CRÍTICO - É o core do escopo

**O que falta:**

#### A) Infraestrutura de Fine-Tuning
```python
# src/finetuning/
├── trainer.py              # Classe principal de treinamento
├── data_preparation.py     # Preparação de datasets
├── lora_config.py          # Configurações LoRA/QLoRA
├── evaluation.py           # Avaliação pós-treinamento
└── model_loader.py         # Loading de modelos base
```

#### B) Scripts de Treinamento
- ❌ Script de fine-tuning com LoRA/QLoRA
- ❌ Suporte a PEFT (Parameter-Efficient Fine-Tuning)
- ❌ Quantização (4-bit/8-bit) para economia de memória
- ❌ Distributed training (se multi-GPU)
- ❌ Gradient accumulation
- ❌ Mixed precision training (fp16/bf16)

#### C) Dataset para Fine-Tuning
- ❌ Dataset proprietário de Retail Media Ads
- ❌ Formato de instrução (instruction-following)
- ❌ Validação e limpeza de dados
- ❌ Train/Val/Test split estratificado
- ❌ Data augmentation para e-commerce

**Exemplo de dataset necessário:**
```json
{
  "instruction": "Explique o que é ACOS no contexto de Retail Media",
  "input": "",
  "output": "ACOS (Advertising Cost of Sale) é a métrica que mede...",
  "context": "retail_media_metrics",
  "metadata": {
    "source": "conecta_ads_blog",
    "difficulty": "medium",
    "category": "metricas"
  }
}
```

#### D) Técnicas de Fine-Tuning
Implementar:
- ✅ Base já tem: PyTorch, Transformers, PEFT
- ❌ LoRA (Low-Rank Adaptation)
- ❌ QLoRA (Quantized LoRA)
- ❌ Prefix Tuning
- ❌ Prompt Tuning
- ❌ Adapter Layers

#### E) Integração com MLflow
- ❌ Log de hiperparâmetros de treinamento
- ❌ Log de métricas durante treinamento (loss, accuracy)
- ❌ Salvamento de checkpoints
- ❌ Registro de modelos fine-tunados
- ❌ Comparação de runs de fine-tuning

**Exemplo de implementação mínima:**
```python
# src/finetuning/trainer.py (PRECISA SER CRIADO)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import mlflow

class LLMFineTuner:
    """Fine-tuning de LLMs com LoRA/QLoRA"""
    
    def __init__(self, base_model_name: str, output_dir: str):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        
    def prepare_model(self, use_4bit: bool = True):
        """Carrega modelo base com quantização"""
        # Implementar loading com bitsandbytes
        pass
    
    def setup_lora(self, r: int = 8, alpha: int = 16):
        """Configura LoRA"""
        lora_config = LoraConfig(
            r=r,  # Rank
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        return lora_config
    
    def train(self, train_dataset, val_dataset, epochs: int = 3):
        """Executa fine-tuning"""
        # Implementar loop de treinamento
        with mlflow.start_run():
            mlflow.log_params({
                "base_model": self.base_model_name,
                "epochs": epochs,
                "lora_r": 8
            })
            
            # Training loop
            for epoch in range(epochs):
                # ...
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, step=epoch)
```

---

### 1.2 API REST com FastAPI ❌

**Status**: FastAPI está nos requirements, mas não implementado  
**Impacto**: CRÍTICO - Necessário para produção

**O que falta:**

#### A) Estrutura da API
```python
# src/api/
├── main.py              # FastAPI app principal
├── routes/
│   ├── query.py        # Endpoints de queries
│   ├── models.py       # Endpoints de modelos
│   ├── health.py       # Health checks
│   └── admin.py        # Admin/management
├── schemas/
│   ├── request.py      # Pydantic models request
│   └── response.py     # Pydantic models response
├── middleware/
│   ├── auth.py         # Autenticação
│   ├── rate_limit.py   # Rate limiting
│   └── logging.py      # Request logging
└── dependencies.py      # FastAPI dependencies
```

#### B) Endpoints Necessários

**Queries (Core)**
```python
POST /api/v1/query
POST /api/v1/query/stream    # Streaming responses
GET  /api/v1/query/{query_id}
```

**Modelos**
```python
GET  /api/v1/models          # Listar modelos disponíveis
POST /api/v1/models/select   # Selecionar modelo
GET  /api/v1/models/status   # Status dos modelos
```

**Health & Monitoring**
```python
GET  /health                 # Health check
GET  /metrics                # Prometheus metrics
GET  /api/v1/stats           # Estatísticas de uso
```

**Admin**
```python
POST /api/v1/finetune/start  # Iniciar fine-tuning
GET  /api/v1/finetune/status # Status do treinamento
POST /api/v1/cache/clear     # Limpar cache
```

#### C) Exemplo de Implementação Mínima
```python
# src/api/main.py (PRECISA SER CRIADO)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from src.mlops.model_router import ModelRouter

app = FastAPI(
    title="MelancIA API",
    version="1.0.0",
    description="API de IA para Retail Media Ads"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global router
router = ModelRouter()

# Schemas
class QueryRequest(BaseModel):
    question: str
    user_tier: str = "free"
    user_id: int = None
    max_tokens: int = 500

class QueryResponse(BaseModel):
    answer: str
    provider: str
    model_name: str
    latency_seconds: float
    success: bool

# Routes
@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Processa query do usuário"""
    try:
        result = router.route_query(
            question=request.question,
            user_tier=request.user_tier,
            user_id=request.user_id
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "stats": router.get_stats()
    }

@app.get("/api/v1/models")
async def list_models():
    """Lista modelos disponíveis"""
    return {
        "models": [
            {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "status": "available"
            },
            {
                "provider": "ollama",
                "name": "phi3:mini",
                "status": "available"
            }
        ]
    }
```

#### D) Autenticação e Segurança
- ❌ JWT tokens
- ❌ API keys
- ❌ Rate limiting (por usuário/IP)
- ❌ HTTPS/TLS
- ❌ Request validation
- ❌ CORS configurado

#### E) Documentação API
- ❌ OpenAPI/Swagger auto-gerado (FastAPI já faz)
- ❌ Exemplos de uso
- ❌ Postman collection
- ❌ Client SDKs (Python, JavaScript)

---

### 1.3 Dataset Proprietário para Fine-Tuning ❌

**Status**: Tem scraping de blog, mas não formatado para fine-tuning  
**Impacto**: CRÍTICO - Sem dados, não tem fine-tuning

**O que falta:**

#### A) Estrutura de Dados
```python
# data/datasets/
├── raw/
│   └── conecta_blog_raw.json       # Scraping bruto
├── processed/
│   ├── train.jsonl                 # 80% treino
│   ├── val.jsonl                   # 10% validação
│   └── test.jsonl                  # 10% teste
├── synthetic/
│   └── augmented_data.jsonl        # Dados sintéticos
└── metadata/
    └── dataset_info.json            # Metadados do dataset
```

#### B) Formato de Instrução
Converter blog posts em pares instrução-resposta:

```json
// Exemplo de registro no formato de fine-tuning
{
  "messages": [
    {
      "role": "system",
      "content": "Você é a MelancIA, especialista em Retail Media Ads..."
    },
    {
      "role": "user",
      "content": "Como calcular o ROAS de uma campanha no Mercado Livre?"
    },
    {
      "role": "assistant",
      "content": "O ROAS (Return on Ad Spend) é calculado dividindo..."
    }
  ],
  "metadata": {
    "source": "conecta_blog",
    "category": "metricas",
    "difficulty": "intermediate"
  }
}
```

#### C) Pipeline de Preparação de Dados
```python
# src/finetuning/data_preparation.py (PRECISA SER CRIADO)
from datasets import Dataset, load_dataset
import json
from typing import List, Dict

class RetailMediaDatasetBuilder:
    """Constrói dataset de fine-tuning a partir do blog"""
    
    def __init__(self, raw_data_path: str):
        self.raw_data = self.load_raw_data(raw_data_path)
        
    def convert_blog_to_qa(self) -> List[Dict]:
        """Converte posts do blog em QA pairs"""
        # Implementar:
        # 1. Extrair perguntas dos títulos/subtítulos
        # 2. Gerar respostas do conteúdo
        # 3. Validar qualidade
        pass
    
    def generate_synthetic_data(self) -> List[Dict]:
        """Gera dados sintéticos com GPT-4"""
        # Usar OpenAI para gerar variações
        pass
    
    def validate_and_clean(self, dataset: List[Dict]) -> List[Dict]:
        """Valida e limpa dataset"""
        # Remover duplicatas, validar formato, etc
        pass
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1):
        """Divide em train/val/test"""
        pass
```

#### D) Tipos de Dados Necessários
1. **Conceitos** (O que é X?)
   - "O que é Retail Media?"
   - "Explique ACOS"
   
2. **Como Fazer** (Tutoriais)
   - "Como otimizar campanhas no ML?"
   - "Passo a passo para criar anúncios"
   
3. **Comparações** (X vs Y)
   - "ACOS vs ROAS: qual usar?"
   - "Mercado Livre vs Shopee"
   
4. **Troubleshooting** (Problemas)
   - "Meu ACOS está alto, o que fazer?"
   - "Como reduzir CPC?"
   
5. **Estratégias** (Conselhos)
   - "Melhores práticas para Black Friday"
   - "Como aumentar conversão"

#### E) Validação de Qualidade
```python
# src/finetuning/data_validation.py (PRECISA SER CRIADO)
class DatasetValidator:
    """Valida qualidade do dataset"""
    
    def check_diversity(self, dataset):
        """Verifica diversidade de exemplos"""
        pass
    
    def check_length_distribution(self, dataset):
        """Analisa distribuição de tamanhos"""
        pass
    
    def check_balance(self, dataset):
        """Verifica balanceamento de categorias"""
        pass
    
    def detect_duplicates(self, dataset):
        """Detecta duplicatas/near-duplicates"""
        pass
```

---

### 1.4 Evaluation Loops Automatizados ❌

**Status**: Tem benchmark manual, mas não loops automatizados  
**Impacto**: CRÍTICO - Necessário para validar fine-tuning

**O que falta:**

#### A) Framework de Avaliação
```python
# src/evaluation/
├── evaluator.py        # Classe principal
├── metrics/
│   ├── accuracy.py     # Métricas de accuracy
│   ├── semantic.py     # Similaridade semântica
│   ├── rouge.py        # ROUGE scores
│   └── bertscore.py    # BERTScore
├── benchmarks/
│   ├── retail_media_benchmark.json  # Benchmark proprietário
│   └── generic_benchmark.json       # Benchmark genérico
└── reports/
    └── report_generator.py
```

#### B) Métricas de Avaliação
Implementar:
- ✅ Já tem: Qualidade e relevância básicas
- ❌ ROUGE-L (overlap de n-gramas)
- ❌ BERTScore (similaridade semântica)
- ❌ Exact Match
- ❌ F1 Score
- ❌ Perplexity
- ❌ Human evaluation score

#### C) Benchmark Proprietário
Criar dataset de teste com respostas corretas:

```json
// data/evaluation/retail_media_benchmark.json
{
  "name": "Retail Media Benchmark",
  "version": "1.0",
  "questions": [
    {
      "id": "rm001",
      "question": "O que é ACOS?",
      "reference_answer": "ACOS (Advertising Cost of Sale) é a métrica...",
      "category": "metrics",
      "difficulty": "easy",
      "keywords": ["ACOS", "custo", "venda", "métrica"]
    }
    // ... mais 100-500 questões
  ]
}
```

#### D) Loop de Avaliação
```python
# src/evaluation/evaluator.py (PRECISA SER CRIADO)
from typing import Dict, List
import mlflow
from rouge_score import rouge_scorer
from bert_score import BERTScorer

class ModelEvaluator:
    """Avalia modelos fine-tunados contra benchmark"""
    
    def __init__(self, benchmark_path: str):
        self.benchmark = self.load_benchmark(benchmark_path)
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
        self.bert = BERTScorer(lang='pt')
        
    def evaluate_model(self, model, model_name: str) -> Dict:
        """
        Avalia modelo em todo o benchmark
        
        Returns:
            Dict com todas as métricas
        """
        results = []
        
        for item in self.benchmark['questions']:
            # Gerar resposta
            answer = model.invoke(item['question'])
            
            # Calcular métricas
            rouge_scores = self.rouge.score(
                item['reference_answer'],
                answer
            )
            
            bert_score = self.bert.score(
                [answer],
                [item['reference_answer']]
            )
            
            results.append({
                'question_id': item['id'],
                'rouge_l': rouge_scores['rougeL'].fmeasure,
                'bert_score': bert_score[2].item(),  # F1
                'category': item['category']
            })
        
        # Agregar métricas
        metrics = self._aggregate_metrics(results)
        
        # Log no MLflow
        with mlflow.start_run(run_name=f"eval_{model_name}"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("eval_results.json")
        
        return metrics
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Agrega métricas por categoria e geral"""
        # Calcular médias, std, por categoria
        pass
```

#### E) Avaliação Contínua
- ❌ Avaliação após cada fine-tuning
- ❌ Comparação com baseline
- ❌ Detecção de regressão
- ❌ Alertas se performance cair

```python
# Pipeline de avaliação automatizado
def automated_evaluation_loop():
    """
    Loop que roda após cada fine-tuning
    """
    # 1. Carregar modelo fine-tunado
    model = load_finetuned_model()
    
    # 2. Avaliar
    evaluator = ModelEvaluator("benchmark.json")
    metrics = evaluator.evaluate_model(model)
    
    # 3. Comparar com baseline
    baseline_metrics = load_baseline_metrics()
    
    if metrics['rouge_l'] < baseline_metrics['rouge_l']:
        send_alert("⚠️ Regressão detectada!")
    
    # 4. Se melhor, promover
    if metrics['bert_score'] > baseline_metrics['bert_score']:
        promote_to_production(model)
```

---

## 🟡 2. GAPS IMPORTANTES (MLOps Completo)

### 2.1 CI/CD Pipeline ❌

**O que falta:**
- ❌ GitHub Actions / GitLab CI
- ❌ Testes automatizados em cada commit
- ❌ Build e push de Docker images
- ❌ Deploy automático para staging
- ❌ Smoke tests pós-deploy
- ❌ Rollback automático

**Exemplo mínimo:**
```yaml
# .github/workflows/ci-cd.yml (PRECISA SER CRIADO)
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/ -v
      
      - name: Run linter
        run: |
          black --check src/
          flake8 src/
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t melancia:${{ github.sha }} .
      
      - name: Push to registry
        run: docker push melancia:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/melancia melancia=melancia:${{ github.sha }}
```

---

### 2.2 Testes Automatizados ❌

**O que falta:**

```python
# tests/ (PRECISA SER EXPANDIDO)
├── unit/
│   ├── test_model_router.py
│   ├── test_benchmark.py
│   ├── test_finetuning.py
│   └── test_api.py
├── integration/
│   ├── test_end_to_end.py
│   ├── test_mlflow_integration.py
│   └── test_api_integration.py
├── load/
│   └── test_load_performance.py
└── conftest.py  # Fixtures compartilhados
```

**Exemplos de testes necessários:**

```python
# tests/unit/test_model_router.py (PRECISA SER CRIADO)
import pytest
from src.mlops.model_router import ModelRouter, FeatureFlags

def test_router_escolhe_openai_para_premium():
    """Premium users devem sempre usar OpenAI"""
    router = ModelRouter(enable_tracking=False)
    
    decision = router.decide_routing(
        question="Teste",
        user_tier="premium"
    )
    
    assert decision.provider == "openai"
    assert decision.reason == "premium_user"

def test_router_respeita_a_b_test():
    """A/B test deve ser consistente por user_id"""
    router = ModelRouter(enable_tracking=False)
    
    # Mesmo user_id deve ter mesma experiência
    decision1 = router.decide_routing(
        question="Teste",
        user_tier="free",
        user_id=123
    )
    
    decision2 = router.decide_routing(
        question="Outra pergunta",
        user_tier="free",
        user_id=123
    )
    
    assert decision1.provider == decision2.provider

# tests/integration/test_api_integration.py (PRECISA SER CRIADO)
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_query_endpoint():
    """Testa endpoint de query"""
    response = client.post(
        "/api/v1/query",
        json={
            "question": "O que é Retail Media?",
            "user_tier": "free"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["success"] == True

def test_health_endpoint():
    """Health check deve retornar 200"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

**Coverage target: > 80%**

---

### 2.3 Monitoramento em Produção ❌

**O que falta:**

#### A) Prometheus + Grafana
```yaml
# docker-compose.monitoring.yml (PRECISA SER CRIADO)
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus_data:
  grafana_data:
```

#### B) Métricas Customizadas
```python
# src/monitoring/metrics.py (PRECISA SER CRIADO)
from prometheus_client import Counter, Histogram, Gauge

# Métricas de queries
query_counter = Counter(
    'melancia_queries_total',
    'Total de queries processadas',
    ['provider', 'user_tier']
)

query_latency = Histogram(
    'melancia_query_latency_seconds',
    'Latência de queries',
    ['provider']
)

active_users = Gauge(
    'melancia_active_users',
    'Usuários ativos'
)

model_errors = Counter(
    'melancia_model_errors_total',
    'Erros de modelo',
    ['provider', 'error_type']
)

# Uso na API
@app.post("/api/v1/query")
async def query(request: QueryRequest):
    query_counter.labels(
        provider='openai',
        user_tier=request.user_tier
    ).inc()
    
    with query_latency.labels(provider='openai').time():
        result = router.route_query(request.question)
    
    return result
```

#### C) Dashboards
- ❌ Dashboard de latência (p50, p95, p99)
- ❌ Dashboard de custos
- ❌ Dashboard de uso (queries/hora)
- ❌ Dashboard de erros
- ❌ Dashboard de A/B test

---

### 2.4 Model Registry Completo ❌

**O que falta:**

```python
# src/mlops/registry.py - EXPANDIR
class ModelRegistry:
    """Já existe, mas precisa de features"""
    
    # Adicionar:
    def promote_to_staging(self, model_name, version):
        """Promove modelo para staging"""
        pass
    
    def promote_to_production(self, model_name, version):
        """Promove modelo para produção"""
        # Validações:
        # 1. Passou nos testes?
        # 2. Performance aceitável?
        # 3. Aprovação humana?
        pass
    
    def rollback(self, model_name, to_version):
        """Rollback para versão anterior"""
        pass
    
    def compare_models(self, model1, model2):
        """Compara dois modelos"""
        pass
```

**Workflow necessário:**
```
Fine-tuning → Evaluation → Staging → A/B Test → Production
                  ↓            ↓          ↓         ↓
                Fails?     Fails?     Worse?    Issues?
                  ↓            ↓          ↓         ↓
                Stop       Stop      Rollback   Rollback
```

---

### 2.5 Drift Detection ❌

**O que falta:**

```python
# src/monitoring/drift_detection.py (PRECISA SER CRIADO)
from scipy.stats import ks_2samp
import numpy as np

class DriftDetector:
    """
    Detecta mudanças na distribuição de dados/métricas
    """
    
    def __init__(self):
        self.baseline_latencies = []
        self.baseline_quality_scores = []
        
    def collect_baseline(self, window_hours: int = 24):
        """Coleta métricas baseline"""
        # Coletar últimas 24h como baseline
        pass
    
    def detect_latency_drift(self, current_latencies):
        """Detecta drift em latência"""
        ks_stat, p_value = ks_2samp(
            self.baseline_latencies,
            current_latencies
        )
        
        if p_value < 0.05:
            return {
                "drift_detected": True,
                "metric": "latency",
                "ks_statistic": ks_stat,
                "p_value": p_value
            }
        
        return {"drift_detected": False}
    
    def detect_quality_drift(self, current_scores):
        """Detecta drift em qualidade"""
        # Comparar distribuição de scores
        pass
    
    def detect_input_drift(self, current_questions):
        """Detecta mudança no tipo de perguntas"""
        # Analisar embeddings das perguntas
        pass
```

**Alertas:**
- ❌ Email/Slack quando drift detectado
- ❌ Auto-retreinamento se drift persistir
- ❌ Dashboard de drift

---

## 🟢 3. GAPS DESEJÁVEIS (Polimento)

### 3.1 Integração Cloud MLOps ❌

**Vertex AI (Google Cloud):**
```python
# src/cloud/vertex_ai.py (PRECISA SER CRIADO)
from google.cloud import aiplatform

def deploy_to_vertex_ai(model_path: str):
    """Deploy modelo para Vertex AI"""
    aiplatform.init(project='conecta-ads')
    
    model = aiplatform.Model.upload(
        display_name="melancia-llm",
        artifact_uri=model_path
    )
    
    endpoint = model.deploy(
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1
    )
```

**Azure ML:**
```python
# src/cloud/azure_ml.py (PRECISA SER CRIADO)
from azureml.core import Workspace, Model

def deploy_to_azure_ml(model_path: str):
    """Deploy modelo para Azure ML"""
    ws = Workspace.from_config()
    
    model = Model.register(
        workspace=ws,
        model_name="melancia-llm",
        model_path=model_path
    )
    
    # Deploy to Azure ML endpoint
```

---

### 3.2 Data Versioning (DVC) ❌

```bash
# .dvc/ (PRECISA SER CONFIGURADO)
# data/ será versionado com DVC

# Comandos necessários:
dvc init
dvc add data/datasets/
dvc remote add -d storage s3://melancia-data
dvc push
```

**Benefícios:**
- Versionamento de datasets grandes
- Reprodutibilidade de experimentos
- Rastreabilidade de mudanças em dados

---

### 3.3 Model Cards ❌

```markdown
# model_cards/llama-3-2-3b-retail-media-v1.md (PRECISA SER CRIADO)

# Model Card: Llama 3.2 3B - Retail Media v1

## Informações do Modelo
- **Nome**: llama-3-2-3b-retail-media-v1
- **Base Model**: meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning**: LoRA (r=16, alpha=32)
- **Treinado em**: 19/11/2025
- **Dataset**: Conecta Ads Blog (5,000 exemplos)

## Performance
| Métrica | Valor | Baseline |
|---------|-------|----------|
| ROUGE-L | 0.68 | 0.45 |
| BERTScore | 0.82 | 0.71 |
| Latência (p95) | 3.2s | 150s |

## Casos de Uso
✅ **Recomendado:**
- Perguntas sobre métricas de Retail Media
- Estratégias de otimização de campanhas
- Comparações entre plataformas

❌ **Não Recomendado:**
- Análise financeira complexa
- Recomendações legais
- Decisões críticas sem revisão humana

## Limitações Conhecidas
- Performance inferior em perguntas muito técnicas
- Pode alucinar números/estatísticas
- Viés para Mercado Livre (mais exemplos no dataset)

## Considerações Éticas
- Sempre indicar que é resposta gerada por IA
- Não usar para decisões críticas sem revisão
- Monitora para viés e fairness

## Contato
- Responsável: Equipe MelancIA
- Email: ai@conectaads.com.br
```

---

### 3.4 Dashboard de Analytics ❌

```python
# src/dashboard/app.py (PRECISA SER CRIADO)
import streamlit as st
import plotly.express as px

st.title("🍉 MelancIA Analytics Dashboard")

# Métricas principais
col1, col2, col3 = st.columns(3)
col1.metric("Queries Hoje", "1,234", "+12%")
col2.metric("Latência Média", "4.2s", "-0.5s")
col3.metric("Custo Hoje", "$15.23", "+$2.10")

# Gráficos
st.plotly_chart(latency_over_time_chart)
st.plotly_chart(provider_distribution_chart)
st.plotly_chart(user_tier_breakdown_chart)
```

---

## 📊 4. ROADMAP DE IMPLEMENTAÇÃO

### Sprint 1 (Semanas 1-2): Fundamentos de Fine-Tuning
**Objetivo**: Primeiro modelo fine-tunado funcionando

- [ ] Criar `src/finetuning/` com estrutura básica
- [ ] Implementar `trainer.py` com LoRA
- [ ] Preparar dataset inicial (100 exemplos manuais)
- [ ] Fine-tunar Phi-3-mini ou Llama-3.2-3b
- [ ] Integrar com MLflow tracking
- [ ] Avaliar modelo fine-tunado vs baseline

**Entregável**: Modelo fine-tunado com métricas documentadas

---

### Sprint 2 (Semanas 3-4): API REST
**Objetivo**: API em produção

- [ ] Criar `src/api/` com FastAPI
- [ ] Implementar endpoints principais
- [ ] Adicionar autenticação básica
- [ ] Integrar com ModelRouter
- [ ] Dockerizar API
- [ ] Deploy em staging

**Entregável**: API funcional e documentada

---

### Sprint 3 (Semanas 5-6): Dataset e Evaluation
**Objetivo**: Dataset robusto e avaliação automatizada

- [ ] Converter blog completo em dataset
- [ ] Gerar 500+ exemplos de fine-tuning
- [ ] Criar benchmark de avaliação (100 Q&A)
- [ ] Implementar `src/evaluation/evaluator.py`
- [ ] Integrar ROUGE, BERTScore
- [ ] Loop de avaliação após fine-tuning

**Entregável**: Dataset de 1000+ exemplos + sistema de avaliação

---

### Sprint 4 (Semanas 7-8): Testes e CI/CD
**Objetivo**: Pipeline automatizado

- [ ] Criar suite de testes (>80% coverage)
- [ ] Setup GitHub Actions
- [ ] Testes automatizados em cada commit
- [ ] Build/push Docker automático
- [ ] Deploy automático para staging
- [ ] Smoke tests pós-deploy

**Entregável**: CI/CD funcional

---

### Sprint 5 (Semanas 9-10): Monitoramento
**Objetivo**: Observabilidade completa

- [ ] Setup Prometheus + Grafana
- [ ] Métricas customizadas
- [ ] Dashboards de produção
- [ ] Alertas configurados
- [ ] Drift detection básico

**Entregável**: Sistema de monitoramento em produção

---

### Sprint 6 (Semanas 11-12): Produção
**Objetivo**: Sistema completo em produção

- [ ] Re-treinamento periódico automatizado
- [ ] A/B testing em produção
- [ ] Model registry completo
- [ ] Documentação completa (Model Cards)
- [ ] Otimizações de performance
- [ ] Security audit

**Entregável**: Sistema MLOps completo

---

## 🎯 5. PRÓXIMOS PASSOS IMEDIATOS

### Esta Semana (Prioridade MÁXIMA):

#### 1. Prototipar Fine-Tuning (2-3 dias)
```bash
# Criar script básico de fine-tuning
cd src/finetuning
touch trainer.py data_preparation.py

# Preparar 50-100 exemplos manuais
# Fine-tunar Phi-3-mini com LoRA
# Avaliar se funciona
```

#### 2. Criar API Básica (2-3 dias)
```bash
# Criar API mínima funcional
cd src/api
touch main.py

# Implementar 2-3 endpoints essenciais
# Testar localmente
```

#### 3. Preparar Dataset Inicial (1-2 dias)
```bash
# Converter 100 posts do blog
# Formatar para fine-tuning
# Validar qualidade
```

---

## 📋 6. CHECKLIST DE VALIDAÇÃO

Antes de considerar o projeto "completo":

### Fine-Tuning
- [ ] Modelo fine-tunado melhora > 20% vs baseline
- [ ] Pipeline de treinamento reproduzível
- [ ] Hiperparâmetros otimizados
- [ ] Checkpoints salvos no MLflow
- [ ] Tempo de treinamento < 2 horas

### API
- [ ] Latência p95 < 5 segundos
- [ ] Rate limit implementado
- [ ] Autenticação funcionando
- [ ] Documentação OpenAPI completa
- [ ] Testes de integração passando

### MLOps
- [ ] CI/CD funcionando
- [ ] Coverage de testes > 80%
- [ ] Monitoramento em produção
- [ ] Alertas configurados
- [ ] Drift detection ativo

### Dataset
- [ ] > 1000 exemplos de fine-tuning
- [ ] Train/Val/Test splits corretos
- [ ] Benchmark de avaliação (100+ Q&A)
- [ ] Diversidade de tópicos
- [ ] Qualidade validada

### Documentação
- [ ] README atualizado
- [ ] Model Cards criados
- [ ] API documentation
- [ ] Runbook de operações
- [ ] Guia de troubleshooting

---

## 💰 7. ESTIMATIVA DE ESFORÇO

| Categoria | Sprints | Dias | Pessoas |
|-----------|---------|------|---------|
| Fine-Tuning | 1 | 10 | 1-2 |
| API REST | 1 | 10 | 1 |
| Dataset | 1 | 10 | 1-2 |
| CI/CD + Testes | 1 | 10 | 1 |
| Monitoramento | 1 | 10 | 1 |
| Produção | 1 | 10 | 2 |
| **TOTAL** | **6** | **60** | **1-2** |

**Timeline**: 3 meses (com 1-2 devs dedicados)

---

## 🚀 8. RECOMENDAÇÕES FINAIS

### Começar por:
1. ✅ **Fine-Tuning MVP** - É o core do projeto
2. ✅ **API REST** - Necessário para produção
3. ✅ **Dataset Inicial** - Sem dados, não tem IA

### Pode esperar:
- Cloud integrations (Vertex/Azure)
- DVC para versionamento
- Dashboard fancy de analytics

### Nunca pular:
- Testes automatizados
- Monitoramento básico
- Documentação

---

**Próxima Revisão**: Após implementação do Sprint 1  
**Contato**: Equipe MelancIA


# 🚀 Plano de Ação Imediata - MelancIA

**Data**: 19 de Novembro de 2025  
**Objetivo**: Implementar fine-tuning + API em 2 semanas

---

## 📊 RESUMO EXECUTIVO

### ✅ O que VOCÊ JÁ TEM (Muito bom!)
- RAG funcional com ChromaDB
- MLflow configurado
- Multi-LLM (OpenAI + Ollama + HuggingFace)
- Model Router inteligente
- Pipeline ETL para curadoria de dados

### ❌ O que FALTA (Crítico para o escopo)
1. **Fine-tuning de LLMs** - Core do projeto
2. **API REST com FastAPI** - Necessário para produção
3. **Dataset proprietário formatado** - Para treinar
4. **Evaluation loops** - Para validar qualidade

### 🎯 Prioridade Absoluta (2 semanas)
**Semana 1**: Fine-tuning MVP  
**Semana 2**: API REST funcional

---

## 🔥 SEMANA 1: FINE-TUNING MVP

### Dia 1-2: Setup e Dataset Inicial

#### 1. Criar estrutura de diretórios
```bash
cd /home/coneta/ai_agents_rag_melancia

# Criar estrutura
mkdir -p src/finetuning
mkdir -p data/datasets/{raw,processed,synthetic}
mkdir -p data/evaluation
mkdir -p model_cards

# Criar arquivos base
touch src/finetuning/__init__.py
touch src/finetuning/trainer.py
touch src/finetuning/data_preparation.py
touch src/finetuning/config.py
```

#### 2. Preparar dataset inicial
Crie 50-100 exemplos manuais primeiro:

```python
# scripts/create_initial_dataset.py (CRIAR ESTE ARQUIVO)
"""
Converte posts do blog em dataset de fine-tuning
"""
import json
import os
from pathlib import Path

def create_training_example(title, content, category="retail_media"):
    """
    Converte post do blog em exemplo de fine-tuning
    """
    # Formato ChatML (compatível com Llama/Mistral/Phi)
    return {
        "messages": [
            {
                "role": "system",
                "content": "Você é a MelancIA, especialista em Retail Media Ads e E-commerce. Responda de forma clara, objetiva e focada em ações práticas."
            },
            {
                "role": "user",
                "content": f"Explique sobre: {title}"
            },
            {
                "role": "assistant",
                "content": content[:500]  # Limitar resposta
            }
        ],
        "metadata": {
            "category": category,
            "source": "conecta_blog"
        }
    }

def main():
    # Exemplos manuais para começar
    examples = [
        {
            "title": "O que é ACOS?",
            "content": "ACOS (Advertising Cost of Sale) é a métrica fundamental em Retail Media que mede quanto você gastou em publicidade para cada real vendido. Calculado como: ACOS = (Gasto em Ads / Receita de Ads) × 100. Por exemplo, se você gastou R$ 100 em anúncios e vendeu R$ 500, seu ACOS é 20%. Quanto menor o ACOS, mais eficiente é sua campanha.",
            "category": "metricas"
        },
        {
            "title": "Como otimizar campanhas no Mercado Livre?",
            "content": "Para otimizar campanhas de Product Ads no Mercado Livre: 1) Ajuste lances baseado no ACOS target. 2) Teste diferentes títulos e imagens. 3) Use palavras-chave de cauda longa. 4) Monitore a concorrência diariamente. 5) Pause produtos com baixo ROI. 6) Concentre budget nos top performers. 7) Teste horários e dias da semana.",
            "category": "otimizacao"
        },
        {
            "title": "Diferença entre ACOS e ROAS?",
            "content": "ACOS e ROAS medem a mesma coisa de formas diferentes. ACOS mostra o custo (%) - quanto gastou para vender. ROAS mostra o retorno (múltiplo) - quanto vendeu por real gasto. ACOS = 1/ROAS. Exemplo: ACOS 20% = ROAS 5x. Use ACOS quando quer controlar custos. Use ROAS quando quer maximizar receita.",
            "category": "metricas"
        },
        {
            "title": "O que é Retail Media?",
            "content": "Retail Media é a estratégia de anunciar produtos diretamente em marketplaces e varejistas online como Mercado Livre, Amazon, Magalu. Diferente de Google Ads, você anuncia onde o cliente já está comprando, aumentando conversão. Inclui Product Ads, Display Ads e Sponsored Brands. É o formato de publicidade digital que mais cresce no Brasil.",
            "category": "conceitos"
        },
        # Adicione mais 46-96 exemplos...
    ]
    
    # Converter para formato de fine-tuning
    dataset = []
    for ex in examples:
        dataset.append(create_training_example(
            ex["title"],
            ex["content"],
            ex["category"]
        ))
    
    # Salvar
    output_path = Path("data/datasets/processed/initial_dataset.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Dataset criado: {len(dataset)} exemplos")
    print(f"📁 Salvo em: {output_path}")

if __name__ == "__main__":
    main()
```

**Executar:**
```bash
python scripts/create_initial_dataset.py
```

---

### Dia 3-4: Implementar Fine-Tuning com LoRA

```python
# src/finetuning/trainer.py (CRIAR)
"""
Fine-tuning de LLMs com LoRA/QLoRA
Otimizado para hardware CPU/GPU limitado
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import mlflow
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetailMediaFineTuner:
    """
    Fine-tuning de LLMs para Retail Media Ads
    
    Suporta:
    - LoRA (Low-Rank Adaptation)
    - QLoRA (4-bit quantization)
    - MLflow tracking
    - CPU e GPU
    
    Exemplo:
        tuner = RetailMediaFineTuner("microsoft/phi-3-mini-4k-instruct")
        tuner.train(
            train_file="data/datasets/processed/train.jsonl",
            output_dir="models/phi3-retail-v1"
        )
    """
    
    def __init__(
        self,
        base_model_name: str,
        use_4bit: bool = True,
        device: str = "auto"
    ):
        """
        Args:
            base_model_name: Nome do modelo base (HuggingFace)
            use_4bit: Usar quantização 4-bit (economiza memória)
            device: "cpu", "cuda", ou "auto"
        """
        self.base_model_name = base_model_name
        self.use_4bit = use_4bit
        self.device = device if device != "auto" else self._get_device()
        
        logger.info(f"🚀 Inicializando fine-tuner")
        logger.info(f"   Modelo base: {base_model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   4-bit: {use_4bit}")
        
        self.model = None
        self.tokenizer = None
    
    def _get_device(self):
        """Detecta melhor device disponível"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def load_model(self):
        """Carrega modelo e tokenizer"""
        logger.info("📥 Carregando modelo...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuração de quantização
        if self.use_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # CPU ou sem quantização
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
        
        logger.info("✅ Modelo carregado")
    
    def setup_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        """
        Configura LoRA
        
        Args:
            r: Rank (quanto maior, mais parâmetros treináveis)
            lora_alpha: Scaling factor
            lora_dropout: Dropout para regularização
        """
        logger.info("🔧 Configurando LoRA...")
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Preparar modelo
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Estatísticas
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"✅ LoRA configurado")
        logger.info(f"   Parâmetros treináveis: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return lora_config
    
    def prepare_dataset(self, data_file: str):
        """
        Prepara dataset para treinamento
        
        Args:
            data_file: Caminho para arquivo .jsonl
        """
        logger.info(f"📚 Preparando dataset: {data_file}")
        
        # Carregar
        dataset = load_dataset("json", data_files=data_file)
        
        # Tokenizar
        def tokenize_function(examples):
            # Converter mensagens para texto
            texts = []
            for messages in examples["messages"]:
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|{role}|>\n{content}\n"
                texts.append(text)
            
            # Tokenizar
            result = self.tokenizer(
                texts,
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            
            # Labels = inputs para causal LM
            result["labels"] = result["input_ids"].copy()
            
            return result
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        logger.info(f"✅ Dataset preparado: {len(tokenized_dataset['train'])} exemplos")
        
        return tokenized_dataset["train"]
    
    def train(
        self,
        train_file: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        mlflow_experiment: str = "retail-media-finetuning"
    ):
        """
        Executa fine-tuning
        
        Args:
            train_file: Arquivo de treino (.jsonl)
            output_dir: Diretório para salvar modelo
            epochs: Número de epochs
            batch_size: Batch size
            learning_rate: Learning rate
            mlflow_experiment: Nome do experimento MLflow
        """
        # MLflow
        mlflow.set_experiment(mlflow_experiment)
        
        with mlflow.start_run(run_name=f"{self.base_model_name.split('/')[-1]}-lora"):
            # Log params
            mlflow.log_params({
                "base_model": self.base_model_name,
                "use_4bit": self.use_4bit,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "device": self.device
            })
            
            # Carregar modelo
            if self.model is None:
                self.load_model()
            
            # Setup LoRA
            lora_config = self.setup_lora()
            mlflow.log_params({
                "lora_r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha
            })
            
            # Preparar dataset
            train_dataset = self.prepare_dataset(train_file)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,  # Simula batch maior
                learning_rate=learning_rate,
                fp16=self.device == "cuda",  # Mixed precision
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=2,
                warmup_steps=100,
                optim="adamw_torch",
                report_to=["none"],  # Desabilitar wandb
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM, não masked
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            # Treinar!
            logger.info("🚀 Iniciando treinamento...")
            result = trainer.train()
            
            # Log métricas
            mlflow.log_metrics({
                "train_loss": result.training_loss,
                "train_runtime": result.metrics["train_runtime"],
                "train_samples_per_second": result.metrics["train_samples_per_second"]
            })
            
            # Salvar modelo
            logger.info(f"💾 Salvando modelo em: {output_dir}")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Log modelo no MLflow
            mlflow.log_artifacts(output_dir, artifact_path="model")
            
            logger.info("✅ Fine-tuning concluído!")
            
            return result


def main():
    """Exemplo de uso"""
    
    # Configuração
    tuner = RetailMediaFineTuner(
        base_model_name="microsoft/Phi-3-mini-4k-instruct",  # 3.8B, leve
        use_4bit=True,
        device="auto"
    )
    
    # Treinar
    tuner.train(
        train_file="data/datasets/processed/initial_dataset.jsonl",
        output_dir="models/phi3-retail-v1",
        epochs=3,
        batch_size=2,  # Pequeno para CPU
        learning_rate=2e-4
    )


if __name__ == "__main__":
    main()
```

**Executar:**
```bash
cd /home/coneta/ai_agents_rag_melancia
python src/finetuning/trainer.py
```

**Estimativa de tempo:**
- CPU: 2-4 horas (50 exemplos, 3 epochs)
- GPU: 15-30 minutos

---

### Dia 5: Avaliar Modelo Fine-Tunado

```python
# scripts/evaluate_finetuned.py (CRIAR)
"""
Avalia modelo fine-tunado vs baseline
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import time

def load_finetuned_model(base_model_name, lora_path):
    """Carrega modelo com LoRA"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model, tokenizer

def test_model(model, tokenizer, question):
    """Testa modelo com pergunta"""
    prompt = f"""<|system|>
Você é a MelancIA, especialista em Retail Media.
<|user|>
{question}
<|assistant|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
    latency = time.time() - start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extrair apenas resposta do assistant
    response = response.split("<|assistant|>")[-1].strip()
    
    return response, latency

def main():
    print("🔬 Avaliando modelo fine-tunado vs baseline\n")
    
    # Perguntas de teste
    questions = [
        "O que é ACOS?",
        "Como otimizar campanhas no Mercado Livre?",
        "Qual a diferença entre ACOS e ROAS?",
        "O que é Retail Media?",
        "Como calcular ROI de Product Ads?"
    ]
    
    # Carregar modelos
    print("📥 Carregando baseline...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    print("📥 Carregando modelo fine-tunado...")
    ft_model, _ = load_finetuned_model(
        "microsoft/Phi-3-mini-4k-instruct",
        "models/phi3-retail-v1"
    )
    
    # Testar
    print("\n" + "="*80)
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Pergunta {i}: {question}")
        print("-"*80)
        
        # Baseline
        print("\n🔵 BASELINE:")
        response_base, latency_base = test_model(base_model, tokenizer, question)
        print(f"   Latência: {latency_base:.2f}s")
        print(f"   Resposta: {response_base[:200]}...")
        
        # Fine-tuned
        print("\n🟢 FINE-TUNED:")
        response_ft, latency_ft = test_model(ft_model, tokenizer, question)
        print(f"   Latência: {latency_ft:.2f}s")
        print(f"   Resposta: {response_ft[:200]}...")
        
        print("-"*80)
    
    print("\n✅ Avaliação concluída!")

if __name__ == "__main__":
    main()
```

**Executar:**
```bash
python scripts/evaluate_finetuned.py
```

---

## 🔥 SEMANA 2: API REST COM FASTAPI

### Dia 6-7: Criar API Base

```python
# src/api/main.py (CRIAR)
"""
API REST do MelancIA
Endpoints para queries, modelos e monitoramento
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import time
import os
from datetime import datetime

from src.mlops.model_router import ModelRouter
from src.mlops.tracking import ExperimentTracker

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="🍉 MelancIA API",
    description="API de IA para Retail Media Ads",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
router = ModelRouter(enable_tracking=True)
tracker = ExperimentTracker("melancia-api-production")

# ============================================================================
# SCHEMAS
# ============================================================================

class QueryRequest(BaseModel):
    """Request para query"""
    question: str = Field(..., min_length=3, max_length=500)
    user_tier: str = Field("free", regex="^(free|premium)$")
    user_id: Optional[int] = None
    max_tokens: int = Field(500, ge=50, le=2000)
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "question": "O que é ACOS?",
                "user_tier": "free",
                "user_id": 123,
                "max_tokens": 500,
                "temperature": 0.7
            }
        }

class QueryResponse(BaseModel):
    """Response de query"""
    answer: str
    provider: str
    model_name: str
    latency_seconds: float
    routing_reason: str
    estimated_cost: float
    success: bool
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "ACOS (Advertising Cost of Sale) é a métrica...",
                "provider": "openai",
                "model_name": "gpt-4o-mini",
                "latency_seconds": 1.5,
                "routing_reason": "premium_user",
                "estimated_cost": 0.0001,
                "success": True,
                "timestamp": "2025-11-19T10:30:00"
            }
        }

class ModelInfo(BaseModel):
    """Informações de modelo"""
    provider: str
    name: str
    status: str
    description: str

class HealthResponse(BaseModel):
    """Health check"""
    status: str
    timestamp: float
    version: str
    stats: dict

# ============================================================================
# DEPENDENCIES
# ============================================================================

def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """
    Verifica API key (simples)
    Em produção, usar JWT ou OAuth
    """
    # Por enquanto, aceitar qualquer key ou sem key
    # TODO: Implementar autenticação real
    return True

# ============================================================================
# ROUTES - QUERIES
# ============================================================================

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Queries"])
async def query(
    request: QueryRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Processa query do usuário
    
    - **question**: Pergunta sobre Retail Media
    - **user_tier**: free ou premium
    - **user_id**: ID do usuário (opcional, para A/B test)
    """
    try:
        result = router.route_query(
            question=request.question,
            user_tier=request.user_tier,
            user_id=request.user_id
        )
        
        # Adicionar timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar query: {str(e)}"
        )

@app.get("/api/v1/query/{query_id}", tags=["Queries"])
async def get_query(query_id: str):
    """Busca query por ID (TODO: implementar cache)"""
    raise HTTPException(
        status_code=501,
        detail="Endpoint não implementado ainda"
    )

# ============================================================================
# ROUTES - MODELS
# ============================================================================

@app.get("/api/v1/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """Lista modelos disponíveis"""
    return [
        ModelInfo(
            provider="openai",
            name="gpt-4o-mini",
            status="available",
            description="OpenAI GPT-4o-mini - Alta qualidade, rápido"
        ),
        ModelInfo(
            provider="ollama",
            name="phi3:mini",
            status="available",
            description="Microsoft Phi-3 Mini - Local, gratuito"
        ),
        ModelInfo(
            provider="ollama",
            name="llama3.2:3b",
            status="available",
            description="Meta Llama 3.2 3B - Local, leve"
        )
    ]

@app.get("/api/v1/models/stats", tags=["Models"])
async def models_stats():
    """Estatísticas de uso dos modelos"""
    return router.get_stats()

# ============================================================================
# ROUTES - HEALTH & MONITORING
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check
    Usado por load balancers e monitoramento
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        stats=router.get_stats()
    )

@app.get("/metrics", tags=["Health"])
async def metrics():
    """
    Métricas Prometheus
    TODO: Implementar formato Prometheus
    """
    stats = router.get_stats()
    
    # Formato texto Prometheus
    metrics_text = f"""# HELP melancia_queries_total Total de queries
# TYPE melancia_queries_total counter
melancia_queries_total{{provider="ollama"}} {stats['ollama_queries']}
melancia_queries_total{{provider="openai"}} {stats['openai_queries']}

# HELP melancia_errors_total Total de erros
# TYPE melancia_errors_total counter
melancia_errors_total {stats['errors']}
"""
    
    return metrics_text

@app.get("/api/v1/stats", tags=["Health"])
async def stats():
    """Estatísticas detalhadas"""
    return {
        "router_stats": router.get_stats(),
        "uptime_seconds": time.time(),  # TODO: track real uptime
        "version": "1.0.0"
    }

# ============================================================================
# ROUTES - ADMIN (TODO: proteger com autenticação)
# ============================================================================

@app.post("/api/v1/cache/clear", tags=["Admin"])
async def clear_cache():
    """Limpa cache"""
    # TODO: Implementar cache com Redis
    return {"message": "Cache cleared"}

@app.get("/", tags=["Root"])
async def root():
    """Redirect para docs"""
    return {
        "message": "🍉 MelancIA API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Executado ao iniciar API"""
    print("🍉 MelancIA API iniciando...")
    print(f"   Docs: http://localhost:8000/docs")
    print(f"   Health: http://localhost:8000/health")

@app.on_event("shutdown")
async def shutdown_event():
    """Executado ao desligar API"""
    router.print_stats()
    print("👋 MelancIA API encerrada")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload em desenvolvimento
        log_level="info"
    )
```

**Executar:**
```bash
cd /home/coneta/ai_agents_rag_melancia
python src/api/main.py
```

**Testar:**
```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "O que é ACOS?", "user_tier": "free"}'

# Ver docs
open http://localhost:8000/docs
```

---

### Dia 8-9: Testes da API

```python
# tests/test_api.py (CRIAR)
"""
Testes da API
"""
from fastapi.testclient import TestClient
from src.api.main import app
import pytest

client = TestClient(app)

def test_health_endpoint():
    """Health check deve retornar 200"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_query_endpoint_success():
    """Query válida deve retornar resposta"""
    response = client.post(
        "/api/v1/query",
        json={
            "question": "O que é Retail Media?",
            "user_tier": "free",
            "user_id": 123
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "provider" in data
    assert "latency_seconds" in data
    assert data["success"] == True

def test_query_endpoint_validation():
    """Query inválida deve retornar erro"""
    response = client.post(
        "/api/v1/query",
        json={
            "question": "",  # Vazio = inválido
            "user_tier": "free"
        }
    )
    
    assert response.status_code == 422  # Validation error

def test_list_models():
    """Listar modelos deve retornar lista"""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_stats_endpoint():
    """Stats deve retornar estatísticas"""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "router_stats" in data

@pytest.mark.parametrize("user_tier", ["free", "premium"])
def test_query_different_tiers(user_tier):
    """Testar diferentes tiers"""
    response = client.post(
        "/api/v1/query",
        json={
            "question": "Teste",
            "user_tier": user_tier
        }
    )
    
    assert response.status_code == 200
```

**Executar:**
```bash
pip install pytest pytest-cov
pytest tests/test_api.py -v
```

---

### Dia 10: Dockerizar e Deploy

```dockerfile
# Dockerfile.api (CRIAR)
FROM python:3.11-slim

WORKDIR /app

# Dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY src/ ./src/
COPY models/ ./models/

# Porta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.prod.yml (CRIAR)
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./models:/app/models
      - ./mlruns:/app/mlruns
    restart: unless-stopped
    networks:
      - melancia-network
  
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.1
    ports:
      - "5000:5000"
    command: mlflow ui --host 0.0.0.0 --port 5000
    volumes:
      - ./mlruns:/mlruns
    networks:
      - melancia-network

networks:
  melancia-network:
    driver: bridge
```

**Deploy:**
```bash
# Build
docker compose -f docker-compose.prod.yml build

# Run
docker compose -f docker-compose.prod.yml up -d

# Logs
docker compose -f docker-compose.prod.yml logs -f

# Stop
docker compose -f docker-compose.prod.yml down
```

---

## ✅ CHECKLIST DE 2 SEMANAS

### Semana 1: Fine-Tuning
- [ ] Dataset inicial criado (50-100 exemplos)
- [ ] `src/finetuning/trainer.py` implementado
- [ ] Modelo fine-tunado salvo em `models/`
- [ ] Avaliação mostra melhoria vs baseline
- [ ] Tracking no MLflow funcionando

### Semana 2: API
- [ ] `src/api/main.py` implementado
- [ ] Endpoints principais funcionando
- [ ] Testes passando (>80% coverage)
- [ ] Docker funcionando
- [ ] Documentação OpenAPI completa

---

## 🚀 COMANDOS RÁPIDOS

```bash
# Setup inicial
cd /home/coneta/ai_agents_rag_melancia
mkdir -p src/{finetuning,api,evaluation}
mkdir -p data/datasets/{raw,processed}
mkdir -p scripts
mkdir -p tests

# Criar dataset
python scripts/create_initial_dataset.py

# Fine-tuning
python src/finetuning/trainer.py

# Avaliar
python scripts/evaluate_finetuned.py

# API local
python src/api/main.py

# Testes
pytest tests/ -v --cov=src

# Docker
docker compose -f docker-compose.prod.yml up -d

# MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

---

## 📞 PRÓXIMOS PASSOS APÓS 2 SEMANAS

1. **Expandir dataset** (50 → 1000+ exemplos)
2. **Evaluation automatizada** (ROUGE, BERTScore)
3. **CI/CD** (GitHub Actions)
4. **Monitoramento** (Prometheus + Grafana)
5. **Model Registry** completo

---

**🎯 Objetivo**: Em 2 semanas você terá:
- ✅ Modelo fine-tunado funcionando
- ✅ API REST em produção
- ✅ Base sólida para expandir

**Dúvidas?** Consulte `docs/GAP_ANALYSIS.md` para visão completa!


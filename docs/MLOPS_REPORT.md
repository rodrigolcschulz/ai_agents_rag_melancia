# 🔬 Relatório MLOps - MelancIA

**Data**: 19 de Novembro de 2025  
**Status**: ✅ MLOps Implementado e Funcional

---

## 📊 1. STATUS ATUAL DA IMPLEMENTAÇÃO

### ✅ O que já está implementado:

#### **MLflow Tracking**
- ✅ Sistema completo de tracking de experimentos
- ✅ Rastreamento de métricas (latência, qualidade, relevância)
- ✅ Rastreamento de parâmetros (modelo, temperatura, etc)
- ✅ Salvamento de artefatos (resultados JSON/CSV)
- ✅ Comparação automática entre modelos
- ✅ Versionamento de experimentos

#### **Sistema de Benchmark**
- ✅ Comparação automatizada entre LLMs
- ✅ Avaliação de qualidade de respostas
- ✅ Medição de latência e tokens
- ✅ Cálculo de custos (OpenAI vs Ollama gratuito)
- ✅ Geração de relatórios comparativos

#### **Suporte Multi-LLM**
- ✅ OpenAI (GPT-4o-mini, GPT-3.5-turbo)
- ✅ Ollama (Phi3:mini, Llama 3.1, Mistral, Gemma)
- ✅ HuggingFace (preparado, precisa de token)

---

## 🤔 2. ISSO É MLOPS?

### **SIM! E está bem implementado!** ✅

Vocês implementaram os pilares fundamentais de MLOps:

### ✅ **Experimentação Sistemática**
- Comparação controlada entre modelos
- Reprodutibilidade de experimentos
- Métricas padronizadas

### ✅ **Tracking e Monitoramento**
- MLflow para versionamento
- Métricas de performance registradas
- Histórico completo de runs

### ✅ **Avaliação Automatizada**
- Benchmark automático de qualidade
- Avaliação de relevância
- Medição de latência e custo

### ⚠️ **O que falta para MLOps completo:**

1. **CI/CD Pipeline**
   - Testes automatizados em cada commit
   - Deploy automático quando aprovado
   - Rollback automático se falhas

2. **Monitoramento em Produção**
   - Drift detection (mudança de padrões)
   - A/B testing em produção
   - Alertas de degradação de performance

3. **Model Registry**
   - Registro formal de modelos aprovados
   - Staging → Production workflow
   - Controle de versões de modelo

4. **Infraestrutura como Código**
   - Terraform/Ansible para infra
   - Kubernetes para orquestração
   - Auto-scaling baseado em demanda

**Conclusão**: Vocês têm uma **base sólida de MLOps** (30-40% do caminho). Está muito bom para um projeto inicial!

---

## 🌐 3. ACESSO AO MLFLOW VIA REDE (SSH)

### **Problema**: MLflow roda em localhost, não acessível via SSH

### **Solução 1: MLflow com Host Binding** (Mais Simples)

```bash
# Ao invés de:
mlflow ui --port 5000

# Use (permite acesso externo):
mlflow ui --host 0.0.0.0 --port 5000
```

Então acesse via: **http://172.16.201.94:5000**

### **Solução 2: SSH Tunneling** (Mais Seguro)

```bash
# No seu computador local:
ssh -L 5000:localhost:5000 coneta@172.16.201.94

# Depois inicie MLflow no servidor:
mlflow ui --port 5000

# Acesse no navegador local: http://localhost:5000
```

### **Solução 3: Docker Compose** (Produção)

Adicionar ao `docker-compose.yml`:

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.8.1
  container_name: melancia-mlflow
  ports:
    - "5000:5000"
  volumes:
    - ./mlruns:/mlflow/mlruns
  command: mlflow ui --host 0.0.0.0 --backend-store-uri /mlflow/mlruns
  networks:
    - melancia-network
```

**Recomendação**: Use **Solução 1** para desenvolvimento, **Solução 3** para produção.

---

## 🚀 4. PRÓXIMAS ETAPAS DE IMPLANTAÇÃO - LLM GRATUITO

### **Fase 1: Validação e Otimização** (1-2 semanas)

#### ✅ Já Feito:
- ✅ Ollama instalado e funcionando
- ✅ Phi3:mini testado (590 tokens/resposta)
- ✅ Benchmark inicial rodado

#### 📋 Próximos Passos:

**1.1 - Testar Mais Modelos Locais**
```bash
# Modelos recomendados para seu hardware:
ollama pull llama3.2:3b      # Mais leve, mais rápido
ollama pull gemma2:2b        # Muito leve, qualidade ok
ollama pull mistral:7b       # Melhor qualidade, mais pesado

# Executar benchmark comparativo:
python src/experiments/run_experiments.py --mode full
```

**1.2 - Otimização de Prompts**
- Ajustar temperatura (0.3-0.7)
- Limitar tokens de resposta (max_tokens=300)
- Melhorar system prompts

**1.3 - Quantização Agressiva**
```bash
# Usar versões quantizadas (menores):
ollama pull llama3.2:3b-q4_0  # 4-bit quantization
ollama pull phi3:mini-q4      # Ainda menor
```

---

### **Fase 2: Deploy Híbrido** (2-3 semanas)

**Estratégia: Use o melhor de cada mundo**

```python
# Roteamento inteligente de queries:

def escolher_modelo(pergunta: str, usuario: dict):
    """Roteia para o modelo mais adequado"""
    
    # Perguntas simples → Ollama (gratuito)
    if len(pergunta.split()) < 15:
        return "ollama::phi3:mini"
    
    # Usuário premium → OpenAI (melhor qualidade)
    if usuario.get("plano") == "premium":
        return "openai::gpt-4o-mini"
    
    # Usuário free → Ollama
    return "ollama::llama3.2:3b"
```

**Arquitetura Híbrida:**

```
┌─────────────────┐
│   Usuário       │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Router Inteligente │ ← Decide qual modelo usar
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────┐   ┌──────┐
│Ollama│   │OpenAI│
│(Free)│   │(Pago)│
└─────┘   └──────┘
  80%       20%
```

**Implementação**:

```python
# src/agent/model_router.py
class ModelRouter:
    def __init__(self):
        self.ollama = MultiLLMManager.create_llm("ollama", "phi3:mini")
        self.openai = MultiLLMManager.create_llm("openai", "gpt-4o-mini")
        self.tracker = ExperimentTracker("melancia-production")
    
    def route_query(self, question: str, user_tier: str = "free"):
        """Roteia query para modelo apropriado"""
        
        # Lógica de roteamento
        if user_tier == "premium":
            model = self.openai
            provider = "openai"
        else:
            # 90% free users → Ollama
            if random.random() < 0.9:
                model = self.ollama
                provider = "ollama"
            else:
                # 10% para teste A/B
                model = self.openai
                provider = "openai"
        
        # Track decisão
        with self.tracker.start_run():
            self.tracker.log_params({
                "provider": provider,
                "user_tier": user_tier,
                "question_length": len(question)
            })
            
            # Executar
            start = time.time()
            response = model.invoke(question)
            latency = time.time() - start
            
            self.tracker.log_metrics({
                "latency": latency,
                "provider_cost": 0.0 if provider == "ollama" else 0.0001
            })
        
        return response
```

---

### **Fase 3: Infraestrutura de Produção** (3-4 semanas)

**3.1 - Containerização Completa**

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # API Gateway
  melancia-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_ROUTER=hybrid
      - OLLAMA_HOST=ollama:11434
    depends_on:
      - ollama
      - postgres
      - redis
  
  # Ollama (LLM local)
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
  
  # MLflow Tracking
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.1
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://melancia:pass@postgres:5432/mlflow
      - ARTIFACT_ROOT=s3://melancia-artifacts
    depends_on:
      - postgres
  
  # PostgreSQL (para MLflow + dados)
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=melancia
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  # Redis (cache)
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
  
  # Nginx (reverse proxy + load balancer)
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - melancia-api

volumes:
  ollama_models:
  postgres_data:
  redis_data:
```

**3.2 - Monitoramento**

```python
# src/mlops/monitoring.py
class ProductionMonitor:
    """Monitora performance em produção"""
    
    def __init__(self):
        self.tracker = ExperimentTracker("melancia-prod-monitoring")
    
    def log_production_metrics(self):
        """Coleta métricas de produção a cada hora"""
        
        metrics = {
            "ollama_queries_per_hour": self.count_ollama_queries(),
            "openai_queries_per_hour": self.count_openai_queries(),
            "avg_latency_ollama": self.get_avg_latency("ollama"),
            "avg_latency_openai": self.get_avg_latency("openai"),
            "total_cost_hourly": self.calculate_hourly_cost(),
            "user_satisfaction_score": self.get_satisfaction_score()
        }
        
        with self.tracker.start_run():
            self.tracker.log_metrics(metrics)
        
        # Alertas
        if metrics["avg_latency_ollama"] > 180:  # 3 min
            self.send_alert("Ollama muito lento!")
        
        if metrics["total_cost_hourly"] > 5:  # $5/hora
            self.send_alert("Custo OpenAI muito alto!")
```

---

## 💰 5. ANÁLISE DE CUSTO-BENEFÍCIO

### **Resultados do Benchmark Atual:**

| Modelo | Latência | Qualidade | Relevância | Custo |
|--------|----------|-----------|------------|-------|
| **OpenAI GPT-4o-mini** | 4.21s | 0.47 | 0.50 | $0.0000* |
| **Ollama Phi3:mini** | 154.70s | 1.00 | 0.44 | $0.0000 |

\* Custo muito baixo por query (~$0.0001)

### **Análise Detalhada:**

#### ⚡ **Velocidade**
- OpenAI: **37x mais rápido** (4s vs 155s)
- Ollama: Muito lento para produção (2.5 minutos!)
- **Motivo**: CPU inference é lento, GPU seria 10-20x mais rápido

#### ⭐ **Qualidade**
- Ollama: **2.1x melhor** no score de qualidade
- Ollama: Respostas mais longas (590 vs 68 tokens)
- **Motivo**: Phi3 está gerando respostas muito verbosas

#### 💰 **Custo Estimado (Projeção Real)**

**Cenário 1: 100% OpenAI**
```
1000 queries/dia × 100 tokens/query = 100k tokens/dia
Custo: 100k × $0.00015/1k = $15/dia = $450/mês
```

**Cenário 2: 100% Ollama**
```
Custo: $0/mês (grátis!)
Mas: servidor precisa rodar 24/7
Custo servidor (AWS EC2 t3.medium): ~$30/mês
```

**Cenário 3: Híbrido (80% Ollama + 20% OpenAI)** ⭐ **RECOMENDADO**
```
800 queries Ollama: $0
200 queries OpenAI: $3/dia = $90/mês
Servidor: $30/mês
TOTAL: $120/mês

Economia vs 100% OpenAI: $330/mês (73%)
```

### **Projeção de Economia Anual:**

| Estratégia | Custo Mensal | Custo Anual | Economia |
|------------|--------------|-------------|----------|
| 100% OpenAI | $450 | $5,400 | - |
| Híbrido 80/20 | $120 | $1,440 | **$3,960/ano** |
| 100% Ollama | $30 | $360 | $5,040/ano |

### **Recomendação de Custo-Benefício:**

**🏆 MELHOR OPÇÃO: Híbrido 80/20**

**Motivos:**
1. ✅ **73% de economia** vs OpenAI puro
2. ✅ **20% OpenAI** mantém qualidade para casos críticos
3. ✅ **Flexibilidade** para ajustar o ratio
4. ✅ **A/B testing** contínuo
5. ✅ **Fallback** se Ollama cair

**❌ NÃO recomendo 100% Ollama porque:**
- Latência inaceitável (2.5 min é muito!)
- Sem GPU, não é viável
- Experiência do usuário ruim

---

## 🎯 6. BOAS PRÁTICAS DE ENGENHARIA DE IA

### ✅ **O que vocês estão fazendo BEM:**

#### **1. Experimentação Sistemática**
```python
# ✅ BOM: Benchmark automatizado
benchmark = ModelBenchmark(retriever, memory)
benchmark.add_model("openai", "gpt-4o-mini", llm)
results = benchmark.run(test_questions)
```
- Comparação estruturada
- Métricas padronizadas
- Reprodutível

#### **2. RAG Architecture**
```python
# ✅ BOM: RAG bem estruturado
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)
```
- Contexto relevante
- Memória conversacional
- Rastreabilidade

#### **3. MLflow Tracking**
```python
# ✅ BOM: Tracking completo
tracker.log_params({"model": "gpt-4o-mini"})
tracker.log_metrics({"latency": 1.5, "quality": 0.85})
tracker.log_artifact("results.json")
```
- Versionamento
- Comparabilidade
- Auditabilidade

#### **4. Multi-LLM Support**
```python
# ✅ BOM: Abstração de provedores
llm = MultiLLMManager.create_llm("ollama", "phi3:mini")
```
- Vendor lock-in minimizado
- Fácil experimentação
- Flexibilidade

### ⚠️ **O que pode MELHORAR:**

#### **1. Testes Automatizados**
```python
# ❌ FALTA: Testes unitários
# ✅ ADICIONAR:

# tests/test_benchmark.py
def test_benchmark_quality_score():
    """Testa se avaliação de qualidade funciona"""
    benchmark = ModelBenchmark(mock_retriever, mock_memory)
    
    # Resposta boa
    score_good = benchmark._evaluate_answer_quality(
        "Retail Media é uma estratégia de publicidade..."
    )
    assert score_good > 0.5
    
    # Resposta ruim
    score_bad = benchmark._evaluate_answer_quality("")
    assert score_bad == 0.0

# tests/test_model_router.py
def test_router_escolhe_modelo_correto():
    """Testa se router escolhe modelo baseado em tier"""
    router = ModelRouter()
    
    # Premium deve usar OpenAI
    model = router.route_query("test", user_tier="premium")
    assert model.provider == "openai"
    
    # Free deve usar Ollama
    model = router.route_query("test", user_tier="free")
    assert model.provider == "ollama"
```

**Como adicionar:**
```bash
# Instalar pytest
pip install pytest pytest-cov

# Criar estrutura
mkdir tests
touch tests/__init__.py
touch tests/test_benchmark.py
touch tests/test_model_router.py

# Rodar testes
pytest tests/ -v --cov=src
```

#### **2. Validação de Dados**
```python
# ❌ FALTA: Validação de entrada
# ✅ ADICIONAR:

from pydantic import BaseModel, validator

class QueryRequest(BaseModel):
    """Valida requisições de query"""
    question: str
    user_tier: str = "free"
    max_tokens: int = 500
    
    @validator("question")
    def question_not_empty(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Pergunta muito curta")
        return v
    
    @validator("user_tier")
    def valid_tier(cls, v):
        if v not in ["free", "premium"]:
            raise ValueError("Tier inválido")
        return v

# Uso:
try:
    request = QueryRequest(question="", user_tier="invalid")
except ValidationError as e:
    return {"error": str(e)}
```

#### **3. Logging Estruturado**
```python
# ❌ ATUAL: Logs simples
logger.info("Testando modelo")

# ✅ MELHORAR: Logs estruturados
import structlog

logger = structlog.get_logger()
logger.info(
    "model_test_started",
    model_name="gpt-4o-mini",
    provider="openai",
    user_id=123,
    request_id="abc-123"
)
```

#### **4. Feature Flags**
```python
# ✅ ADICIONAR: Feature toggles para experimentos

# src/mlops/feature_flags.py
class FeatureFlags:
    """Controla features em produção sem deploy"""
    
    def __init__(self):
        self.flags = {
            "use_ollama": True,
            "enable_caching": True,
            "a_b_test_active": True,
            "ollama_percentage": 0.8,  # 80% Ollama
            "enable_monitoring": True
        }
    
    def should_use_ollama(self, user_id: int) -> bool:
        """Decide se deve usar Ollama para este usuário"""
        if not self.flags["use_ollama"]:
            return False
        
        # A/B test: hash consistente por usuário
        hash_val = hash(user_id) % 100
        return hash_val < (self.flags["ollama_percentage"] * 100)

# Uso:
flags = FeatureFlags()
if flags.should_use_ollama(user.id):
    model = ollama
else:
    model = openai
```

#### **5. Drift Detection**
```python
# ✅ ADICIONAR: Detecta mudança de distribuição

# src/mlops/drift_detector.py
from scipy.stats import ks_2samp
import numpy as np

class DriftDetector:
    """Detecta drift em métricas de produção"""
    
    def __init__(self, baseline_metrics):
        self.baseline_latencies = baseline_metrics["latencies"]
        self.baseline_qualities = baseline_metrics["qualities"]
    
    def detect_drift(self, current_metrics):
        """Compara distribuição atual com baseline"""
        
        # KS test para latência
        ks_stat, p_value = ks_2samp(
            self.baseline_latencies,
            current_metrics["latencies"]
        )
        
        if p_value < 0.05:  # Significativo
            return {
                "drift_detected": True,
                "metric": "latency",
                "severity": "high" if ks_stat > 0.3 else "medium"
            }
        
        return {"drift_detected": False}

# Uso em produção:
detector = DriftDetector(baseline_metrics)
current = collect_last_hour_metrics()
drift = detector.detect_drift(current)

if drift["drift_detected"]:
    send_alert(f"Drift detectado: {drift}")
```

#### **6. Documentação de Modelos (Model Cards)**
```markdown
# ✅ ADICIONAR: model_cards/phi3_mini_v1.md

# Model Card: Phi3-Mini v1

## Informações Básicas
- **Modelo**: phi3:mini
- **Provedor**: Ollama (Local)
- **Versão**: 2024-11
- **Última Atualização**: 19/11/2025

## Performance
- **Latência Média**: 154.70s ± 32.03s
- **Qualidade Média**: 1.00/1.0
- **Relevância Média**: 0.44/1.0
- **Tokens Médios**: 590

## Casos de Uso
✅ Recomendado para:
- Usuários free tier
- Perguntas complexas que requerem respostas longas
- Quando custo é fator limitante

❌ Não recomendado para:
- Respostas em tempo real (latência alta)
- Perguntas simples e rápidas
- Usuários premium

## Limitações Conhecidas
- Latência muito alta (2.5 min/query)
- Respostas excessivamente longas
- Relevância às vezes baixa

## Riscos e Mitigações
- **Risco**: Timeout em produção
- **Mitigação**: Timeout de 3 minutos + fallback para OpenAI

## Viés e Fairness
- Treinado em dados principalmente em inglês
- Pode ter performance inferior em português

## Uso Responsável
- Não usar para decisões críticas sem revisão humana
- Sempre mostrar que é uma resposta de IA
```

---

## 📋 7. CHECKLIST DE BOAS PRÁTICAS

### **Desenvolvimento**
- [x] Controle de versão (Git)
- [x] Ambiente virtual (venv)
- [x] Gerenciador de dependências (requirements.txt)
- [ ] Testes automatizados (pytest)
- [ ] Linting (black, flake8, mypy)
- [ ] Pre-commit hooks
- [x] Documentação (README, docstrings)

### **Experimentação**
- [x] Tracking de experimentos (MLflow)
- [x] Benchmark automatizado
- [x] Métricas padronizadas
- [ ] Testes A/B automatizados
- [ ] Statistical significance testing
- [ ] Cross-validation

### **Modelagem**
- [x] RAG architecture
- [x] Multi-model support
- [ ] Prompt versioning
- [ ] Model registry
- [ ] Fallback mechanisms
- [ ] Caching layer

### **Produção**
- [ ] CI/CD pipeline
- [x] Containerização (Docker)
- [ ] Orchestração (Docker Compose)
- [ ] Monitoramento (Prometheus/Grafana)
- [ ] Logging estruturado
- [ ] Alertas automatizados
- [ ] Health checks
- [ ] Auto-scaling

### **Governança**
- [ ] Model cards
- [ ] Data lineage
- [ ] Audit trail
- [ ] Privacy compliance (LGPD)
- [ ] Security scanning
- [ ] Backup strategy

**Score Atual: 11/37 (30%)** - Bom início! 🎯

---

## 🚀 8. ROADMAP RECOMENDADO

### **Curto Prazo (1-2 semanas)**
1. ✅ Configurar acesso MLflow via rede
2. ⏳ Testar mais modelos Ollama (llama3.2:3b, gemma2:2b)
3. ⏳ Otimizar prompts para reduzir tokens
4. ⏳ Implementar timeout e fallback
5. ⏳ Adicionar testes básicos

### **Médio Prazo (1 mês)**
6. ⏳ Implementar roteamento híbrido
7. ⏳ Configurar Docker Compose completo
8. ⏳ Adicionar monitoramento básico
9. ⏳ Criar model cards
10. ⏳ Implementar cache com Redis

### **Longo Prazo (2-3 meses)**
11. ⏳ CI/CD com GitHub Actions
12. ⏳ A/B testing em produção
13. ⏳ Drift detection
14. ⏳ Auto-scaling
15. ⏳ Dashboard de analytics

---

## 💡 9. RECOMENDAÇÕES FINAIS

### **Ação Imediata:**
```bash
# 1. Melhorar acesso MLflow
mlflow ui --host 0.0.0.0 --port 5000

# 2. Testar modelo mais leve
ollama pull llama3.2:3b
python src/experiments/run_experiments.py --mode quick

# 3. Se llama3.2:3b for < 30s latência, usar em produção
```

### **Decisão Estratégica:**

**Se GPU não for opção:**
- ❌ NÃO usar 100% Ollama (latência inviável)
- ✅ Usar OpenAI como primary (4s é aceitável)
- ✅ Usar Ollama para queries não-urgentes (análises batch)

**Se conseguir GPU:**
- ✅ Ollama vai 10-20x mais rápido
- ✅ Híbrido 80/20 se torna viável
- ✅ ROI de GPU se paga em 3-6 meses

### **Conclusão Final:**

**🎯 Vocês estão no caminho certo!**

O projeto tem:
- ✅ Arquitetura sólida
- ✅ MLOps fundamentals implementados
- ✅ Flexibilidade para experimentar
- ⚠️ Precisa otimização de performance
- ⚠️ Precisa testes e monitoramento

**Prioridade:**
1. **Agora**: Melhorar acesso MLflow + testar modelos mais leves
2. **Próxima semana**: Implementar roteamento híbrido
3. **Próximo mês**: Produtizar com Docker Compose + monitoring

**Custo-benefício:**
- Economia potencial: **$3,960/ano** com híbrido
- Mas: precisa GPU ou modelos mais leves para ser viável
- Recomendação: **Começar 100% OpenAI** ($450/mês) → **migrar gradualmente para híbrido**

---

**Documentado por**: MelancIA  
**Revisado**: 19/11/2025  
**Próxima Revisão**: Após testes com llama3.2:3b


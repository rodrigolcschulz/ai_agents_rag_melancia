# 🔄 Como Usar Evaluation Loops

## 🚀 Quick Start

### 1. Subir o Docker com Evaluation

```bash
# Descer containers antigos
docker-compose down

# Subir com a nova interface (com evaluation)
docker-compose up -d

# Ver logs
docker-compose logs -f melancia-ai
```

### 2. Acessar Interface

Abra no navegador: **http://localhost:8000**

## 🎯 Funcionalidades Novas

### ✅ Logging Automático
Todas interações são automaticamente logadas:
- Query do usuário
- Resposta gerada
- Documentos recuperados
- Modelo usado
- Latência
- Timestamp

### ✅ Métricas Automáticas (RAGAS)
Cada resposta é avaliada automaticamente:
- **Faithfulness**: Resposta baseada nos documentos?
- **Answer Relevancy**: Responde realmente a pergunta?
- **Context Relevancy**: Documentos relevantes?

### ✅ Feedback de Usuários
Na interface você verá:
- Botão **👍 Sim, ajudou!** - Feedback positivo
- Botão **👎 Não ajudou** - Feedback negativo  
- Campo para comentários opcionais

### ✅ Estatísticas em Tempo Real
Painel lateral mostra:
- Total de interações
- Latência média
- Rating médio
- Contagem de feedback positivo/negativo

## 📁 Dados Armazenados

Todos os dados ficam em:
```
data/evaluation/interactions.db
```

É um banco SQLite com 4 tabelas:
- `interactions` - Queries e respostas
- `retrieved_docs` - Documentos recuperados
- `feedback` - Feedback dos usuários
- `metrics` - Métricas calculadas

## 🔍 Analisar Dados

### Via Python

```python
from src.evaluation.interaction_logger import InteractionLogger

logger = InteractionLogger()

# Ver estatísticas
stats = logger.get_stats()
print(f"Total: {stats['total_interactions']}")
print(f"Rating médio: {stats['avg_rating']}")

# Ver interações recentes
recent = logger.get_recent_interactions(limit=10)
for interaction in recent:
    print(f"{interaction['query']} -> {interaction['response'][:50]}...")
```

### Via SQL

```bash
# Abrir banco
sqlite3 data/evaluation/interactions.db

# Ver feedback negativo (para revisar)
SELECT i.query, i.response, f.rating, f.comment
FROM interactions i
JOIN feedback f ON i.interaction_id = f.interaction_id
WHERE f.feedback_type = 'negative'
ORDER BY f.timestamp DESC;

# Métricas médias
SELECT 
    AVG(metric_value) as avg
FROM metrics
WHERE metric_name = 'faithfulness';

# Queries mais lentas
SELECT query, latency_seconds
FROM interactions
ORDER BY latency_seconds DESC
LIMIT 10;
```

## 🎯 Loop de Melhoria Contínua

### Processo Semanal

1. **Revisar Feedback Negativo**
```python
# Script para análise
import sqlite3

conn = sqlite3.connect("data/evaluation/interactions.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT i.query, i.response, f.comment
    FROM interactions i
    JOIN feedback f ON i.interaction_id = f.interaction_id
    WHERE f.feedback_type = 'negative'
    AND f.timestamp > datetime('now', '-7 days')
""")

for row in cursor.fetchall():
    print(f"\nQuery: {row[0]}")
    print(f"Resposta: {row[1][:100]}...")
    print(f"Comentário: {row[2]}")
```

2. **Identificar Padrões**
- Perguntas que sempre têm feedback negativo?
- Tópicos faltando na documentação?
- Queries confusas?

3. **Melhorar**
- Adicionar/melhorar markdowns
- Ajustar prompts
- Refinar parâmetros do RAG

4. **Testar**
- Fazer as mesmas queries problemáticas
- Verificar se melhorou

## 🔧 Customizar

### Desabilitar Evaluation

Edite `docker-compose.yml`:
```yaml
command: python -m src.agent.web_interface  # Interface antiga (sem evaluation)
```

### Mostrar Métricas nas Respostas

Edite `src/agent/web_interface_with_eval.py`:
```python
melancia = MelanciaWithEvaluation(
    enable_eval=True,
    enable_metrics_display=True  # Mudar para True
)
```

### Mudar Local do Banco

Edite `src/agent/web_interface_with_eval.py`:
```python
self.logger = InteractionLogger(
    db_path="caminho/customizado/interactions.db"
)
```

## 📊 Métricas Explicadas

### Faithfulness (0.0 - 1.0)
Mede se a resposta está "fiel" aos documentos recuperados.
- **🟢 > 0.7**: Resposta bem baseada nos docs
- **🟡 0.4-0.7**: Parcialmente baseada
- **🔴 < 0.4**: Muita alucinação

### Answer Relevancy (0.0 - 1.0)
Mede se a resposta realmente responde a pergunta.
- **🟢 > 0.7**: Responde bem
- **🟡 0.4-0.7**: Responde parcialmente
- **🔴 < 0.4**: Não responde

### Context Relevancy (0.0 - 1.0)
Mede se os documentos recuperados são relevantes.
- **🟢 > 0.7**: Docs muito relevantes
- **🟡 0.4-0.7**: Parcialmente relevantes
- **🔴 < 0.4**: Docs não relevantes

**Nota**: Estas são métricas heurísticas simples. Para maior precisão, implementar LLM-based evaluation (futuro).

## 🚧 Próximos Passos

- [ ] Dashboard Streamlit para visualizações
- [ ] Alertas automáticos (email) se qualidade cair
- [ ] A/B testing de prompts/parâmetros
- [ ] LLM-based evaluation (mais preciso)
- [ ] Relatórios semanais automáticos
- [ ] Integração com MLflow

## 🆘 Troubleshooting

### Banco não criado
```bash
# Criar pasta manualmente
mkdir -p data/evaluation
```

### Import errors
```bash
# Rebuild container
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Ver logs
```bash
docker-compose logs -f melancia-ai
```

---

**🍉 MelancIA com Evaluation Loops** - Melhoria contínua baseada em dados!


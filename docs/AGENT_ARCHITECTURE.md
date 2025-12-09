# 📁 Estrutura do Folder `src/agent/`

Este documento explica a função de cada arquivo no diretório `src/agent/` do projeto MelâncIA.

---

## 📋 Visão Geral dos Arquivos

### 🎯 **Scripts Principais (Entry Points)**

#### 1. **`web_interface_with_eval.py`** 
**🌟 SCRIPT PRINCIPAL - INTERFACE WEB MODERNA**

- ✅ **Interface web com Gradio** (http://localhost:8000)
- ✅ **Model Router integrado** (Ollama 80% + OpenAI 20%)
- ✅ **Evaluation Loops** (métricas RAGAS automáticas)
- ✅ **Sistema de feedback** (👍👎)
- ✅ **Visualização de estatísticas** em tempo real
- ✅ **Logging de interações** no banco SQLite
- ✅ **Otimizado**: Não reindexava a cada startup (só se necessário)

**Como usar:**
```bash
python src/agent/web_interface_with_eval.py
# ou
python -m src.agent.web_interface_with_eval
```

**Características:**
- Interface moderna e amigável
- Suporta Ollama (Llama 3.2:3b) e OpenAI (GPT-4o-mini)
- Fallback automático se Ollama demorar >15s
- Sistema completo de monitoramento e melhoria contínua

---

#### 2. **`main.py`**
**💻 SCRIPT ALTERNATIVO - INTERFACE CLI (Terminal)**

- ✅ **Interface de linha de comando** (terminal/console)
- ✅ **Apenas OpenAI** (GPT-4o-mini)
- ✅ **Sem evaluation loops** (mais simples)
- ✅ **Sem Model Router** 
- ✅ **Leve e rápido** para testes rápidos

**Como usar:**
```bash
python src/agent/main.py
# ou
python -m src.agent.main
```

**Características:**
- Loop de conversa no terminal
- Ideal para desenvolvimento e debugging
- Mais leve que o web_interface
- Bom para testes rápidos de prompts

---

### 🧩 **Módulos de Suporte**

#### 3. **`config.py`**
**⚙️ CONFIGURAÇÕES CENTRALIZADAS**

Define todas as constantes e configurações do projeto:
- `MODEL_NAME`: Nome do modelo LLM (gpt-4o-mini)
- `EMBEDDING_MODEL`: Modelo de embeddings (text-embedding-3-small)
- `TEMPERATURE`: Criatividade do modelo (0.5)
- Caminhos de diretórios (DATA_DIR, VECTOR_DB_DIR, LOG_DIR, etc.)
- Palavras-chave de contexto (CONTEXT_KEYWORDS)
- API Keys (OPENAI_API_KEY)

---

#### 4. **`retriever.py`**
**🔍 GESTÃO DO BANCO VETORIAL (RAG)**

Funções principais:
- `carregar_markdowns()`: Lê arquivos .md do disco
- `indexar_novos_markdowns()`: Cria embeddings e popula o Chroma DB
- `carregar_db_existente()`: Carrega banco vetorial existente
- `get_retriever()`: Cria retriever para buscar documentos similares

**Configurações importantes:**
- `k=15`: Número de documentos retornados (aumentado para capturar mais contexto)
- `chunk_size=1000`: Tamanho dos chunks de texto
- `chunk_overlap=200`: Overlap entre chunks

---

#### 5. **`prompt.py`**
**📝 TEMPLATE DE PROMPTS**

Define o template de instrução para o LLM:
- Personalidade do agente (MelâncIA)
- Instruções sobre como responder
- Como usar o contexto recuperado
- Tom de voz e estilo de resposta

---

#### 6. **`memory.py`**
**🧠 GESTÃO DE MEMÓRIA CONVERSACIONAL**

Gerencia o histórico de conversas:
- `get_memory()`: Carrega memória do chat
- `save_memory()`: Salva memória no disco
- Usa `ConversationBufferMemory` do LangChain
- Persiste em arquivo `.pkl`

---

#### 7. **`utils.py`**
**🛠️ FUNÇÕES UTILITÁRIAS**

Funções auxiliares:
- `garantir_pasta_log()`: Cria diretórios se não existirem
- `registrar_log()`: Salva logs de interações
- `is_relevant()`: Verifica se a pergunta é relevante ao domínio
- Outras funções helper

---

#### 8. **`keywords.py`**
**🔑 PALAVRAS-CHAVE DE CONTEXTO**

Lista de palavras-chave para:
- Detectar se pergunta é relevante ao domínio
- Filtrar queries fora do escopo
- Melhorar detecção de tópicos

---

## 🎯 Quando Usar Cada Script?

### Use `web_interface_with_eval.py` quando:
- ✅ Precisa de interface web moderna
- ✅ Quer usar Ollama (Llama local) 
- ✅ Precisa de métricas e evaluation
- ✅ Quer feedback de usuários
- ✅ Ambiente de produção ou demonstração

### Use `main.py` quando:
- ✅ Quer testar rápido no terminal
- ✅ Está desenvolvendo/debugando
- ✅ Não precisa de interface web
- ✅ Quer apenas OpenAI (sem Ollama)
- ✅ Ambiente de desenvolvimento

---

## 🔧 Relação entre os Scripts

```
┌──────────────────────────────────────────┐
│     web_interface_with_eval.py           │
│  (Interface Web + Ollama + Evaluation)   │
└────────────────┬─────────────────────────┘
                 │
                 │ importa
                 ▼
┌──────────────────────────────────────────┐
│  Módulos compartilhados:                 │
│  - config.py                             │
│  - retriever.py (k=15 para mais docs)    │
│  - prompt.py                             │
│  - memory.py                             │
│  - utils.py                              │
│  - keywords.py                           │
└────────────────┬─────────────────────────┘
                 │
                 │ importa
                 ▼
┌──────────────────────────────────────────┐
│            main.py                       │
│  (Interface CLI simples + só OpenAI)     │
└──────────────────────────────────────────┘
```

---

## 🚀 Melhorias Recentes

### ✅ Otimização do Retriever (k=6 → k=15)
- **Problema**: Com k=6, documentos do Central Vendedores não apareciam
- **Solução**: Aumentado para k=15, agora captura 6 docs do Central + 9 do Blog
- **Resultado**: Melhora significativa na qualidade das respostas

### ✅ Otimização do Startup do Web Interface
- **Problema**: Reindexava todo o banco vetorial a cada startup (~15s)
- **Solução**: Verifica se banco existe antes de reindexar
- **Resultado**: Startup 10x mais rápido (se banco já existe)

### ✅ Model Router com Fallback
- **80% Ollama** (Llama 3.2:3b local)
- **20% OpenAI** (GPT-4o-mini)
- **Fallback automático** se Ollama >15s
- **Configurável** via variáveis de ambiente

---

## 📊 Fluxo de Execução

### Web Interface (web_interface_with_eval.py)
```
1. Startup
   ├── Verifica se vector_db existe
   ├── Carrega (ou indexa se não existir)
   ├── Cria retriever (k=15)
   ├── Configura Model Router (Ollama + OpenAI)
   └── Inicia Gradio Server (port 8000)

2. Query do Usuário
   ├── Verifica relevância (is_relevant)
   ├── Model Router decide: Ollama ou OpenAI
   ├── Retriever busca k=15 docs similares
   ├── LLM gera resposta com contexto
   ├── Calcula métricas RAGAS (se enabled)
   ├── Loga interação no SQLite
   └── Retorna resposta + opção de feedback
```

### CLI (main.py)
```
1. Startup
   ├── Indexa documentos (sempre)
   ├── Carrega retriever (k=15)
   ├── Configura OpenAI LLM
   └── Inicia loop de conversa

2. Query do Usuário
   ├── Verifica relevância
   ├── Retriever busca docs
   ├── OpenAI gera resposta
   ├── Salva em log .txt
   └── Aguarda próxima pergunta
```

---

## 🎨 Personalização

### Mudar o modelo LLM
Edite `config.py`:
```python
MODEL_NAME = "gpt-4o-mini"  # ou "gpt-4", "gpt-3.5-turbo"
```

### Ajustar número de documentos recuperados
Edite `retriever.py`:
```python
def get_retriever(..., k=15):  # ajuste o valor de k
```

### Configurar Model Router
Variáveis de ambiente (`.env`):
```bash
USE_MODEL_ROUTER=true
OLLAMA_PERCENTAGE=0.8  # 80% Ollama, 20% OpenAI
```

---

## 🐛 Troubleshooting

### "Banco vetorial não populado"
Execute:
```bash
python src/etl/populate_vector_db.py
```

### "Ollama não responde"
1. Verifique se Ollama está rodando: `ollama list`
2. Baixe o modelo: `ollama pull llama3.2:3b`
3. Ou desabilite: `USE_MODEL_ROUTER=false`

### "Respostas não incluem documentos novos"
1. Rode o populate novamente
2. Verifique se k=15 em `retriever.py`
3. Reinicie o web_interface

---

Desenvolvido por [Conecta Ads](https://conectaads.com.br) 🍉


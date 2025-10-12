# 🍉 MelancIA - AI RAG Agente de Product Ads

**MelancIA** é um agente de IA especializado em Product Ads e E-commerce, desenvolvido pela Conecta Ads.

## 🚀 Funcionalidades

- **RAG (Retrieval-Augmented Generation)** com base de conhecimento sobre Retail Media
- **Scraping automático** de conteúdo do blog Conecta Ads
- **Interface web** interativa com Gradio
- **Pipeline ETL** completo para processamento de dados
- **Análise de conteúdo** e geração de relatórios

## 🛠️ Instalação

### Pré-requisitos

- Python 3.8+
- OpenAI API Key

### Setup Local

```bash
# Clone o repositório
git clone https://github.com/conectaads/melancia-ai-rag.git
cd melancia-ai-rag

# Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instale as dependências
pip install -r requirements.txt

# Configure a API Key
echo "OPENAI_API_KEY=sua_api_key_aqui" > .env
```

## 🎯 Como Usar

### Interface Web

```bash
python src/agent/web_interface.py
```

Acesse: http://localhost:8000

### Terminal

```bash
python src/agent/main.py
```

### Pipeline ETL

```bash
# Scraping + Análise
python src/etl/run_etl.py

# Apenas scraping
python src/etl/run_etl.py --no-analyze

# Limitar artigos
python src/etl/run_etl.py --max-articles 10
```

## 🐳 Docker

```bash
docker compose up -d
```

## 📁 Estrutura do Projeto

```text
melancia-ai-rag/
├── src/
│   ├── agent/          # Agente RAG principal
│   └── etl/            # Pipeline ETL
├── data/
│   ├── input/          # Arquivos markdown
│   ├── output/         # Relatórios e análises
│   └── vector_db/      # Base vetorial (ChromaDB)
├── logs/               # Logs do sistema
└── requirements.txt    # Dependências
```

## 📝 Sistema de Logs

O MelancIA mantém um sistema completo de logs:

- **`logs/chat_history.txt`** - Histórico completo de conversas
- **`data/output/chat_history.pkl`** - Memória da conversa (últimas 5 interações)
- **`logs/etl_pipeline.log`** - Logs do pipeline ETL
- **`logs/scraper.log`** - Logs do web scraping
- **`logs/vector_db.log`** - Logs da base vetorial

> **Nota**: Todos os arquivos de log são ignorados pelo Git (`.gitignore`)

## 🧠 Como Funciona o RAG

O MelancIA utiliza uma arquitetura RAG (Retrieval-Augmented Generation) sofisticada:

1. **📚 Base de Conhecimento**: Conteúdo do blog Conecta Ads em formato Markdown
2. **🔍 Embeddings**: Transforma o conteúdo em vetores usando OpenAI embeddings
3. **💾 Banco Vetorial**: Armazena os vetores no ChromaDB para busca rápida
4. **🤖 LLM**: GPT-4o-mini gera respostas baseadas no contexto recuperado
5. **🔄 Memória**: Mantém contexto das últimas 5 conversas
6. **🎯 Filtros**: Só responde perguntas relevantes sobre Retail Media

## 🤖 Sobre o MelancIA

MelancIA é especializado em:
- **Retail Media** e estratégias de anúncios
- **E-commerce** e marketplaces (Mercado Livre, Shopee)
- **Métricas de performance** (ACOS, ROAS, CTR, CPC)
- **Logística** e fulfillment
- **Análise de concorrência** e tendências

## 📊 Tecnologias

- **LangChain** - Framework RAG e orquestração
- **OpenAI GPT-4o-mini** - Modelo de linguagem
- **OpenAI Embeddings** - Modelo de embeddings (text-embedding-3-small)
- **ChromaDB** - Base de dados vetorial
- **Gradio** - Interface web interativa
- **BeautifulSoup** - Web scraping
- **Pandas** - Análise de dados
- **Docker** - Containerização

## 📄 Licença

Apache License 2.0 - veja [LICENSE](LICENSE) para detalhes.

## 🤝 Contribuição

Desenvolvido por [Conecta Ads](https://conectaads.com.br)

---

**🍉 Transformando perguntas em estratégias de sucesso no Retail Media!**
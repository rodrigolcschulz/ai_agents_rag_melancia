# 🍉 MelâncIA - AI RAG Agent for Retail Media

**Jou** é um agente de IA especializado em Retail Media e E-commerce, desenvolvido pela Conecta Ads.

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

```
melancia-ai-rag/
├── src/
│   ├── agent/          # Agente RAG principal
│   └── etl/            # Pipeline ETL
├── data/
│   ├── input/          # Arquivos markdown
│   ├── output/         # Relatórios e análises
│   └── vector_db/      # Base vetorial
├── logs/               # Logs do sistema
└── requirements.txt    # Dependências
```

## 🤖 Sobre o Jou

Jou é especializado em:
- **Retail Media** e estratégias de anúncios
- **E-commerce** e marketplaces (Mercado Livre, Shopee)
- **Métricas de performance** (ACOS, ROAS, CTR, CPC)
- **Logística** e fulfillment
- **Análise de concorrência** e tendências

## 📊 Tecnologias

- **LangChain** - Framework RAG
- **OpenAI GPT-4** - Modelo de linguagem
- **ChromaDB** - Base de dados vetorial
- **Gradio** - Interface web
- **BeautifulSoup** - Web scraping
- **Pandas** - Análise de dados

## 📄 Licença

Apache License 2.0 - veja [LICENSE](LICENSE) para detalhes.

## 🤝 Contribuição

Desenvolvido por [Conecta Ads](https://conectaads.com.br)

---

**🍉 Transformando perguntas em estratégias de sucesso no Retail Media!**
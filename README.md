# 🍉 MelâncIA - AI RAG Agent

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**MelâncIA** é um agente de IA especializado em Retail Media e E-commerce, construído com RAG (Retrieval-Augmented Generation) para fornecer respostas precisas e contextuais sobre anúncios, marketplaces e estratégias de vendas online.

## 🎯 Características

- **🤖 Agente Conversacional**: Interface natural para consultas sobre Retail Media
- **📚 RAG Avançado**: Sistema de recuperação de informações baseado em embeddings
- **🏪 Especialização**: Foco em Mercado Livre, Shopee e outros marketplaces
- **💾 Memória Persistente**: Mantém histórico de conversas
- **🔍 Busca Inteligente**: Recuperação semântica de documentos relevantes
- **📊 ETL Integrado**: Pipeline para coleta e processamento de dados

## 🏗️ Arquitetura

```
ai_agents_rag_melancia/
├── agent/                    # Módulo principal do agente
│   ├── main.py              # Ponto de entrada da aplicação
│   ├── config.py            # Configurações e variáveis de ambiente
│   ├── retriever.py         # Sistema de recuperação de documentos
│   ├── memory.py            # Gerenciamento de memória conversacional
│   ├── prompt.py            # Templates de prompts
│   ├── utils.py             # Utilitários e funções auxiliares
│   └── keywords.py          # Palavras-chave para filtragem de contexto
├── etl_and_scrapping/       # Pipeline de ETL
│   ├── urls.ipynb          # Coleta de URLs do blog
│   └── chunks.ipynb        # Processamento e análise de chunks
├── vector_db/              # Banco de dados vetorial (ChromaDB)
├── melanc.ia/              # Dados de entrada e saída
│   ├── Input/Blog/         # Documentos Markdown
│   └── Output/Log_JouMelancIA/  # Logs e histórico
└── docker-compose.yml      # Orquestração de containers
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- Docker e Docker Compose (opcional)
- Chave da API OpenAI

### Instalação Local

1. **Clone o repositório**
```bash
git clone https://github.com/conectaads/melancia-ai-rag.git
cd melancia-ai-rag
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**
```bash
pip install -e .
# ou para desenvolvimento
pip install -e ".[dev]"
```

4. **Configure as variáveis de ambiente**
```bash
cp env.example .env
# Edite o arquivo .env com suas configurações
```

5. **Execute o agente**
```bash
python -m agent.main
```

### Instalação com Docker

1. **Clone e configure**
```bash
git clone https://github.com/conectaads/melancia-ai-rag.git
cd melancia-ai-rag
cp env.example .env
# Edite o arquivo .env
```

2. **Execute com Docker Compose**
```bash
# Apenas o agente principal
docker-compose up melancia-ai

# Com Jupyter para desenvolvimento
docker-compose --profile dev up

# Com cache e banco de dados
docker-compose --profile cache --profile database up
```

## ⚙️ Configuração

### Variáveis de Ambiente Principais

```bash
# OpenAI (obrigatório)
OPENAI_API_KEY=your_api_key_here

# Modelos
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.5

# Caminhos
DB_DIR=./vector_db
INPUT_MARKDOWN=melanc.ia/Input/Blog/**/*.md
HISTORY_FILE=melanc.ia/Output/Log_JouMelancIA/chat_history.pkl
```

### Configuração Avançada

- **Chunk Size**: Tamanho dos fragmentos de texto (padrão: 800)
- **Chunk Overlap**: Sobreposição entre fragmentos (padrão: 100)
- **Retrieval K**: Número de documentos recuperados (padrão: 4)
- **Memory Window**: Janela de memória conversacional (padrão: 5)

## 🎮 Uso

### Interface de Linha de Comando

```bash
python -m agent.main
```

O agente iniciará uma conversa interativa onde você pode fazer perguntas sobre:
- Estratégias de Retail Media
- Otimização de campanhas no Mercado Livre
- Métricas de performance (ACOS, ROAS, CTR)
- Logística e fulfillment
- Anúncios patrocinados

### Exemplos de Perguntas

```
Você: Como otimizar o ACOS no Mercado Livre?
Jou 🍉: Para otimizar o ACOS no Mercado Livre, você deve...

Você: Qual a diferença entre Product Ads e Display Ads?
Jou 🍉: Product Ads e Display Ads são formatos distintos...
```

### Comandos Especiais

- `sair`, `exit`, `quit`: Encerra a conversa
- O agente filtra automaticamente perguntas fora do contexto de Retail Media

## 🔧 Desenvolvimento

### Estrutura do Projeto

- **`agent/`**: Módulo principal com a lógica do agente
- **`etl_and_scrapping/`**: Notebooks para coleta e processamento de dados
- **`tests/`**: Testes automatizados (a implementar)
- **`docs/`**: Documentação (a implementar)

### Comandos de Desenvolvimento

```bash
# Instalar dependências de desenvolvimento
pip install -e ".[dev]"

# Executar testes
pytest

# Formatação de código
black agent/
flake8 agent/

# Verificação de tipos
mypy agent/

# Pre-commit hooks
pre-commit install
```

### ETL e Processamento de Dados

1. **Coleta de URLs**: Execute `etl_and_scrapping/urls.ipynb`
2. **Análise de Chunks**: Execute `etl_and_scrapping/chunks.ipynb`
3. **Indexação**: O agente indexa automaticamente novos documentos

## 📊 Monitoramento

### Logs

- **Histórico de Conversas**: `melanc.ia/Output/Log_JouMelancIA/chat_history.pkl`
- **Logs de Texto**: `melanc.ia/Output/Log_JouMelancIA/chat_history.txt`
- **Logs do Sistema**: Configurável via variáveis de ambiente

### Métricas

- Número de perguntas processadas
- Taxa de relevância das perguntas
- Tempo de resposta
- Uso de tokens

## 🚀 Deploy

### Docker

```bash
# Build da imagem
docker build -t melancia-ai-rag .

# Execução
docker run -d \
  --name melancia-ai \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/vector_db:/app/vector_db \
  -v $(pwd)/melanc.ia:/app/melanc.ia \
  melancia-ai-rag
```

### Docker Compose

```bash
# Produção
docker-compose up -d

# Desenvolvimento
docker-compose --profile dev up
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Padrões de Código

- Use `black` para formatação
- Siga PEP 8 para estilo
- Adicione testes para novas funcionalidades
- Documente mudanças significativas

## 📝 Licença

Este projeto está licenciado sob a Licença Apache 2.0 - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🏢 Sobre a Conecta Ads

A Conecta Ads é especialista em Retail Media e performance digital, oferecendo soluções estratégicas para marketplaces e e-commerce.

- **Website**: [conectaads.com.br](https://conectaads.com.br)
- **Blog**: [conectaads.com.br/conteudos](https://conectaads.com.br/conteudos)
- **Contato**: contato@conectaads.com.br

## 🙏 Agradecimentos

- OpenAI pela API GPT-4
- LangChain pelo framework de RAG
- ChromaDB pelo banco de dados vetorial
- Comunidade Python pelo ecossistema

---

**🍉 MelâncIA** - Transformando perguntas em estratégias de sucesso no Retail Media!

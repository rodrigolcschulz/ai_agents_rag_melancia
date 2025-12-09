import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configurações do modelo
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.5

# Configurações de RAG
RETRIEVER_K = 15  # Número de documentos a recuperar
RETRIEVER_SEARCH_TYPE = "mmr"  # "similarity", "mmr", ou "similarity_score_threshold"
# MMR = Maximum Marginal Relevance (melhor para diversidade de respostas)

# Diretórios base
BASE_DIR = Path(__file__).parent.parent.parent  # src/agent -> src -> raiz
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
LOG_DIR = BASE_DIR / "logs"

# Caminhos de dados
INPUT_MARKDOWN = str(INPUT_DIR / "**" / "*.md")
HISTORY_FILE = str(OUTPUT_DIR / "chat_history.pkl")

# Configurações da API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_NAME = "MelancIA"

# Palavras-chave para contexto (versão resumida - lista completa em keywords.py)
CONTEXT_KEYWORDS = [
    # Core
    "retail media", "e-commerce", "marketplace", "mercado livre",
    "anúncios", "campanhas", "performance", "product ads",
    # Métricas
    "acos", "tacos", "roas", "ctr", "cpc", "roi", "conversão",
    # Estratégias
    "vendas", "palavras-chave", "lances", "orçamento", "otimização",
    # Conteúdo
    "vídeos", "fotos", "clips", "imagens",
    # Logística
    "envio", "frete", "fulfillment", "estoque", "logística",
    # Catálogo
    "produtos", "categoria", "ficha técnica", "compatibilidade",
    "autopeças", "veículo", "variações",
    # Atores
    "comprador", "vendedor", "cliente",
    # Qualidade
    "reputação", "avaliações", "mensagens", "atendimento",
    # Financeiro
    "preço", "custos", "taxa", "mercado pago", "pagamento",
    # Ferramentas
    "excel", "editor", "planilha",
    # Marca
    "loja oficial", "branding", "marca",
    # Eventos
    "black friday", "sazonalidade",
    # Tech
    "dados", "inteligência artificial", "automação"
]
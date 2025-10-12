import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configurações do modelo
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.5

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
AGENT_NAME = "Jou, a MelâncIA"

# Palavras-chave para contexto
CONTEXT_KEYWORDS = [
    "retail media", "e-commerce", "marketplace", "mercado livre",
    "anúncios", "campanhas", "performance", "acos", "roas", "ctr", "cpc",
    "conversão", "vendas", "produtos", "categoria", "palavras-chave",
    "lances", "orçamento", "impressões", "cliques", "logística",
    "fulfillment", "estoque", "preço", "concorrência", "análise",
    "relatórios", "métricas", "otimização", "estratégia", "tendências"
]
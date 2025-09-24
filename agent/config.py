import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "vector_db")
INPUT_MARKDOWN = "melanc.ia/Input/Blog/**/*.md"
HISTORY_FILE = "melanc.ia/Output/Log_JouMelancIA/chat_history.pkl"
CONTEXT_KEYWORDS = [...]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_NAME = "Jou, a MelâncIA"
LOG_DIR = os.path.dirname(HISTORY_FILE)
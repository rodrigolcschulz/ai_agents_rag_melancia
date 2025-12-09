from langchain_chroma import Chroma  # Importação atualizada
from langchain_openai import OpenAIEmbeddings  # Importação atualizada
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document  # Importar Document para criar documentos corretamente
from .config import EMBEDDING_MODEL
from .keywords import CONTEXT_KEYWORDS
import glob
import os

def carregar_markdowns(markdown_path):
    arquivos = glob.glob(markdown_path, recursive=True)
    docs = []
    for file in arquivos:
        with open(file, "r", encoding="utf-8") as f:
            texto = f.read()
            docs.append({"text": texto, "source": file})
    return docs

def indexar_novos_markdowns(docs, persist_directory, model_name=EMBEDDING_MODEL):
    # Verificar se o modelo é válido para embeddings
    valid_embedding_models = [
        "text-embedding-ada-002", 
        "text-embedding-3-small", 
        "text-embedding-3-large"
    ]
    
    # Se o modelo não for válido para embeddings, usar o padrão
    if model_name not in valid_embedding_models:
        print(f"Aviso: {model_name} não é um modelo de embedding válido. Usando text-embedding-ada-002")
        model_name = "text-embedding-ada-002"
    
    # Configurar splitter com separadores mais adequados para markdown
    # Prioriza quebras de seção (##), depois parágrafos, depois linhas
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Aumentado para capturar mais contexto
        chunk_overlap=200,  # Aumentado overlap para manter continuidade
        separators=[
            "\n## ",  # Seções de markdown
            "\n### ",  # Subseções
            "\n\n",  # Parágrafos
            "\n",  # Linhas
            ". ",  # Sentenças
            " ",  # Palavras
            ""  # Caracteres
        ],
        length_function=len,
        is_separator_regex=False
    )
    
    # Criar documentos Document corretamente com metadados
    documentos = []
    for doc in docs:
        documentos.append(Document(
            page_content=doc["text"],
            metadata={"source": doc["source"]}
        ))
    
    # Dividir os documentos em chunks
    textos = splitter.split_documents(documentos)
    
    print(f"   📄 Total de chunks criados: {len(textos)}")
    
    # Criar embeddings com o modelo correto
    embeddings = OpenAIEmbeddings(model=model_name)
    
    # Criar o banco vetorial
    db = Chroma.from_documents(textos, embeddings, persist_directory=persist_directory)
    
    # Não precisa mais do persist() - é automático no Chroma moderno
    # try:
    #     db.persist()
    # except AttributeError:
    #     # Versões mais novas do Chroma persistem automaticamente
    #     pass
    
    return db

def carregar_db_existente(persist_directory, model_name=EMBEDDING_MODEL):
    """Carrega um banco de dados Chroma existente"""
    # Verificar se o modelo é válido para embeddings
    valid_embedding_models = [
        "text-embedding-ada-002", 
        "text-embedding-3-small", 
        "text-embedding-3-large"
    ]
    
    if model_name not in valid_embedding_models:
        model_name = "text-embedding-ada-002"
    
    embeddings = OpenAIEmbeddings(model=model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db

def get_retriever(persist_directory, model_name=EMBEDDING_MODEL, k=15, search_type="mmr"):
    """
    Cria um retriever a partir do banco de dados existente
    
    Args:
        persist_directory: Diretório do banco vetorial
        model_name: Modelo de embedding
        k: Número de documentos a retornar
        search_type: Tipo de busca - "similarity" (padrão), "mmr" (diversidade), ou "similarity_score_threshold"
    
    Tipos de busca:
        - "similarity": Busca por similaridade simples (mais rápida)
        - "mmr": Maximum Marginal Relevance - retorna documentos relevantes E diversos (melhor qualidade)
        - "similarity_score_threshold": Filtra por threshold de similaridade
    """
    db = carregar_db_existente(persist_directory, model_name)
    
    if search_type == "mmr":
        # MMR: Balanceia relevância e diversidade
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 3,  # Busca 3x mais documentos e depois filtra para diversidade
                "lambda_mult": 0.7  # 0.7 = balanceado (0=máxima diversidade, 1=máxima relevância)
            }
        )
    elif search_type == "similarity_score_threshold":
        # Filtra documentos com score mínimo
        return db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": 0.5  # Apenas documentos com 50%+ de similaridade
            }
        )
    else:
        # Busca por similaridade simples (padrão)
        return db.as_retriever(
            search_kwargs={"k": k}
        )
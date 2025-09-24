from langchain_chroma import Chroma  # Importação atualizada
from langchain_openai import OpenAIEmbeddings  # Importação atualizada
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document  # Importar Document para criar documentos corretamente
from config import EMBEDDING_MODEL
from keywords import CONTEXT_KEYWORDS
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
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    # Criar documentos Document corretamente com metadados
    documentos = []
    for doc in docs:
        documentos.append(Document(
            page_content=doc["text"],
            metadata={"source": doc["source"]}
        ))
    
    # Dividir os documentos em chunks
    textos = splitter.split_documents(documentos)
    
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

def get_retriever(persist_directory, model_name=EMBEDDING_MODEL, k=4):
    """Cria um retriever a partir do banco de dados existente"""
    db = carregar_db_existente(persist_directory, model_name)
    return db.as_retriever(search_kwargs={"k": k})
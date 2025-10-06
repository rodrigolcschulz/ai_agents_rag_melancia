from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
import pickle
import os
import warnings

def get_memory(history_file, k=5):
    """
    Cria ou carrega memória de conversa com a nova API do LangChain
    k = número de interações anteriores para manter
    """
    # Criar histórico de mensagens
    chat_history = ChatMessageHistory()
    
    # Carregar histórico se existir
    if os.path.exists(history_file):
        try:
            with open(history_file, "rb") as f:
                saved_history = pickle.load(f)
                # Restaurar as mensagens no histórico
                for human_msg, ai_msg in saved_history:
                    chat_history.add_user_message(human_msg)
                    chat_history.add_ai_message(ai_msg)
        except Exception as e:
            print(f"Aviso: Não foi possível carregar o histórico: {e}")
    
    # Criar memória com o histórico
    memory = ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        chat_memory=chat_history
    )
    
    return memory

def save_memory(memory, history_file):
    """Salva a memória no arquivo"""
    try:
        # Extrair as mensagens da memória
        chat_history = []
        messages = memory.chat_memory.messages
        
        # Agrupar mensagens em pares (humano, AI)
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i].content
                ai_msg = messages[i + 1].content
                chat_history.append((human_msg, ai_msg))
        
        # Salvar no arquivo
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "wb") as f:
            pickle.dump(chat_history, f)
    except Exception as e:
        print(f"Aviso: Não foi possível salvar o histórico: {e}")
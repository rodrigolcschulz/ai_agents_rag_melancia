from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import config
from prompt import get_prompt_template
from memory import get_memory, save_memory
from retriever import carregar_markdowns, get_retriever, indexar_novos_markdowns
from utils import garantir_pasta_log, registrar_log, is_relevant
import os

def main():
    # Etapa 1 - Preparação
    garantir_pasta_log(str(config.LOG_DIR))
    memory = get_memory(config.HISTORY_FILE)

    # Etapa 2 - Indexação
    docs = carregar_markdowns(config.INPUT_MARKDOWN)
    indexar_novos_markdowns(docs, config.DB_DIR, config.EMBEDDING_MODEL)

    # Etapa 3 - Criação do retriever e da cadeia de resposta
    retriever = get_retriever(config.DB_DIR, config.EMBEDDING_MODEL)
    
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
        max_tokens=1000  # ADICIONADO: Limita o tamanho da resposta
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": get_prompt_template(),
            "document_separator": "\n\n---\n\n"  # ADICIONADO: Separador de documentos
        },
        return_source_documents=True,  # ADICIONADO: Para debugging
        verbose=False  # ADICIONADO: Reduz logs verbosos
    )

    # Etapa 4 - Loop de conversa
    print("🍉 Jou, a MelâncIA está online! Pergunte algo sobre Retail Media...")
    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("🍉 bem-te-vi... Até a próxima sinfonia de cliques!")
            break

        if not is_relevant(pergunta):
            print("🎵 Jou: Hmm... Isso soa como um tema fora do palco do Retail Media.\n"
                  "Me pergunte sobre anúncios, performance, marketplaces, logística de e-commerce... ")
            continue

        try:
            # Usar invoke com tratamento melhorado
            resultado = qa_chain.invoke({"question": pergunta})
            
            # Extrair a resposta (pode estar em 'answer' ou diretamente no resultado)
            if isinstance(resultado, dict) and 'answer' in resultado:
                resposta_texto = resultado['answer']
            else:
                resposta_texto = str(resultado)
            
            # Limpar resposta de possíveis duplicações
            resposta_texto = limpar_resposta(resposta_texto)
            
            print(f"\nJou 🍉: {resposta_texto}")
            log_file = str(config.LOG_DIR / "chat_history.txt")
            registrar_log(pergunta, resposta_texto, log_file)
            save_memory(memory, config.HISTORY_FILE)
            
        except Exception as e:
            print(f"🍉 Oops! Algo deu errado: {e}")
            print("🎵 Tentando uma abordagem diferente...")

def limpar_resposta(resposta: str) -> str:
    """
    Remove duplicações e fragmentos cortados na resposta
    """
    if not resposta:
        return resposta
    
    # Remove linhas duplicadas
    linhas = resposta.split('\n')
    linhas_unicas = []
    for linha in linhas:
        if linha.strip() and linha.strip() not in [l.strip() for l in linhas_unicas]:
            linhas_unicas.append(linha)
    
    # Remove fragmentos no final (linhas muito curtas que podem ser cortes)
    while linhas_unicas and len(linhas_unicas[-1].strip()) < 10:
        linhas_unicas.pop()
    
    return '\n'.join(linhas_unicas)

if __name__ == "__main__":
    main()

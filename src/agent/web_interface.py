"""
Interface web Gradio para o agente MelâncIA
"""
import gradio as gr
import os
import sys
from typing import List, Tuple
import config
from prompt import get_prompt_template
from memory import get_memory, save_memory
from retriever import carregar_markdowns, get_retriever, indexar_novos_markdowns
from utils import garantir_pasta_log, registrar_log, is_relevant
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


class MelanciaWebInterface:
    def __init__(self):
        """Inicializa a interface web do MelâncIA"""
        self.qa_chain = None
        self.memory = None
        self.chat_history = []
        self._setup_agent()
    
    def _setup_agent(self):
        """Configura o agente RAG"""
        try:
            # Preparação
            garantir_pasta_log(str(config.LOG_DIR))
            self.memory = get_memory(config.HISTORY_FILE)
            
            # Indexação
            docs = carregar_markdowns(config.INPUT_MARKDOWN)
            indexar_novos_markdowns(docs, config.DB_DIR, config.EMBEDDING_MODEL)
            
            # Criação do retriever e cadeia
            retriever = get_retriever(config.DB_DIR, config.EMBEDDING_MODEL)
            
            llm = ChatOpenAI(
                model=config.MODEL_NAME,
                temperature=config.TEMPERATURE,
                api_key=config.OPENAI_API_KEY,
                max_tokens=1000
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={
                    "prompt": get_prompt_template(),
                    "document_separator": "\n\n---\n\n"
                },
                return_source_documents=True,
                verbose=False
            )
            
            print("🍉 Interface web do MelâncIA inicializada com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao inicializar agente: {e}")
            raise
    
    def _limpar_resposta(self, resposta: str) -> str:
        """Remove duplicações e fragmentos cortados na resposta"""
        if not resposta:
            return resposta
        
        # Remove linhas duplicadas
        linhas = resposta.split('\n')
        linhas_unicas = []
        for linha in linhas:
            if linha.strip() and linha.strip() not in [l.strip() for l in linhas_unicas]:
                linhas_unicas.append(linha)
        
        # Remove fragmentos no final
        while linhas_unicas and len(linhas_unicas[-1].strip()) < 10:
            linhas_unicas.pop()
        
        return '\n'.join(linhas_unicas)
    
    def chat_response(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """
        Processa uma mensagem do usuário e retorna a resposta do agente
        """
        if not message.strip():
            return "", history
        
        # Verificar se a pergunta é relevante
        if not is_relevant(message):
            resposta = ("🎵 Hmm... Isso soa como um tema fora do palco do Retail Media.\n\n"
                       "Me pergunte sobre anúncios, performance, marketplaces, logística de e-commerce...")
            history.append([message, resposta])
            return "", history
        
        try:
            # Processar com o agente
            resultado = self.qa_chain.invoke({"question": message})
            
            # Extrair resposta
            if isinstance(resultado, dict) and 'answer' in resultado:
                resposta_texto = resultado['answer']
            else:
                resposta_texto = str(resultado)
            
            # Limpar resposta
            resposta_texto = self._limpar_resposta(resposta_texto)
            
            # Adicionar emoji do Jou
            resposta_final = f"Jou 🍉: {resposta_texto}"
            
            # Atualizar histórico
            history.append([message, resposta_final])
            
            # Salvar logs
            log_file = str(config.LOG_DIR / "chat_history.txt")
            registrar_log(message, resposta_texto, log_file)
            save_memory(self.memory, config.HISTORY_FILE)
            
            return "", history
            
        except Exception as e:
            erro_msg = f"🍉 Oops! Algo deu errado: {str(e)}\n\n🎵 Tentando uma abordagem diferente..."
            history.append([message, erro_msg])
            return "", history
    
    def clear_chat(self) -> Tuple[str, List]:
        """Limpa o histórico do chat"""
        return "", []
    
    def create_interface(self) -> gr.Blocks:
        """Cria a interface Gradio"""
        
        # CSS customizado
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        .user-message {
            background-color: #e3f2fd;
            text-align: right;
        }
        .bot-message {
            background-color: #f1f8e9;
        }
        """
        
        with gr.Blocks(
            title="🍉 MelâncIA - Agente de Retail Media",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🍉 MelâncIA - Agente de Retail Media
            
            **Jou** está aqui para ajudar com suas dúvidas sobre Retail Media, E-commerce e Marketplaces!
            
            💡 **Dicas**: Pergunte sobre ACOS, ROAS, campanhas no Mercado Livre, Shopee, estratégias de anúncios, etc.
            """)
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="Conversa com Jou 🍉",
                height=500,
                show_label=True,
                avatar_images=("👤", "🍉"),
                bubble_full_width=False
            )
            
            # Input area
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Digite sua pergunta sobre Retail Media...",
                    label="Sua pergunta",
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Enviar", variant="primary", scale=1)
                clear_btn = gr.Button("Limpar", variant="secondary", scale=1)
            
            # Event handlers
            msg_input.submit(
                self.chat_response,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            send_btn.click(
                self.chat_response,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                self.clear_chat,
                outputs=[msg_input, chatbot]
            )
            
            # Footer
            gr.Markdown("""
            ---
            **🍉 MelâncIA** - Transformando perguntas em estratégias de sucesso no Retail Media!
            
            Desenvolvido por [Conecta Ads](https://conectaads.com.br)
            """)
        
        return interface


def main():
    """Função principal para executar a interface web"""
    try:
        # Criar interface
        melancia_interface = MelanciaWebInterface()
        interface = melancia_interface.create_interface()
        
        # Executar
        interface.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Erro ao iniciar interface web: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

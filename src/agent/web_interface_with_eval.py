"""
Interface web Gradio para o agente MelancIA com Evaluation Loops integrado.

Adiciona:
- Logging automático de todas interações
- Cálculo de métricas RAGAS
- Feedback de usuário (👍👎)
- Visualização de estatísticas
- Model Router para LLMs open source (Ollama) e OpenAI
"""
import gradio as gr
import os
import sys
import time
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading

from . import config
from .prompt import get_prompt_template
from .memory import get_memory, save_memory
from .retriever import carregar_markdowns, get_retriever, indexar_novos_markdowns
from .utils import garantir_pasta_log, registrar_log, is_relevant
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Import evaluation modules
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.interaction_logger import InteractionLogger
from evaluation.rag_evaluator import RAGEvaluator

# Import MLOps components
from mlops.model_router import ModelRouter, FeatureFlags


class MelanciaWithEvaluation:
    """MelancIA com Evaluation Loops integrado."""
    
    def __init__(self, enable_eval=True, enable_metrics_display=False, use_router=True, ollama_percentage=0.8):
        """
        Inicializa a interface com evaluation.
        
        Args:
            enable_eval: Se True, loga interações e calcula métricas
            enable_metrics_display: Se True, mostra métricas na resposta
            use_router: Se True, usa ModelRouter (Ollama + OpenAI), senão usa só OpenAI
            ollama_percentage: Porcentagem de queries para Ollama (0.0 a 1.0)
        """
        self.qa_chain = None
        self.memory = None
        self.chat_history = []
        self.enable_eval = enable_eval
        self.enable_metrics_display = enable_metrics_display
        self.use_router = use_router
        
        # Evaluation components
        if self.enable_eval:
            self.logger = InteractionLogger(
                db_path=str(Path(config.DATA_DIR) / "evaluation" / "interactions.db")
            )
            self.evaluator = RAGEvaluator()
        
        # Model Router (se habilitado)
        self.model_router = None
        if self.use_router:
            print(f"🔀 Model Router ativo: {ollama_percentage*100:.0f}% Ollama / {(1-ollama_percentage)*100:.0f}% OpenAI")
            feature_flags = FeatureFlags({
                "ollama_percentage": ollama_percentage,
                "max_latency_ollama": 15.0,     # Fallback após 15s
                "enable_fallback": True,         # OpenAI se demorar
                "enable_caching": True,
            })
            self.model_router = ModelRouter(feature_flags, enable_tracking=enable_eval)
        
        # Mapear interaction_ids para poder dar feedback
        self.message_to_interaction_id = {}
        
        # User ID counter para A/B testing
        self.user_id_counter = 0
        
        self._setup_agent()
    
    def _setup_agent(self):
        """Configura o agente RAG."""
        try:
            # Preparação
            garantir_pasta_log(str(config.LOG_DIR))
            self.memory = get_memory(config.HISTORY_FILE)
            
            # Verificar se banco vetorial existe
            vector_db_path = Path(config.VECTOR_DB_DIR)
            if not (vector_db_path / "chroma.sqlite3").exists():
                print("⚠️  Banco vetorial não encontrado. Indexando documentos...")
                docs = carregar_markdowns(config.INPUT_MARKDOWN)
                indexar_novos_markdowns(docs, str(config.VECTOR_DB_DIR), config.EMBEDDING_MODEL)
                print("✅ Indexação concluída!")
            else:
                print("✅ Usando banco vetorial existente")
            
            # Criação do retriever
            self.retriever = get_retriever(
                str(config.VECTOR_DB_DIR), 
                config.EMBEDDING_MODEL,
                k=config.RETRIEVER_K,
                search_type=config.RETRIEVER_SEARCH_TYPE
            )
            
            # Criar QA chains (com ou sem router)
            if self.use_router:
                # Com router: criar chains para ambos os modelos
                print("🔧 Configurando chains com Model Router...")
                self.qa_chains = {
                    "openai": self._create_qa_chain(self.model_router.openai_model),
                    "ollama": self._create_qa_chain(self.model_router.ollama_model)
                }
                self.qa_chain = None  # Será escolhido dinamicamente
            else:
                # Sem router: apenas OpenAI (comportamento original)
                llm = ChatOpenAI(
                    model=config.MODEL_NAME,
                    temperature=config.TEMPERATURE,
                    api_key=config.OPENAI_API_KEY,
                    max_tokens=1000
                )
                self.qa_chain = self._create_qa_chain(llm)
            
            status_msg = "🍉 MelancIA com Evaluation Loops" if self.enable_eval else "🍉 MelancIA"
            if self.use_router:
                status_msg += " + Model Router"
            print(f"{status_msg} inicializado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao inicializar agente: {e}")
            raise
    
    def _create_qa_chain(self, llm):
        """Cria uma ConversationalRetrievalChain para um LLM específico."""
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={
                "prompt": get_prompt_template(),
                "document_separator": "\n\n---\n\n"
            },
            return_source_documents=True,
            verbose=False
        )
    
    def _limpar_resposta(self, resposta: str) -> str:
        """Remove duplicações e fragmentos cortados na resposta."""
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
    
    def chat_response(
        self, 
        message: str, 
        history: List[List[str]]
    ) -> Tuple[str, List[List[str]], str]:
        """
        Processa uma mensagem do usuário e retorna a resposta.
        
        Returns:
            (input_cleared, updated_history, interaction_id)
        """
        if not message.strip():
            return "", history, ""
        
        # Verificar relevância
        if not is_relevant(message):
            resposta = ("🎵 Hmm... Isso soa como um tema fora do palco do Retail Media.\n\n"
                       "Me pergunte sobre anúncios, performance, marketplaces, logística de e-commerce...")
            history.append([message, resposta])
            return "", history, ""
        
        start_time = time.time()
        interaction_id = ""
        provider_used = "openai"  # Default
        model_name_used = config.MODEL_NAME
        
        try:
            # Escolher modelo com router (se habilitado)
            if self.use_router:
                # Incrementar user_id para A/B testing
                self.user_id_counter += 1
                
                # Decidir qual modelo usar
                decision = self.model_router.decide_routing(
                    question=message,
                    user_tier="free",  # Pode ser parametrizado
                    user_id=self.user_id_counter
                )
                
                provider_used = decision.provider
                model_name_used = decision.model_name
                
                # Usar chain apropriada
                qa_chain = self.qa_chains[decision.provider]
                
                print(f"🔀 Roteado para: {decision.provider}::{decision.model_name} (motivo: {decision.reason})")
                
                # Se for Ollama, usar timeout proativo
                if provider_used == "ollama":
                    timeout_seconds = 15.0
                    resultado = None
                    
                    # Executar com timeout usando ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(qa_chain.invoke, {"question": message})
                        try:
                            resultado = future.result(timeout=timeout_seconds)
                            latency = time.time() - start_time
                            print(f"✅ Ollama respondeu em {latency:.2f}s")
                        except FuturesTimeoutError:
                            latency = time.time() - start_time
                            print(f"⚠️  Ollama timeout após {timeout_seconds}s - Fallback para OpenAI...")
                            self.model_router.stats["fallbacks"] += 1
                            
                            # Fallback para OpenAI
                            start_time = time.time()
                            qa_chain = self.qa_chains["openai"]
                            resultado = qa_chain.invoke({"question": message})
                            latency = time.time() - start_time
                            provider_used = "openai"
                            model_name_used = "gpt-4o-mini"
                            print(f"✅ OpenAI respondeu em {latency:.2f}s (fallback)")
                else:
                    # OpenAI direto (sem timeout necessário)
                    resultado = qa_chain.invoke({"question": message})
                    latency = time.time() - start_time
            else:
                # Sem router: apenas OpenAI
                qa_chain = self.qa_chain
                resultado = qa_chain.invoke({"question": message})
                latency = time.time() - start_time
            
            # Extrair resposta e documentos
            if isinstance(resultado, dict) and 'answer' in resultado:
                resposta_texto = resultado['answer']
                source_docs = resultado.get('source_documents', [])
            else:
                resposta_texto = str(resultado)
                source_docs = []
            
            # Limpar resposta
            resposta_texto = self._limpar_resposta(resposta_texto)
            
            # ==== EVALUATION LOOP ====
            metrics = {}
            if self.enable_eval:
                # Preparar documentos para logging
                retrieved_docs = []
                doc_contents = []
                
                for i, doc in enumerate(source_docs):
                    doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    doc_contents.append(doc_content)
                    
                    retrieved_docs.append({
                        "id": f"doc_{i}",
                        "content": doc_content,
                        "score": None,  # LangChain não retorna score direto
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                    })
                
                # Log interaction
                interaction_id = self.logger.log_interaction(
                    query=message,
                    response=resposta_texto,
                    retrieved_docs=retrieved_docs,
                    model_name=model_name_used,
                    provider=provider_used,
                    latency_seconds=latency,
                    metadata={
                        "interface": "gradio",
                        "router_enabled": self.use_router
                    }
                )
                
                # Calcular métricas
                if doc_contents:
                    metrics = self.evaluator.evaluate_interaction(
                        query=message,
                        response=resposta_texto,
                        retrieved_docs=doc_contents
                    )
                    
                    # Log metrics
                    numeric_metrics = {
                        k: v for k, v in metrics.items() 
                        if isinstance(v, (int, float))
                    }
                    self.logger.log_metrics(interaction_id, numeric_metrics)
                
                # Mapear para poder dar feedback depois
                self.message_to_interaction_id[message] = interaction_id
            
            # Preparar resposta final
            model_emoji = "🦙" if provider_used == "ollama" else "🤖"
            resposta_final = f"🍉 **MelancIA** {model_emoji}: {resposta_texto}"
            
            # Adicionar métricas se habilitado
            if self.enable_metrics_display and metrics:
                metrics_str = self._format_metrics(metrics, latency, provider_used, model_name_used)
                resposta_final += f"\n\n{metrics_str}"
            
            # Atualizar histórico
            history.append([message, resposta_final])
            
            # Salvar logs tradicionais
            log_file = str(config.LOG_DIR / "chat_history.txt")
            registrar_log(message, resposta_texto, log_file)
            save_memory(self.memory, config.HISTORY_FILE)
            
            return "", history, interaction_id
            
        except Exception as e:
            erro_msg = f"🍉 **MelancIA**: Oops! Algo deu errado: {str(e)}\n\n🎵 Tentando uma abordagem diferente..."
            history.append([message, erro_msg])
            return "", history, ""
    
    def _format_metrics(self, metrics: Dict, latency: float, provider: str = "openai", model_name: str = "gpt-4o-mini") -> str:
        """Formata métricas para exibição."""
        lines = ["\n---", "📊 **Métricas de Qualidade**:"]
        
        if "faithfulness" in metrics:
            emoji = "🟢" if metrics["faithfulness"] > 0.7 else "🟡" if metrics["faithfulness"] > 0.4 else "🔴"
            lines.append(f"{emoji} Faithfulness: {metrics['faithfulness']:.2f}")
        
        if "answer_relevancy" in metrics:
            emoji = "🟢" if metrics["answer_relevancy"] > 0.7 else "🟡" if metrics["answer_relevancy"] > 0.4 else "🔴"
            lines.append(f"{emoji} Answer Relevancy: {metrics['answer_relevancy']:.2f}")
        
        if "context_relevancy" in metrics:
            emoji = "🟢" if metrics["context_relevancy"] > 0.7 else "🟡" if metrics["context_relevancy"] > 0.4 else "🔴"
            lines.append(f"{emoji} Context Relevancy: {metrics['context_relevancy']:.2f}")
        
        lines.append(f"⏱️ Latência: {latency:.2f}s")
        lines.append(f"🤖 Modelo: {provider}::{model_name}")
        
        return "\n".join(lines)
    
    def submit_feedback(
        self, 
        feedback_type: str, 
        last_query: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None
    ) -> str:
        """Registra feedback do usuário."""
        if not self.enable_eval:
            return "⚠️ Evaluation desabilitado."
        
        # Encontrar interaction_id da última query
        interaction_id = self.message_to_interaction_id.get(last_query)
        
        if not interaction_id:
            return "⚠️ Não foi possível encontrar a interação para dar feedback."
        
        # Registrar feedback
        self.logger.log_feedback(
            interaction_id=interaction_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment
        )
        
        emoji = "👍" if feedback_type == "positive" else "👎"
        return f"{emoji} Feedback registrado! Obrigado por nos ajudar a melhorar."
    
    def get_stats_display(self) -> str:
        """Retorna estatísticas formatadas com detalhes de latência."""
        if not self.enable_eval:
            return "⚠️ Evaluation desabilitado."
        
        try:
            stats = self.logger.get_stats()
            
            lines = [
                "📊 **Estatísticas do Sistema**",
                "",
                f"📈 Total de interações: {stats.get('total_interactions', 0)}",
                "",
                "### ⏱️ **Latência**",
                f"• Média geral: {stats.get('avg_latency', 0):.2f}s",
                f"• Média (24h): {stats.get('avg_latency_24h', 0):.2f}s",
                f"• Queries lentas (>10s): {stats.get('slow_queries_count', 0)}",
            ]
            
            # Latency por provider
            latency_by_provider = stats.get('latency_by_provider', {})
            if latency_by_provider:
                lines.append("\n### 🤖 **Por Modelo**")
                for provider, provider_stats in latency_by_provider.items():
                    emoji = "🦙" if provider == "ollama" else "🤖"
                    lines.append(f"\n**{emoji} {provider.upper()}** ({provider_stats['count']} queries)")
                    lines.append(f"  • Média: {provider_stats['avg']:.2f}s")
                    lines.append(f"  • P50: {provider_stats['p50']:.2f}s | P95: {provider_stats['p95']:.2f}s")
                    lines.append(f"  • Min: {provider_stats['min']:.2f}s | Max: {provider_stats['max']:.2f}s")
            
            # Router stats
            if self.use_router and hasattr(self.model_router, 'stats'):
                router_stats = self.model_router.stats
                lines.append("\n### 🔀 **Model Router**")
                lines.append(f"• Fallbacks (timeout): {router_stats.get('fallbacks', 0)}")
                
                provider_dist = stats.get('provider_distribution', {})
                if provider_dist:
                    total = sum(provider_dist.values())
                    for provider, count in provider_dist.items():
                        pct = (count / total * 100) if total > 0 else 0
                        emoji = "🦙" if provider == "ollama" else "🤖"
                        lines.append(f"  {emoji} {provider}: {pct:.1f}% ({count} queries)")
            
            # Rating
            if stats.get('avg_rating'):
                lines.append(f"\n### ⭐ **Qualidade**")
                lines.append(f"• Rating médio: {stats['avg_rating']:.2f}/5")
            
            # Feedback
            feedback_counts = stats.get('feedback_counts', {})
            if feedback_counts:
                lines.append("\n### 📝 **Feedback**")
                for feedback_type, count in feedback_counts.items():
                    emoji = "👍" if feedback_type == "positive" else "👎"
                    lines.append(f"  {emoji} {feedback_type}: {count}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"❌ Erro ao buscar estatísticas: {e}"
    
    def clear_chat(self) -> Tuple[str, List, str]:
        """Limpa o histórico do chat."""
        return "", [], ""
    
    def create_interface(self) -> gr.Blocks:
        """Cria a interface Gradio com feedback integrado."""
        
        custom_css = """
        .gradio-container {
            max-width: 1400px !important;
        }
        .feedback-buttons {
            margin-top: 10px;
        }
        """
        
        with gr.Blocks(
            title="🍉 MelancIA - Assistente de Marketplace",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            # State para guardar última query e interaction_id
            last_query_state = gr.State("")
            current_interaction_id = gr.State("")
            
            # Header
            gr.Markdown("""
            # 🍉 MelancIA - Assistente de Marketplace

            **MelancIA** está aqui para ajudar com suas dúvidas sobre Retail Media, E-commerce e Marketplaces!
            
            💡 **Dicas**: Pergunte sobre ACOS, campanhas no Mercado Livre, Product Ads, estratégias de anúncios, otimização de performance, etc.
            
            ⚡ **Importante**: Evite perguntas com poucas palavras! Quanto mais contexto você fornecer, melhor o assistente consegue buscar informações relevantes na base de conhecimento.
            """)
            
            with gr.Row():
                # Coluna principal - Chat
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversa com MelancIA 🍉",
                        height=500,
                        show_label=True,
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
                    
                    # Feedback buttons
                    if self.enable_eval:
                        gr.Markdown("### 📝 Gostou da última resposta?")
                        with gr.Row():
                            feedback_positive_btn = gr.Button("👍 Sim, ajudou!", variant="primary")
                            feedback_negative_btn = gr.Button("👎 Não ajudou", variant="stop")
                        
                        feedback_comment = gr.Textbox(
                            placeholder="(Opcional) Deixe um comentário...",
                            label="Comentário",
                            lines=2
                        )
                        
                        feedback_output = gr.Markdown("")
                
                # Coluna lateral - Estatísticas
                if self.enable_eval:
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Estatísticas")
                        stats_display = gr.Markdown("")
                        refresh_stats_btn = gr.Button("🔄 Atualizar", size="sm")
            
            # Event handlers para chat
            def chat_with_state_update(message, history):
                _, updated_history, interaction_id = self.chat_response(message, history)
                return "", updated_history, message, interaction_id
            
            msg_input.submit(
                chat_with_state_update,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, last_query_state, current_interaction_id]
            )
            
            send_btn.click(
                chat_with_state_update,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, last_query_state, current_interaction_id]
            )
            
            clear_btn.click(
                self.clear_chat,
                outputs=[msg_input, chatbot, current_interaction_id]
            )
            
            # Event handlers para feedback
            if self.enable_eval:
                def submit_positive_feedback(last_query, comment):
                    return self.submit_feedback("positive", last_query, rating=5, comment=comment)
                
                def submit_negative_feedback(last_query, comment):
                    return self.submit_feedback("negative", last_query, rating=2, comment=comment)
                
                feedback_positive_btn.click(
                    submit_positive_feedback,
                    inputs=[last_query_state, feedback_comment],
                    outputs=[feedback_output]
                )
                
                feedback_negative_btn.click(
                    submit_negative_feedback,
                    inputs=[last_query_state, feedback_comment],
                    outputs=[feedback_output]
                )
                
                refresh_stats_btn.click(
                    self.get_stats_display,
                    outputs=[stats_display]
                )
                
                # Atualizar stats ao carregar
                interface.load(
                    self.get_stats_display,
                    outputs=[stats_display]
                )
            
            # Footer
            gr.Markdown("""
            ---
            **🍉 MelancIA** - Transformando perguntas em estratégias de sucesso no Retail Media!
            
            🔄 Sistema de melhoria contínua baseado em feedback e métricas automáticas.
            
            Desenvolvido por [Conecta Ads](https://conectaads.com.br)
            """)
        
        return interface


def main():
    """Função principal para executar a interface web com evaluation."""
    try:
        # Configurações via variáveis de ambiente
        use_router = os.getenv("USE_MODEL_ROUTER", "true").lower() == "true"
        ollama_percentage = float(os.getenv("OLLAMA_PERCENTAGE", "0.8"))  # 80% padrão
        
        print("\n" + "="*60)
        print("🍉 Inicializando MelancIA RAG Agent")
        print("="*60)
        print(f"📊 Evaluation Loops: ATIVO")
        print(f"🔀 Model Router: {'ATIVO' if use_router else 'DESATIVADO'}")
        if use_router:
            print(f"🦙 Ollama: {ollama_percentage*100:.0f}%")
            print(f"🤖 OpenAI: {(1-ollama_percentage)*100:.0f}%")
        print("="*60 + "\n")
        
        # Criar interface com evaluation e router
        melancia = MelanciaWithEvaluation(
            enable_eval=True,              # Ativar evaluation loops
            enable_metrics_display=False,  # Não mostrar métricas nas respostas (opcional)
            use_router=use_router,         # Usar Model Router (Ollama + OpenAI)
            ollama_percentage=ollama_percentage
        )
        interface = melancia.create_interface()
        
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


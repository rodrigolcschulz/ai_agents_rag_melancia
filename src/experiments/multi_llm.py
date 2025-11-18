"""
Gerenciador de múltiplos LLMs (OpenAI, Ollama, HuggingFace)
Suporta troca fácil entre diferentes modelos para experimentação
"""
import os
from typing import Optional, Dict, Any, Literal, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.llms import HuggingFaceHub
import logging

# Carregar variáveis de ambiente do .env
load_dotenv()

logger = logging.getLogger(__name__)

LLMProvider = Literal["openai", "ollama", "huggingface"]


class MultiLLMManager:
    """
    Gerenciador unificado de múltiplos provedores de LLM
    
    Exemplos:
        # OpenAI
        llm = MultiLLMManager.create_llm("openai", model_name="gpt-4o-mini")
        
        # Ollama (local)
        llm = MultiLLMManager.create_llm("ollama", model_name="llama3.1:8b")
        
        # HuggingFace
        llm = MultiLLMManager.create_llm("huggingface", model_name="mistralai/Mistral-7B-Instruct-v0.2")
    """
    
    # Modelos recomendados por provedor
    RECOMMENDED_MODELS = {
        "openai": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        "ollama": [
            "llama3.1:8b",
            "llama3.1:70b",
            "mistral:7b",
            "phi3:mini",
            "gemma2:9b",
            "qwen2.5:7b",
        ],
        "huggingface": [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "tiiuae/falcon-7b-instruct",
            "microsoft/phi-2",
        ],
    }
    
    @staticmethod
    def create_llm(
        provider: LLMProvider,
        model_name: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = 1000,
        **kwargs
    ) -> Union[ChatOpenAI, OllamaLLM, HuggingFaceHub]:
        """
        Cria uma instância de LLM baseado no provedor escolhido
        
        Args:
            provider: Provedor do LLM ("openai", "ollama", "huggingface")
            model_name: Nome do modelo (usa default se não especificado)
            temperature: Temperatura para geração (0-1)
            max_tokens: Número máximo de tokens na resposta
            **kwargs: Argumentos adicionais específicos do provedor
            
        Returns:
            Instância do LLM configurado
        """
        if provider == "openai":
            return MultiLLMManager._create_openai_llm(
                model_name, temperature, max_tokens, **kwargs
            )
        elif provider == "ollama":
            return MultiLLMManager._create_ollama_llm(
                model_name, temperature, **kwargs
            )
        elif provider == "huggingface":
            return MultiLLMManager._create_huggingface_llm(
                model_name, temperature, max_tokens, **kwargs
            )
        else:
            raise ValueError(
                f"Provedor '{provider}' não suportado. "
                f"Use: openai, ollama, ou huggingface"
            )
    
    @staticmethod
    def _create_openai_llm(
        model_name: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> ChatOpenAI:
        """Cria LLM da OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada no ambiente")
        
        model = model_name or "gpt-4o-mini"
        logger.info(f"Inicializando OpenAI LLM: {model}")
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
    
    @staticmethod
    def _create_ollama_llm(
        model_name: Optional[str],
        temperature: float,
        **kwargs
    ) -> OllamaLLM:
        """Cria LLM do Ollama (local)"""
        model = model_name or "llama3.1:8b"
        base_url = kwargs.pop("base_url", "http://localhost:11434")
        
        logger.info(f"Inicializando Ollama LLM: {model} @ {base_url}")
        
        return OllamaLLM(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def _create_huggingface_llm(
        model_name: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> HuggingFaceHub:
        """Cria LLM do HuggingFace"""
        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN não encontrada no ambiente")
        
        model = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
        logger.info(f"Inicializando HuggingFace LLM: {model}")
        
        model_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_tokens or 1000,
            **kwargs.pop("model_kwargs", {})
        }
        
        return HuggingFaceHub(
            repo_id=model,
            huggingfacehub_api_token=api_token,
            model_kwargs=model_kwargs,
            **kwargs
        )
    
    @staticmethod
    def list_models(provider: LLMProvider) -> list[str]:
        """Lista modelos recomendados para um provedor"""
        return MultiLLMManager.RECOMMENDED_MODELS.get(provider, [])
    
    @staticmethod
    def get_model_info(provider: LLMProvider, model_name: str) -> Dict[str, Any]:
        """Retorna informações sobre um modelo específico"""
        info = {
            "provider": provider,
            "model_name": model_name,
            "available": model_name in MultiLLMManager.RECOMMENDED_MODELS.get(provider, []),
        }
        
        # Adicionar informações específicas
        if provider == "ollama":
            info["local"] = True
            info["cost"] = "Gratuito (local)"
        elif provider == "openai":
            info["local"] = False
            info["cost"] = "Pago (API)"
        elif provider == "huggingface":
            info["local"] = False
            info["cost"] = "Gratuito (API com limites)"
        
        return info


class LLMFactory:
    """
    Factory simplificada para casos de uso comuns
    """
    
    @staticmethod
    def get_best_free_llm() -> Union[ChatOpenAI, OllamaLLM, HuggingFaceHub]:
        """Retorna o melhor LLM gratuito disponível (Ollama local)"""
        try:
            return MultiLLMManager.create_llm("ollama", "llama3.1:8b")
        except Exception as e:
            logger.warning(f"Ollama não disponível: {e}")
            # Fallback para HuggingFace
            return MultiLLMManager.create_llm(
                "huggingface", 
                "mistralai/Mistral-7B-Instruct-v0.2"
            )
    
    @staticmethod
    def get_best_quality_llm() -> Union[ChatOpenAI, OllamaLLM, HuggingFaceHub]:
        """Retorna o melhor LLM em qualidade (OpenAI)"""
        return MultiLLMManager.create_llm("openai", "gpt-4o-mini")
    
    @staticmethod
    def get_fastest_llm() -> Union[ChatOpenAI, OllamaLLM, HuggingFaceHub]:
        """Retorna o LLM mais rápido (Phi-3 local)"""
        try:
            return MultiLLMManager.create_llm("ollama", "phi3:mini")
        except Exception:
            return MultiLLMManager.create_llm("openai", "gpt-3.5-turbo")


# Exemplo de uso
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Testar diferentes provedores
    print("=== Modelos Recomendados ===")
    for provider in ["openai", "ollama", "huggingface"]:
        print(f"\n{provider.upper()}:")
        for model in MultiLLMManager.list_models(provider):
            info = MultiLLMManager.get_model_info(provider, model)
            print(f"  - {model} ({info['cost']})")
    
    # Criar LLMs
    print("\n=== Criando LLMs ===")
    try:
        llm_openai = MultiLLMManager.create_llm("openai")
        print(f"✓ OpenAI criado: {llm_openai}")
    except Exception as e:
        print(f"✗ OpenAI falhou: {e}")
    
    try:
        llm_ollama = MultiLLMManager.create_llm("ollama")
        print(f"✓ Ollama criado: {llm_ollama}")
    except Exception as e:
        print(f"✗ Ollama falhou: {e}")


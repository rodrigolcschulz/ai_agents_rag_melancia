"""
Gerenciador de múltiplos LLMs.
Suporta OpenAI, Ollama e outros providers.
"""

from typing import Any
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import os


class MultiLLMManager:
    """Gerenciador para criar e configurar diferentes LLMs."""
    
    @staticmethod
    def create_llm(provider: str, model_name: str, **kwargs) -> Any:
        """
        Cria uma instância de LLM baseado no provider.
        
        Args:
            provider: "openai", "ollama", ou "huggingface"
            model_name: Nome do modelo específico
            **kwargs: Parâmetros adicionais para o modelo
        
        Returns:
            Instância do LLM configurado
        
        Raises:
            ValueError: Se provider não é suportado
        """
        if provider == "openai":
            return MultiLLMManager._create_openai_llm(model_name, **kwargs)
        elif provider == "ollama":
            return MultiLLMManager._create_ollama_llm(model_name, **kwargs)
        else:
            raise ValueError(f"Provider não suportado: {provider}")
    
    @staticmethod
    def _create_openai_llm(model_name: str, **kwargs) -> ChatOpenAI:
        """Cria instância do ChatOpenAI."""
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada")
        
        return ChatOpenAI(
            model=model_name,
            temperature=kwargs.get("temperature", 0.5),
            api_key=api_key,
            max_tokens=kwargs.get("max_tokens", 1000)
        )
    
    @staticmethod
    def _create_ollama_llm(model_name: str, **kwargs) -> Ollama:
        """Cria instância do Ollama."""
        return Ollama(
            model=model_name,
            temperature=kwargs.get("temperature", 0.5),
            base_url=kwargs.get("base_url", "http://localhost:11434")
        )


class LLMFactory:
    """Factory class para criar LLMs (alias para MultiLLMManager)."""
    
    @staticmethod
    def create(provider: str, model_name: str, **kwargs):
        """Cria LLM usando MultiLLMManager."""
        return MultiLLMManager.create_llm(provider, model_name, **kwargs)


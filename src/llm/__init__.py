"""
Módulo de gerenciamento de LLMs.

Fornece interface unificada para diferentes providers (OpenAI, Ollama, etc).
"""

from .manager import MultiLLMManager, LLMFactory

__all__ = ["MultiLLMManager", "LLMFactory"]


"""
🎯 Fine-Tuning Module - QLoRA Implementation

Este módulo contém ferramentas para fine-tuning de LLMs usando QLoRA.
Projetado para rodar em Google Colab com GPU gratuita.
"""

from .data_prep import DatasetPreparator, format_instruction_dataset
from .qlora_trainer import QLorATrainer, get_qlora_config
from .export_to_ollama import export_to_gguf, ModelExporter

__version__ = "1.0.0"

__all__ = [
    "DatasetPreparator",
    "format_instruction_dataset",
    "QLorATrainer",
    "get_qlora_config",
    "export_to_gguf",
    "ModelExporter",
]


"""
🎯 QLoRA Trainer - Fine-tuning Eficiente de LLMs

Implementa fine-tuning usando QLoRA (Quantized LoRA) para treinar
modelos grandes com memória limitada.

Baseado em:
- Paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- bitsandbytes para quantização 4-bit
- PEFT (Parameter-Efficient Fine-Tuning)
"""

import os
import torch
from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import Dataset, DatasetDict
import wandb


@dataclass
class QLorAConfig:
    """Configuração para QLoRA fine-tuning."""
    
    # Modelo
    model_name: str = "meta-llama/Llama-2-7b-hf"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_bias: str = "none"
    
    # Treinamento
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Otimização
    optim: str = "paged_adamw_32bit"
    weight_decay: float = 0.001
    fp16: bool = False
    bf16: bool = True
    max_seq_length: int = 2048
    
    # Logging
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Outros
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "qlora-finetuning"
    wandb_run_name: Optional[str] = None


def get_qlora_config(
    model_size: str = "7b",
    gpu_memory: str = "16gb",
    quick_test: bool = False
) -> QLorAConfig:
    """
    Retorna configuração pré-definida baseada em tamanho do modelo e GPU.
    
    Args:
        model_size: Tamanho do modelo ('3b', '7b', '13b')
        gpu_memory: Memória da GPU ('12gb', '16gb', '24gb', '40gb')
        quick_test: Se True, usa configuração rápida para testes
    
    Returns:
        QLorAConfig configurado
    """
    config = QLorAConfig()
    
    if quick_test:
        config.num_train_epochs = 1
        config.max_seq_length = 512
        config.save_steps = 50
        config.eval_steps = 50
        return config
    
    # Ajustar batch size baseado na memória
    if gpu_memory == "12gb":
        config.per_device_train_batch_size = 2
        config.gradient_accumulation_steps = 8
    elif gpu_memory == "16gb":
        config.per_device_train_batch_size = 4
        config.gradient_accumulation_steps = 4
    elif gpu_memory == "24gb":
        config.per_device_train_batch_size = 8
        config.gradient_accumulation_steps = 2
    elif gpu_memory == "40gb":
        config.per_device_train_batch_size = 16
        config.gradient_accumulation_steps = 1
    
    # Ajustar LoRA rank baseado no tamanho do modelo
    if model_size == "3b":
        config.lora_r = 8
        config.lora_alpha = 16
    elif model_size == "7b":
        config.lora_r = 16
        config.lora_alpha = 32
    elif model_size == "13b":
        config.lora_r = 32
        config.lora_alpha = 64
    
    return config


class QLorATrainer:
    """Trainer para fine-tuning com QLoRA."""
    
    def __init__(
        self,
        config: QLorAConfig,
        model_name: Optional[str] = None
    ):
        """
        Args:
            config: Configuração QLoRA
            model_name: Nome do modelo (override config)
        """
        self.config = config
        if model_name:
            self.config.model_name = model_name
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Inicializar W&B se habilitado
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
    
    def load_model_and_tokenizer(self):
        """Carrega modelo e tokenizer com quantização."""
        print(f"🔄 Carregando modelo: {self.config.model_name}")
        
        # Configurar quantização
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
        )
        
        # Carregar modelo
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,  # Necessário para gradient checkpointing
        )
        
        # Preparar modelo para treinamento k-bit
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configurar LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias=self.config.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        # Aplicar LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # Configurar padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.print_trainable_parameters()
        print("✅ Modelo e tokenizer carregados!")
    
    def tokenize_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        text_column: str = "text"
    ) -> Union[Dataset, DatasetDict]:
        """
        Tokeniza dataset.
        
        Args:
            dataset: Dataset a tokenizar
            text_column: Nome da coluna com texto
        
        Returns:
            Dataset tokenizado
        """
        def tokenize_function(examples):
            outputs = self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs
        
        print("🔄 Tokenizando dataset...")
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names if isinstance(dataset, Dataset) else dataset["train"].column_names,
            desc="Tokenizing",
        )
        
        print("✅ Dataset tokenizado!")
        return tokenized
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ):
        """
        Inicia treinamento.
        
        Args:
            train_dataset: Dataset de treino (já tokenizado)
            eval_dataset: Dataset de validação (opcional)
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Configurar argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            report_to="wandb" if self.config.use_wandb else "none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Criar trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Treinar
        print("🚀 Iniciando treinamento...")
        self.trainer.train()
        print("✅ Treinamento concluído!")
    
    def save_model(self, output_dir: Union[str, Path]):
        """
        Salva modelo treinado.
        
        Args:
            output_dir: Diretório de saída
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Salvando modelo em: {output_dir}")
        
        # Salvar apenas adaptadores LoRA (muito menor)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("✅ Modelo salvo!")
        print(f"📁 Adaptadores LoRA: {output_dir}")
        print("\n💡 Para usar o modelo:")
        print("1. Carregue o modelo base")
        print("2. Aplique os adaptadores LoRA com PeftModel.from_pretrained()")
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Avalia modelo.
        
        Args:
            eval_dataset: Dataset de avaliação (tokenizado)
        
        Returns:
            Métricas de avaliação
        """
        if self.trainer is None:
            raise ValueError("Trainer não inicializado. Execute train() primeiro.")
        
        print("📊 Avaliando modelo...")
        metrics = self.trainer.evaluate(eval_dataset)
        
        print("\n✅ Resultados da avaliação:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def load_finetuned_model(
        base_model_name: str,
        adapter_path: Union[str, Path],
        load_in_4bit: bool = True
    ):
        """
        Carrega modelo fine-tunado.
        
        Args:
            base_model_name: Nome do modelo base
            adapter_path: Caminho para adaptadores LoRA
            load_in_4bit: Se True, carrega em 4-bit
        
        Returns:
            Modelo e tokenizer
        """
        print(f"🔄 Carregando modelo fine-tunado...")
        print(f"   Base: {base_model_name}")
        print(f"   Adapter: {adapter_path}")
        
        # Configurar quantização
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Carregar modelo base
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Carregar adaptadores
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Carregar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        print("✅ Modelo fine-tunado carregado!")
        
        return model, tokenizer


if __name__ == "__main__":
    print("🧪 Testando QLorATrainer...")
    
    # Criar configuração de teste
    config = get_qlora_config(model_size="3b", gpu_memory="16gb", quick_test=True)
    
    print("\n📋 Configuração de teste:")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  Max sequence length: {config.max_seq_length}")
    
    print("\n✅ Configuração criada com sucesso!")
    print("💡 Para treinar, use o notebook fine_tuning_qlora_colab.ipynb")


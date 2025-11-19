"""
📦 Export Fine-Tuned Models to Ollama

Converte modelos fine-tunados para formato GGUF compatível com Ollama.
Permite usar seus modelos customizados localmente.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Union, Dict, Any
import shutil


class ModelExporter:
    """Exporta modelos fine-tunados para diferentes formatos."""
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Args:
            model_path: Caminho para modelo fine-tunado (com adaptadores LoRA)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ValueError(f"Modelo não encontrado: {model_path}")
    
    def merge_lora_adapters(
        self,
        base_model_name: str,
        output_path: Union[str, Path]
    ):
        """
        Merge adaptadores LoRA com modelo base.
        
        Args:
            base_model_name: Nome do modelo base HuggingFace
            output_path: Onde salvar modelo merged
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
        except ImportError:
            raise ImportError(
                "Instale dependências: pip install transformers peft torch"
            )
        
        print(f"🔄 Merging LoRA adapters...")
        print(f"   Base model: {base_model_name}")
        print(f"   Adapters: {self.model_path}")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Carregar modelo base
        print("📥 Carregando modelo base...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Carregar e fazer merge dos adaptadores
        print("🔗 Aplicando adaptadores LoRA...")
        model = PeftModel.from_pretrained(base_model, str(self.model_path))
        model = model.merge_and_unload()
        
        # Salvar modelo merged
        print(f"💾 Salvando modelo merged em: {output_path}")
        model.save_pretrained(output_path)
        
        # Salvar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        tokenizer.save_pretrained(output_path)
        
        print("✅ Merge concluído!")
        return output_path
    
    def convert_to_gguf(
        self,
        merged_model_path: Union[str, Path],
        output_path: Union[str, Path],
        quantization: str = "q4_k_m"
    ):
        """
        Converte modelo para formato GGUF (Ollama/llama.cpp).
        
        IMPORTANTE: Requer llama.cpp instalado!
        
        Args:
            merged_model_path: Caminho para modelo merged
            output_path: Caminho de saída (.gguf)
            quantization: Tipo de quantização ('q4_k_m', 'q5_k_m', 'q8_0')
        
        Returns:
            Caminho para arquivo GGUF
        """
        print("⚠️ AVISO: Esta função requer llama.cpp instalado!")
        print("   Visite: https://github.com/ggerganov/llama.cpp")
        
        merged_model_path = Path(merged_model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Verificar se llama.cpp está disponível
        llama_cpp_path = os.getenv("LLAMA_CPP_PATH")
        if not llama_cpp_path:
            print("\n❌ LLAMA_CPP_PATH não configurado!")
            print("   Execute: export LLAMA_CPP_PATH=/path/to/llama.cpp")
            return None
        
        llama_cpp_path = Path(llama_cpp_path)
        convert_script = llama_cpp_path / "convert.py"
        quantize_binary = llama_cpp_path / "quantize"
        
        if not convert_script.exists():
            print(f"❌ Script de conversão não encontrado: {convert_script}")
            return None
        
        # Passo 1: Converter para GGUF FP16
        print("\n🔄 Passo 1: Convertendo para GGUF FP16...")
        fp16_path = output_path.with_suffix(".fp16.gguf")
        
        cmd = [
            "python",
            str(convert_script),
            str(merged_model_path),
            "--outfile", str(fp16_path),
            "--outtype", "f16"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Convertido para FP16: {fp16_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro na conversão: {e}")
            return None
        
        # Passo 2: Quantizar
        if quantization != "f16":
            print(f"\n🔄 Passo 2: Quantizando para {quantization}...")
            
            if not quantize_binary.exists():
                print(f"❌ Binário de quantização não encontrado: {quantize_binary}")
                print("   Execute: make quantize no diretório llama.cpp")
                return fp16_path
            
            cmd = [
                str(quantize_binary),
                str(fp16_path),
                str(output_path),
                quantization
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"✅ Quantizado: {output_path}")
                
                # Remover FP16 temporário
                fp16_path.unlink()
                
                return output_path
            except subprocess.CalledProcessError as e:
                print(f"❌ Erro na quantização: {e}")
                return fp16_path
        
        return fp16_path
    
    def create_ollama_modelfile(
        self,
        gguf_path: Union[str, Path],
        output_path: Union[str, Path],
        model_name: str,
        template: Optional[str] = None,
        system_message: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Cria Modelfile para Ollama.
        
        Args:
            gguf_path: Caminho para arquivo GGUF
            output_path: Onde salvar Modelfile
            model_name: Nome do modelo
            template: Template de prompt (opcional)
            system_message: Mensagem de sistema (opcional)
            parameters: Parâmetros do modelo (opcional)
        """
        output_path = Path(output_path)
        gguf_path = Path(gguf_path)
        
        # Template padrão (Alpaca)
        if template is None:
            template = """{{- if .System }}### System:
{{ .System }}

{{ end }}### Instruction:
{{ .Prompt }}

### Response:
"""
        
        # System message padrão
        if system_message is None:
            system_message = (
                "Você é um assistente especializado em Retail Media e advertising. "
                "Responda de forma clara, precisa e baseada em dados."
            )
        
        # Parâmetros padrão
        if parameters is None:
            parameters = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 2048,
            }
        
        # Criar Modelfile
        modelfile_content = f"""# {model_name} - Fine-tuned Model
FROM {gguf_path.absolute()}

# Template
TEMPLATE """{template}"""

# System message
SYSTEM """{system_message}"""

# Parameters
"""
        
        for key, value in parameters.items():
            modelfile_content += f"PARAMETER {key} {value}\n"
        
        # Salvar
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"✅ Modelfile criado: {output_path}")
        print(f"\n💡 Para usar no Ollama:")
        print(f"   ollama create {model_name} -f {output_path}")
        print(f"   ollama run {model_name}")
        
        return output_path


def export_to_gguf(
    adapter_path: Union[str, Path],
    base_model_name: str,
    output_name: str,
    quantization: str = "q4_k_m",
    output_dir: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """
    Função helper para exportar modelo completo para GGUF.
    
    Args:
        adapter_path: Caminho para adaptadores LoRA
        base_model_name: Nome do modelo base HuggingFace
        output_name: Nome do modelo final
        quantization: Tipo de quantização
        output_dir: Diretório de saída (opcional)
    
    Returns:
        Caminho para Modelfile do Ollama (se bem-sucedido)
    
    Example:
        >>> export_to_gguf(
        ...     adapter_path="./results/checkpoint-100",
        ...     base_model_name="meta-llama/Llama-2-7b-hf",
        ...     output_name="llama2-retail-media",
        ...     quantization="q4_k_m"
        ... )
    """
    if output_dir is None:
        output_dir = Path("./exported_models")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exporter = ModelExporter(adapter_path)
    
    # Passo 1: Merge LoRA
    merged_path = output_dir / f"{output_name}_merged"
    print("\n" + "="*60)
    print("PASSO 1: MERGE LORA ADAPTERS")
    print("="*60)
    exporter.merge_lora_adapters(base_model_name, merged_path)
    
    # Passo 2: Converter para GGUF
    gguf_path = output_dir / f"{output_name}.{quantization}.gguf"
    print("\n" + "="*60)
    print("PASSO 2: CONVERTER PARA GGUF")
    print("="*60)
    gguf_result = exporter.convert_to_gguf(merged_path, gguf_path, quantization)
    
    if gguf_result is None:
        print("\n⚠️ Conversão GGUF falhou ou foi pulada")
        print("   Você pode converter manualmente usando llama.cpp")
        print(f"   Modelo merged disponível em: {merged_path}")
        return None
    
    # Passo 3: Criar Modelfile
    modelfile_path = output_dir / f"{output_name}.Modelfile"
    print("\n" + "="*60)
    print("PASSO 3: CRIAR MODELFILE OLLAMA")
    print("="*60)
    exporter.create_ollama_modelfile(
        gguf_path=gguf_result,
        output_path=modelfile_path,
        model_name=output_name
    )
    
    print("\n" + "="*60)
    print("✅ EXPORT COMPLETO!")
    print("="*60)
    print(f"📁 Diretório de saída: {output_dir}")
    print(f"📦 Modelo merged: {merged_path}")
    print(f"🔧 Arquivo GGUF: {gguf_result}")
    print(f"📝 Modelfile: {modelfile_path}")
    
    return modelfile_path


if __name__ == "__main__":
    print("🧪 Testando ModelExporter...")
    print("\n💡 Este módulo requer:")
    print("   1. Modelo fine-tunado (adaptadores LoRA)")
    print("   2. llama.cpp instalado (para conversão GGUF)")
    print("\n📚 Para usar, veja o notebook: notebooks/fine_tuning_qlora_colab.ipynb")


"""
📊 Dataset Preparation for Fine-Tuning

Prepara datasets no formato correto para fine-tuning com QLoRA.
Suporta múltiplos formatos: instruction-following, Q&A, conversational.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from datasets import Dataset, DatasetDict


class DatasetPreparator:
    """Prepara datasets para fine-tuning de LLMs."""
    
    def __init__(
        self,
        template: str = "alpaca",
        system_message: Optional[str] = None
    ):
        """
        Args:
            template: Formato do prompt ('alpaca', 'chatml', 'llama2')
            system_message: Mensagem de sistema opcional
        """
        self.template = template
        self.system_message = system_message or self._get_default_system_message()
        
    def _get_default_system_message(self) -> str:
        """Retorna mensagem de sistema padrão."""
        return (
            "Você é um assistente especializado em Retail Media e advertising. "
            "Responda de forma clara, precisa e baseada em dados."
        )
    
    def format_alpaca(
        self,
        instruction: str,
        output: str,
        input_text: str = ""
    ) -> str:
        """
        Formato Alpaca para instruction-following.
        
        Args:
            instruction: A instrução/pergunta
            output: A resposta esperada
            input_text: Contexto adicional (opcional)
        
        Returns:
            Prompt formatado
        """
        if input_text:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )
        return prompt
    
    def format_chatml(
        self,
        user_message: str,
        assistant_message: str
    ) -> str:
        """
        Formato ChatML (usado por GPT-3.5/4, Phi-3).
        
        Args:
            user_message: Mensagem do usuário
            assistant_message: Resposta do assistente
        
        Returns:
            Prompt formatado em ChatML
        """
        prompt = (
            f"<|system|>\n{self.system_message}<|end|>\n"
            f"<|user|>\n{user_message}<|end|>\n"
            f"<|assistant|>\n{assistant_message}<|end|>"
        )
        return prompt
    
    def format_llama2(
        self,
        user_message: str,
        assistant_message: str
    ) -> str:
        """
        Formato Llama 2 chat.
        
        Args:
            user_message: Mensagem do usuário
            assistant_message: Resposta do assistente
        
        Returns:
            Prompt formatado para Llama 2
        """
        prompt = (
            f"[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n"
            f"{user_message} [/INST] {assistant_message}"
        )
        return prompt
    
    def prepare_from_json(
        self,
        json_path: Union[str, Path],
        instruction_key: str = "instruction",
        output_key: str = "output",
        input_key: Optional[str] = "input"
    ) -> Dataset:
        """
        Prepara dataset a partir de arquivo JSON.
        
        Args:
            json_path: Caminho para arquivo JSON
            instruction_key: Nome da chave com instruções
            output_key: Nome da chave com respostas
            input_key: Nome da chave com input (opcional)
        
        Returns:
            Dataset formatado
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            instruction = item[instruction_key]
            output = item[output_key]
            input_text = item.get(input_key, "") if input_key else ""
            
            if self.template == "alpaca":
                text = self.format_alpaca(instruction, output, input_text)
            elif self.template == "chatml":
                text = self.format_chatml(instruction, output)
            elif self.template == "llama2":
                text = self.format_llama2(instruction, output)
            else:
                raise ValueError(f"Template inválido: {self.template}")
            
            formatted_data.append({"text": text})
        
        return Dataset.from_list(formatted_data)
    
    def prepare_from_csv(
        self,
        csv_path: Union[str, Path],
        question_col: str = "question",
        answer_col: str = "answer",
        context_col: Optional[str] = None
    ) -> Dataset:
        """
        Prepara dataset a partir de arquivo CSV.
        
        Args:
            csv_path: Caminho para arquivo CSV
            question_col: Nome da coluna com perguntas
            answer_col: Nome da coluna com respostas
            context_col: Nome da coluna com contexto (opcional)
        
        Returns:
            Dataset formatado
        """
        df = pd.read_csv(csv_path)
        
        formatted_data = []
        for _, row in df.iterrows():
            question = row[question_col]
            answer = row[answer_col]
            context = row[context_col] if context_col and context_col in df.columns else ""
            
            if self.template == "alpaca":
                text = self.format_alpaca(question, answer, context)
            elif self.template == "chatml":
                text = self.format_chatml(question, answer)
            elif self.template == "llama2":
                text = self.format_llama2(question, answer)
            else:
                raise ValueError(f"Template inválido: {self.template}")
            
            formatted_data.append({"text": text})
        
        return Dataset.from_list(formatted_data)
    
    def prepare_from_rag_logs(
        self,
        logs_dir: Union[str, Path],
        min_quality_score: float = 4.0
    ) -> Dataset:
        """
        Prepara dataset a partir de logs do sistema RAG.
        Filtra apenas interações com boa qualidade.
        
        Args:
            logs_dir: Diretório com logs
            min_quality_score: Score mínimo de qualidade (1-5)
        
        Returns:
            Dataset formatado
        """
        logs_dir = Path(logs_dir)
        formatted_data = []
        
        # Procurar por arquivos de log JSON
        for log_file in logs_dir.glob("*.json"):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            for log in logs:
                # Filtrar por qualidade (assumindo que logs têm scores)
                if log.get("quality_score", 0) >= min_quality_score:
                    question = log.get("question", "")
                    answer = log.get("answer", "")
                    context = log.get("context", "")
                    
                    if question and answer:
                        if self.template == "alpaca":
                            text = self.format_alpaca(question, answer, context)
                        elif self.template == "chatml":
                            text = self.format_chatml(question, answer)
                        elif self.template == "llama2":
                            text = self.format_llama2(question, answer)
                        
                        formatted_data.append({"text": text})
        
        return Dataset.from_list(formatted_data)
    
    def create_train_test_split(
        self,
        dataset: Dataset,
        test_size: float = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """
        Divide dataset em treino e teste.
        
        Args:
            dataset: Dataset completo
            test_size: Proporção do teste (0-1)
            seed: Seed para reprodutibilidade
        
        Returns:
            DatasetDict com splits 'train' e 'test'
        """
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        return split
    
    def save_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        output_path: Union[str, Path]
    ):
        """
        Salva dataset em disco.
        
        Args:
            dataset: Dataset a salvar
            output_path: Caminho de saída
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        print(f"✅ Dataset salvo em: {output_path}")


def format_instruction_dataset(
    data: List[Dict[str, str]],
    template: str = "alpaca",
    system_message: Optional[str] = None
) -> Dataset:
    """
    Função helper para formatar rapidamente um dataset.
    
    Args:
        data: Lista de dicts com 'instruction' e 'output'
        template: Formato do prompt
        system_message: Mensagem de sistema opcional
    
    Returns:
        Dataset formatado
    
    Example:
        >>> data = [
        ...     {"instruction": "O que é Retail Media?", "output": "Retail Media é..."},
        ...     {"instruction": "Como funciona RTB?", "output": "RTB significa..."}
        ... ]
        >>> dataset = format_instruction_dataset(data)
    """
    preparator = DatasetPreparator(template=template, system_message=system_message)
    
    formatted_data = []
    for item in data:
        if template == "alpaca":
            text = preparator.format_alpaca(
                item["instruction"],
                item["output"],
                item.get("input", "")
            )
        elif template == "chatml":
            text = preparator.format_chatml(
                item["instruction"],
                item["output"]
            )
        elif template == "llama2":
            text = preparator.format_llama2(
                item["instruction"],
                item["output"]
            )
        else:
            raise ValueError(f"Template inválido: {template}")
        
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)


def create_sample_dataset(output_path: Union[str, Path]):
    """
    Cria dataset de exemplo para testes.
    
    Args:
        output_path: Onde salvar o dataset
    """
    sample_data = [
        {
            "instruction": "O que é Retail Media?",
            "output": "Retail Media é uma forma de publicidade digital onde varejistas monetizam seus sites e apps vendendo espaço publicitário para marcas. É um canal em crescimento que permite anunciantes alcançarem consumidores no momento da compra."
        },
        {
            "instruction": "Explique o que é RTB (Real-Time Bidding)",
            "output": "RTB (Real-Time Bidding) é um processo de compra e venda de anúncios em tempo real através de leilões automatizados. Quando um usuário acessa uma página, ocorre um leilão em milissegundos onde anunciantes dão lances para exibir seu anúncio. O maior lance vence e o anúncio é exibido instantaneamente."
        },
        {
            "instruction": "Quais são os principais formatos de anúncios digitais?",
            "output": "Os principais formatos de anúncios digitais incluem: Display (banners), Vídeo (pre-roll, mid-roll), Native (anúncios integrados ao conteúdo), Search (anúncios em buscas), Social (em redes sociais), e Rich Media (interativos). Cada formato tem suas vantagens dependendo do objetivo da campanha."
        },
        {
            "instruction": "O que significa CPM, CPC e CPA?",
            "output": "CPM (Custo Por Mil impressões) - custo para 1000 visualizações do anúncio. CPC (Custo Por Clique) - custo por cada clique no anúncio. CPA (Custo Por Aquisição) - custo por conversão/venda. Cada modelo é usado em diferentes estratégias de campanha dependendo do objetivo."
        },
        {
            "instruction": "Como funciona o targeting de audiência?",
            "output": "Targeting de audiência permite segmentar anúncios para grupos específicos baseado em: dados demográficos (idade, gênero), comportamento (histórico de navegação), interesses, localização geográfica, e contexto (conteúdo da página). Isso aumenta a relevância dos anúncios e melhora performance das campanhas."
        }
    ]
    
    dataset = format_instruction_dataset(sample_data, template="alpaca")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    
    print(f"✅ Dataset de exemplo criado com {len(dataset)} exemplos")
    print(f"📁 Salvo em: {output_path}")
    
    return dataset


if __name__ == "__main__":
    # Teste rápido
    print("🧪 Testando DatasetPreparator...")
    
    # Criar dataset de exemplo
    sample_dataset = create_sample_dataset(
        "/home/coneta/ai_agents_rag_melancia/data/finetuning/sample_dataset"
    )
    
    print("\n📊 Exemplo de dado formatado:")
    print(sample_dataset[0]["text"][:200] + "...")


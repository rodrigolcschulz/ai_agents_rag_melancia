"""
Model Evaluator - Classe para avaliar modelos fine-tunados

Implementa múltiplas métricas automáticas:
- ROUGE (overlap de n-gramas)
- BLEU (qualidade de geração)  
- BERTScore (similaridade semântica)
- Exact Match
- Perplexity (opcional)
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from rouge_score import rouge_scorer
    from bert_score import BERTScorer
    from sacrebleu.metrics import BLEU
except ImportError:
    print("⚠️ Instale: pip install rouge-score bert-score sacrebleu")


@dataclass
class EvaluationMetrics:
    """Métricas de avaliação para uma predição."""
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    bleu: float
    bertscore_f1: float
    bertscore_precision: float
    bertscore_recall: float
    exact_match: float
    length_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        """Converte para dicionário."""
        return {
            'rouge1_f': self.rouge1_f,
            'rouge2_f': self.rouge2_f,
            'rougeL_f': self.rougeL_f,
            'bleu': self.bleu,
            'bertscore_f1': self.bertscore_f1,
            'bertscore_precision': self.bertscore_precision,
            'bertscore_recall': self.bertscore_recall,
            'exact_match': self.exact_match,
            'length_ratio': self.length_ratio,
        }


class ModelEvaluator:
    """
    Avaliador de modelos fine-tunados.
    
    Implementa evaluation loops profissionais com múltiplas métricas automáticas.
    
    Exemplo:
        ```python
        evaluator = ModelEvaluator(model, tokenizer, "my-model")
        
        # Avaliar dataset completo
        results = evaluator.evaluate_dataset(test_data)
        
        # Gerar resposta única
        response = evaluator.generate_response("Como calcular ROAS?")
        ```
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        prompt_format: str = "alpaca"
    ):
        """
        Inicializa o evaluator.
        
        Args:
            model: Modelo do HuggingFace/PEFT
            tokenizer: Tokenizer correspondente
            model_name: Nome do modelo (para identificação)
            prompt_format: Formato do prompt ("alpaca", "chat", "plain")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.prompt_format = prompt_format
        
        # Inicializar scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.bleu_scorer = BLEU()
        self.bert_scorer = BERTScorer(
            lang='pt',
            rescale_with_baseline=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"✅ ModelEvaluator criado para: {model_name}")
    
    def format_prompt(self, prompt: str) -> str:
        """
        Formata prompt de acordo com o template.
        
        Args:
            prompt: Pergunta/instrução
            
        Returns:
            Prompt formatado
        """
        if self.prompt_format == "alpaca":
            return f"### Instruction:\n{prompt}\n\n### Response:\n"
        elif self.prompt_format == "chat":
            return f"<|user|>\n{prompt}\n<|assistant|>\n"
        else:  # plain
            return prompt
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Gera resposta para um prompt.
        
        Args:
            prompt: Pergunta/instrução
            max_new_tokens: Máximo de tokens para gerar
            temperature: Temperatura de amostragem
            top_p: Nucleus sampling
            do_sample: Usar amostragem ou greedy
            
        Returns:
            Resposta gerada
        """
        # Formatar prompt
        formatted_prompt = self.format_prompt(prompt)
        
        # Tokenizar
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Gerar
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decodificar
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrair apenas a resposta (remover prompt)
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()
        elif "<|assistant|>" in response:
            response = response.split("<|assistant|>")[1].strip()
        
        return response
    
    def calculate_metrics(
        self,
        prediction: str,
        reference: str
    ) -> EvaluationMetrics:
        """
        Calcula todas as métricas para uma predição vs referência.
        
        Args:
            prediction: Resposta gerada pelo modelo
            reference: Resposta de referência (ground truth)
            
        Returns:
            EvaluationMetrics com todas as métricas
        """
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, prediction)
        
        # BLEU score
        bleu = self.bleu_scorer.sentence_score(
            prediction,
            [reference]
        )
        
        # BERTScore
        P, R, F1 = self.bert_scorer.score(
            [prediction],
            [reference]
        )
        
        # Exact Match
        exact_match = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
        
        # Length ratio
        length_ratio = len(prediction) / max(len(reference), 1)
        
        return EvaluationMetrics(
            rouge1_f=rouge_scores['rouge1'].fmeasure,
            rouge2_f=rouge_scores['rouge2'].fmeasure,
            rougeL_f=rouge_scores['rougeL'].fmeasure,
            bleu=bleu.score / 100.0,  # Normalizar para 0-1
            bertscore_f1=F1.item(),
            bertscore_precision=P.item(),
            bertscore_recall=R.item(),
            exact_match=exact_match,
            length_ratio=length_ratio,
        )
    
    def evaluate_dataset(
        self,
        test_data: List[Dict],
        verbose: bool = True,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Avalia modelo em um dataset completo.
        
        Args:
            test_data: Lista de dicts com 'question' e 'reference_answer'
                       Opcionalmente pode ter 'category' para análise
            verbose: Mostrar progresso
            save_path: Caminho para salvar resultados (opcional)
        
        Returns:
            DataFrame com resultados detalhados
        """
        results = []
        
        for i, item in enumerate(test_data):
            if verbose:
                print(f"\r📊 Avaliando: {i+1}/{len(test_data)}", end="")
            
            question = item['question']
            reference = item['reference_answer']
            
            # Gerar resposta
            try:
                prediction = self.generate_response(question)
            except Exception as e:
                print(f"\n⚠️ Erro ao gerar resposta: {e}")
                prediction = ""
            
            # Calcular métricas
            metrics = self.calculate_metrics(prediction, reference)
            
            # Criar resultado
            result = {
                'model': self.model_name,
                'question': question,
                'prediction': prediction,
                'reference': reference,
                **metrics.to_dict()
            }
            
            # Adicionar categoria se existir
            if 'category' in item:
                result['category'] = item['category']
            
            # Adicionar ID se existir
            if 'id' in item:
                result['id'] = item['id']
            
            results.append(result)
        
        if verbose:
            print("\n✅ Avaliação concluída!")
        
        df = pd.DataFrame(results)
        
        # Salvar se solicitado
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"💾 Resultados salvos em: {save_path}")
        
        return df
    
    def compare_with_baseline(
        self,
        results_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compara resultados com um baseline.
        
        Args:
            results_df: Resultados do modelo atual
            baseline_df: Resultados do baseline
            metric_columns: Métricas para comparar
            
        Returns:
            DataFrame com comparação
        """
        if metric_columns is None:
            metric_columns = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'bertscore_f1']
        
        # Calcular médias
        current_means = results_df[metric_columns].mean()
        baseline_means = baseline_df[metric_columns].mean()
        
        # Calcular melhorias
        improvements = ((current_means - baseline_means) / baseline_means) * 100
        
        # Criar DataFrame de comparação
        comparison = pd.DataFrame({
            'metric': metric_columns,
            'baseline': baseline_means.values,
            'current': current_means.values,
            'improvement_%': improvements.values
        })
        
        return comparison
    
    def generate_report(
        self,
        results_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Gera relatório de avaliação.
        
        Args:
            results_df: Resultados da avaliação
            output_path: Caminho para salvar relatório JSON
            
        Returns:
            Dicionário com relatório
        """
        metric_columns = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'bertscore_f1']
        
        report = {
            'model_name': self.model_name,
            'total_samples': len(results_df),
            'metrics': {
                'mean': results_df[metric_columns].mean().to_dict(),
                'std': results_df[metric_columns].std().to_dict(),
                'min': results_df[metric_columns].min().to_dict(),
                'max': results_df[metric_columns].max().to_dict(),
            }
        }
        
        # Análise por categoria (se disponível)
        if 'category' in results_df.columns:
            report['by_category'] = {}
            for category in results_df['category'].unique():
                cat_data = results_df[results_df['category'] == category]
                report['by_category'][category] = {
                    'count': len(cat_data),
                    'metrics': cat_data[metric_columns].mean().to_dict()
                }
        
        # Salvar se solicitado
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Relatório salvo em: {output_path}")
        
        return report


def compare_models(
    evaluators: List[ModelEvaluator],
    test_data: List[Dict],
    output_dir: str = "./evaluation_results"
) -> pd.DataFrame:
    """
    Compara múltiplos modelos no mesmo dataset.
    
    Args:
        evaluators: Lista de ModelEvaluators
        test_data: Dataset de teste
        output_dir: Diretório para salvar resultados
        
    Returns:
        DataFrame com comparação agregada
    """
    all_results = []
    
    for evaluator in evaluators:
        print(f"\n🔄 Avaliando: {evaluator.model_name}")
        results = evaluator.evaluate_dataset(test_data, verbose=True)
        all_results.append(results)
    
    # Combinar resultados
    combined = pd.concat(all_results, ignore_index=True)
    
    # Salvar
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    combined.to_csv(f"{output_dir}/multi_model_comparison.csv", index=False)
    
    # Gerar comparação agregada
    metric_columns = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'bertscore_f1']
    comparison = combined.groupby('model')[metric_columns].mean()
    
    print("\n📊 COMPARAÇÃO DE MODELOS:")
    print("="*80)
    print(comparison.round(4))
    
    return comparison


"""
Sistema de tracking de experimentos usando MLflow
Rastreia métricas, parâmetros e artefatos de experimentos
"""
import mlflow
import mlflow.langchain
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Gerenciador de tracking de experimentos com MLflow
    
    Exemplo:
        tracker = ExperimentTracker("melancia-retail-media")
        
        with tracker.start_run("test-llama3"):
            tracker.log_params({"model": "llama3.1:8b", "temperature": 0.5})
            tracker.log_metrics({"latency": 1.2, "quality": 0.85})
            tracker.log_artifact("results.json")
    """
    
    def __init__(
        self,
        experiment_name: str = "melancia-retail-media",
        tracking_uri: Optional[str] = None
    ):
        """
        Args:
            experiment_name: Nome do experimento no MLflow
            tracking_uri: URI do servidor MLflow (None = local)
        """
        self.experiment_name = experiment_name
        
        # Configurar tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Usar diretório local
            mlruns_path = Path("mlruns").absolute()
            mlflow.set_tracking_uri(f"file://{mlruns_path}")
        
        # Criar/obter experimento
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment = mlflow.get_experiment(experiment_id)
        except Exception as e:
            logger.warning(f"Erro ao criar experimento: {e}")
            self.experiment = None
        
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
        logger.info(f"ExperimentTracker inicializado: {experiment_name}")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Inicia uma run de experimento
        
        Args:
            run_name: Nome da run (opcional)
            nested: Se é uma run aninhada
            
        Returns:
            Context manager para a run
        """
        return mlflow.start_run(run_name=run_name, nested=nested)
    
    def log_params(self, params: Dict[str, Any]):
        """
        Loga parâmetros do experimento
        
        Args:
            params: Dicionário de parâmetros
        """
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Erro ao logar parâmetro {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Loga métricas do experimento
        
        Args:
            metrics: Dicionário de métricas
            step: Step opcional para séries temporais
        """
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Erro ao logar métrica {key}: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_dir: Optional[str] = None):
        """
        Loga um arquivo como artefato
        
        Args:
            artifact_path: Caminho do arquivo
            artifact_dir: Diretório de destino no MLflow
        """
        try:
            if artifact_dir:
                mlflow.log_artifact(artifact_path, artifact_dir)
            else:
                mlflow.log_artifact(artifact_path)
        except Exception as e:
            logger.error(f"Erro ao logar artefato {artifact_path}: {e}")
    
    def log_model(self, model: Any, artifact_path: str = "model"):
        """
        Loga um modelo LangChain
        
        Args:
            model: Modelo a ser logado
            artifact_path: Caminho do artefato
        """
        try:
            mlflow.langchain.log_model(model, artifact_path)
        except Exception as e:
            logger.error(f"Erro ao logar modelo: {e}")
    
    def log_dict(self, dictionary: Dict, filename: str):
        """
        Loga um dicionário como JSON
        
        Args:
            dictionary: Dicionário a ser salvo
            filename: Nome do arquivo (sem extensão)
        """
        try:
            mlflow.log_dict(dictionary, f"{filename}.json")
        except Exception as e:
            logger.error(f"Erro ao logar dicionário: {e}")
    
    def log_text(self, text: str, filename: str):
        """
        Loga texto como artefato
        
        Args:
            text: Texto a ser salvo
            filename: Nome do arquivo
        """
        try:
            mlflow.log_text(text, filename)
        except Exception as e:
            logger.error(f"Erro ao logar texto: {e}")
    
    def log_benchmark_results(self, results: List[Dict[str, Any]]):
        """
        Loga resultados de benchmark
        
        Args:
            results: Lista de resultados de benchmark
        """
        # Converter para DataFrame para análise
        df = pd.DataFrame(results)
        
        # Métricas agregadas por modelo
        for model_name in df["model_name"].unique():
            model_df = df[df["model_name"] == model_name]
            
            metrics = {
                f"{model_name}_latency_avg": model_df["latency_seconds"].mean(),
                f"{model_name}_quality_avg": model_df["quality_score"].mean(),
                f"{model_name}_relevance_avg": model_df["relevance_score"].mean(),
                f"{model_name}_cost_total": model_df["cost_usd"].sum(),
            }
            
            self.log_metrics(metrics)
        
        # Salvar resultados completos
        self.log_dict({"results": results}, "benchmark_results")
    
    def get_best_run(self, metric: str = "quality_avg", ascending: bool = False) -> Optional[Dict]:
        """
        Busca a melhor run baseada em uma métrica
        
        Args:
            metric: Nome da métrica para comparar
            ascending: Se menor é melhor (True) ou maior é melhor (False)
            
        Returns:
            Dicionário com informações da melhor run
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )
            
            if not runs.empty:
                return runs.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Erro ao buscar melhor run: {e}")
        
        return None
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compara múltiplas runs
        
        Args:
            run_ids: Lista de IDs de runs para comparar
            
        Returns:
            DataFrame com comparação
        """
        try:
            runs_data = []
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                runs_data.append({
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    **run.data.params,
                    **run.data.metrics
                })
            
            return pd.DataFrame(runs_data)
            
        except Exception as e:
            logger.error(f"Erro ao comparar runs: {e}")
            return pd.DataFrame()
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo do experimento
        
        Returns:
            Dicionário com estatísticas do experimento
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id]
            )
            
            if runs.empty:
                return {"total_runs": 0}
            
            return {
                "total_runs": len(runs),
                "metrics_columns": [col for col in runs.columns if col.startswith("metrics.")],
                "params_columns": [col for col in runs.columns if col.startswith("params.")],
                "best_quality": runs["metrics.quality_avg"].max() if "metrics.quality_avg" in runs.columns else None,
                "avg_latency": runs["metrics.latency_avg"].mean() if "metrics.latency_avg" in runs.columns else None,
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar resumo: {e}")
            return {}
    
    def print_summary(self):
        """Imprime resumo do experimento"""
        summary = self.get_experiment_summary()
        
        print("\n" + "="*60)
        print(f"📊 Resumo do Experimento: {self.experiment_name}")
        print("="*60)
        print(f"Total de Runs: {summary.get('total_runs', 0)}")
        
        if summary.get('best_quality'):
            print(f"Melhor Qualidade: {summary['best_quality']:.3f}")
        
        if summary.get('avg_latency'):
            print(f"Latência Média: {summary['avg_latency']:.3f}s")
        
        print("="*60 + "\n")
    
    @staticmethod
    def launch_ui(port: int = 5000):
        """
        Lança interface web do MLflow
        
        Args:
            port: Porta para o servidor web
        """
        import subprocess
        import webbrowser
        
        print(f"🚀 Iniciando MLflow UI em http://localhost:{port}")
        print("   (Pressione Ctrl+C para parar)")
        
        try:
            # Abrir navegador após 2 segundos
            import threading
            threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
            
            # Iniciar servidor
            subprocess.run(["mlflow", "ui", "--port", str(port)])
        except KeyboardInterrupt:
            print("\n✓ MLflow UI encerrado")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Teste básico
    tracker = ExperimentTracker()
    
    # Simular experimento
    with tracker.start_run("test-run"):
        tracker.log_params({
            "model": "gpt-4o-mini",
            "temperature": 0.5,
            "provider": "openai"
        })
        
        tracker.log_metrics({
            "latency_avg": 1.5,
            "quality_avg": 0.85,
            "cost_total": 0.05
        })
        
        print("✓ Experimento de teste logado com sucesso!")
    
    # Mostrar resumo
    tracker.print_summary()
    
    print("\n💡 Para visualizar os experimentos, execute:")
    print("   python -m src.mlops.tracking")


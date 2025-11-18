"""
Registro de modelos para versionamento e deployment
"""
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Gerenciador de registro de modelos
    
    Exemplo:
        registry = ModelRegistry()
        
        # Registrar novo modelo
        registry.register_model("runs:/abc123/model", "melancia-llm")
        
        # Promover para produção
        registry.promote_to_production("melancia-llm", version=2)
    """
    
    def __init__(self):
        self.client = MlflowClient()
        logger.info("ModelRegistry inicializado")
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Registra um modelo no registry
        
        Args:
            model_uri: URI do modelo (ex: runs:/run_id/model)
            name: Nome do modelo
            tags: Tags adicionais
            
        Returns:
            Versão do modelo registrado
        """
        try:
            result = mlflow.register_model(model_uri, name)
            
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name, result.version, key, value
                    )
            
            logger.info(f"Modelo registrado: {name} v{result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Erro ao registrar modelo: {e}")
            raise
    
    def promote_to_production(self, name: str, version: int):
        """
        Promove um modelo para produção
        
        Args:
            name: Nome do modelo
            version: Versão a promover
        """
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage="Production"
            )
            logger.info(f"Modelo {name} v{version} promovido para produção")
        except Exception as e:
            logger.error(f"Erro ao promover modelo: {e}")
            raise
    
    def get_production_model(self, name: str) -> Optional[str]:
        """
        Retorna URI do modelo em produção
        
        Args:
            name: Nome do modelo
            
        Returns:
            URI do modelo ou None
        """
        try:
            versions = self.client.get_latest_versions(name, stages=["Production"])
            if versions:
                return f"models:/{name}/Production"
            return None
        except Exception as e:
            logger.error(f"Erro ao buscar modelo em produção: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """Lista todos os modelos registrados"""
        try:
            models = self.client.search_registered_models()
            return [model.name for model in models]
        except Exception as e:
            logger.error(f"Erro ao listar modelos: {e}")
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    registry = ModelRegistry()
    models = registry.list_models()
    print(f"Modelos registrados: {models}")


"""
Sistema de roteamento inteligente entre múltiplos LLMs
Decide qual modelo usar baseado em contexto, tier do usuário, e custo-benefício
"""
import time
import random
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

from src.experiments.multi_llm import MultiLLMManager
from src.mlops.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Decisão de roteamento"""
    provider: str
    model_name: str
    reason: str
    estimated_cost: float
    estimated_latency: float


class FeatureFlags:
    """
    Controla features em produção sem necessidade de deploy
    Permite A/B testing e rollout gradual
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Configurações personalizadas (opcional)
        """
        # Configurações padrão
        self.flags = {
            "use_ollama": True,              # Habilitar Ollama
            "use_openai": True,              # Habilitar OpenAI
            "enable_caching": True,          # Cache de respostas
            "a_b_test_active": True,         # A/B testing ativo
            "ollama_percentage": 0.8,        # 80% Ollama, 20% OpenAI
            "enable_monitoring": True,       # Monitoring de performance
            "enable_fallback": True,         # Fallback se modelo falhar
            "max_latency_ollama": 180.0,     # Timeout Ollama (segundos)
            "max_latency_openai": 30.0,      # Timeout OpenAI (segundos)
        }
        
        # Sobrescrever com config customizado
        if config:
            self.flags.update(config)
    
    def should_use_ollama(self, user_id: int) -> bool:
        """
        Decide se deve usar Ollama para este usuário
        Usa hash consistente para A/B testing
        
        Args:
            user_id: ID do usuário
            
        Returns:
            True se deve usar Ollama
        """
        if not self.flags["use_ollama"]:
            return False
        
        if not self.flags["a_b_test_active"]:
            return True  # Sempre Ollama se não for A/B test
        
        # Hash consistente: mesmo usuário sempre terá mesma experiência
        hash_val = hash(user_id) % 100
        threshold = int(self.flags["ollama_percentage"] * 100)
        
        return hash_val < threshold
    
    def get_timeout(self, provider: str) -> float:
        """Retorna timeout apropriado para o provider"""
        return self.flags.get(f"max_latency_{provider}", 30.0)


class ModelRouter:
    """
    Roteador inteligente de queries entre múltiplos LLMs
    
    Estratégias de roteamento:
    - Tier do usuário (free vs premium)
    - Complexidade da query
    - Custo vs performance
    - A/B testing
    - Feature flags
    
    Exemplo:
        router = ModelRouter()
        
        # Roteamento automático
        response = router.route_query(
            question="O que é Retail Media?",
            user_tier="free",
            user_id=123
        )
        
        # Forçar modelo específico
        response = router.route_query(
            question="Pergunta importante",
            force_provider="openai"
        )
    """
    
    def __init__(
        self,
        feature_flags: Optional[FeatureFlags] = None,
        enable_tracking: bool = True
    ):
        """
        Args:
            feature_flags: Configurações de features (opcional)
            enable_tracking: Se deve trackear no MLflow
        """
        self.feature_flags = feature_flags or FeatureFlags()
        self.enable_tracking = enable_tracking
        
        # Inicializar tracker
        if enable_tracking:
            self.tracker = ExperimentTracker("melancia-production")
        
        # Cache de modelos (lazy loading)
        self._ollama_model: Optional[OllamaLLM] = None
        self._openai_model: Optional[ChatOpenAI] = None
        
        # Estatísticas
        self.stats = {
            "total_queries": 0,
            "ollama_queries": 0,
            "openai_queries": 0,
            "fallbacks": 0,
            "errors": 0,
        }
        
        logger.info("ModelRouter inicializado")
    
    @property
    def ollama_model(self) -> OllamaLLM:
        """Lazy loading do modelo Ollama"""
        if self._ollama_model is None:
            logger.info("Carregando modelo Ollama otimizado...")
            self._ollama_model = MultiLLMManager.create_llm(
                "ollama",
                "llama3.2:3b",  # Mais rápido e eficiente que phi3:mini
                temperature=0.3,  # Baixa temperatura = mais focado no contexto
                max_tokens=500,   # Respostas mais concisas
            )
        return self._ollama_model
    
    @property
    def openai_model(self) -> ChatOpenAI:
        """Lazy loading do modelo OpenAI"""
        if self._openai_model is None:
            logger.info("Carregando modelo OpenAI...")
            self._openai_model = MultiLLMManager.create_llm(
                "openai",
                "gpt-4o-mini"
            )
        return self._openai_model
    
    def decide_routing(
        self,
        question: str,
        user_tier: str = "free",
        user_id: Optional[int] = None,
        force_provider: Optional[str] = None
    ) -> RoutingDecision:
        """
        Decide qual modelo usar
        
        Args:
            question: Pergunta do usuário
            user_tier: Tier do usuário (free, premium)
            user_id: ID do usuário (para A/B testing consistente)
            force_provider: Forçar provider específico (ollama/openai)
            
        Returns:
            RoutingDecision com provider escolhido e razão
        """
        # 1. Se forçar provider, usar ele
        if force_provider:
            return RoutingDecision(
                provider=force_provider,
                model_name="llama3.2:3b" if force_provider == "ollama" else "gpt-4o-mini",
                reason="forced_by_user",
                estimated_cost=0.0 if force_provider == "ollama" else 0.0001,
                estimated_latency=5.0 if force_provider == "ollama" else 4.0  # llama3.2:3b é rápido
            )
        
        # 2. Usuários premium sempre OpenAI
        if user_tier == "premium":
            return RoutingDecision(
                provider="openai",
                model_name="gpt-4o-mini",
                reason="premium_user",
                estimated_cost=0.0001,
                estimated_latency=4.0
            )
        
        # 3. Perguntas muito simples → OpenAI (mais rápido)
        word_count = len(question.split())
        if word_count < 5:
            return RoutingDecision(
                provider="openai",
                model_name="gpt-4o-mini",
                reason="simple_question",
                estimated_cost=0.0001,
                estimated_latency=4.0
            )
        
        # 4. A/B testing baseado em user_id
        if user_id and self.feature_flags.should_use_ollama(user_id):
            return RoutingDecision(
                provider="ollama",
                model_name="llama3.2:3b",
                reason="a_b_test_ollama",
                estimated_cost=0.0,
                estimated_latency=5.0  # llama3.2:3b é muito mais rápido
            )
        
        # 5. Fallback para OpenAI
        return RoutingDecision(
            provider="openai",
            model_name="gpt-4o-mini",
            reason="default_openai",
            estimated_cost=0.0001,
            estimated_latency=4.0
        )
    
    def route_query(
        self,
        question: str,
        user_tier: str = "free",
        user_id: Optional[int] = None,
        force_provider: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Roteia query para o modelo apropriado e executa
        
        Args:
            question: Pergunta do usuário
            user_tier: Tier do usuário (free, premium)
            user_id: ID do usuário
            force_provider: Forçar provider específico
            timeout: Timeout customizado (segundos)
            
        Returns:
            Dicionário com resposta e metadados
        """
        self.stats["total_queries"] += 1
        
        # Decidir roteamento
        decision = self.decide_routing(
            question=question,
            user_tier=user_tier,
            user_id=user_id,
            force_provider=force_provider
        )
        
        logger.info(
            f"Roteando para {decision.provider}::{decision.model_name} "
            f"(motivo: {decision.reason})"
        )
        
        # Selecionar modelo
        if decision.provider == "ollama":
            model = self.ollama_model
            self.stats["ollama_queries"] += 1
        else:
            model = self.openai_model
            self.stats["openai_queries"] += 1
        
        # Timeout
        timeout = timeout or self.feature_flags.get_timeout(decision.provider)
        
        # Executar query
        result = self._execute_query(
            model=model,
            question=question,
            decision=decision,
            timeout=timeout
        )
        
        return result
    
    def _execute_query(
        self,
        model: Union[ChatOpenAI, OllamaLLM],
        question: str,
        decision: RoutingDecision,
        timeout: float
    ) -> Dict[str, Any]:
        """
        Executa query no modelo com timeout e fallback
        
        Args:
            model: Modelo LLM
            question: Pergunta
            decision: Decisão de roteamento
            timeout: Timeout em segundos
            
        Returns:
            Dicionário com resposta e metadados
        """
        start_time = time.time()
        
        try:
            # Invocar modelo com timeout
            # Nota: timeout real precisa ser implementado no wrapper do modelo
            response = model.invoke(question)
            
            latency = time.time() - start_time
            
            # Verificar se passou do timeout
            if latency > timeout:
                logger.warning(
                    f"Query excedeu timeout: {latency:.2f}s > {timeout}s"
                )
                
                # Se fallback ativo e não for OpenAI, tentar OpenAI
                if (self.feature_flags.flags["enable_fallback"] and 
                    decision.provider != "openai"):
                    
                    logger.info("Executando fallback para OpenAI...")
                    self.stats["fallbacks"] += 1
                    
                    return self._execute_query(
                        model=self.openai_model,
                        question=question,
                        decision=RoutingDecision(
                            provider="openai",
                            model_name="gpt-4o-mini",
                            reason="fallback_timeout",
                            estimated_cost=0.0001,
                            estimated_latency=4.0
                        ),
                        timeout=30.0
                    )
            
            # Sucesso
            result = {
                "answer": response,
                "provider": decision.provider,
                "model_name": decision.model_name,
                "latency_seconds": latency,
                "routing_reason": decision.reason,
                "estimated_cost": decision.estimated_cost,
                "success": True,
                "fallback_used": False
            }
            
            # Trackear no MLflow
            if self.enable_tracking:
                self._track_query(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao executar query: {e}")
            self.stats["errors"] += 1
            
            latency = time.time() - start_time
            
            # Fallback em caso de erro
            if (self.feature_flags.flags["enable_fallback"] and 
                decision.provider != "openai"):
                
                logger.info("Executando fallback para OpenAI após erro...")
                self.stats["fallbacks"] += 1
                
                try:
                    return self._execute_query(
                        model=self.openai_model,
                        question=question,
                        decision=RoutingDecision(
                            provider="openai",
                            model_name="gpt-4o-mini",
                            reason="fallback_error",
                            estimated_cost=0.0001,
                            estimated_latency=4.0
                        ),
                        timeout=30.0
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback também falhou: {fallback_error}")
            
            # Retornar erro
            return {
                "answer": None,
                "provider": decision.provider,
                "model_name": decision.model_name,
                "latency_seconds": latency,
                "routing_reason": decision.reason,
                "estimated_cost": 0.0,
                "success": False,
                "fallback_used": False,
                "error": str(e)
            }
    
    def _track_query(self, result: Dict[str, Any]):
        """Trackeia query no MLflow"""
        try:
            with self.tracker.start_run(nested=True):
                self.tracker.log_params({
                    "provider": result["provider"],
                    "model_name": result["model_name"],
                    "routing_reason": result["routing_reason"]
                })
                
                self.tracker.log_metrics({
                    "latency": result["latency_seconds"],
                    "cost": result["estimated_cost"],
                    "success": 1.0 if result["success"] else 0.0
                })
        except Exception as e:
            logger.warning(f"Erro ao trackear query: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de uso
        
        Returns:
            Dicionário com estatísticas
        """
        total = self.stats["total_queries"]
        
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "ollama_percentage": (self.stats["ollama_queries"] / total) * 100,
            "openai_percentage": (self.stats["openai_queries"] / total) * 100,
            "fallback_rate": (self.stats["fallbacks"] / total) * 100,
            "error_rate": (self.stats["errors"] / total) * 100,
        }
    
    def print_stats(self):
        """Imprime estatísticas formatadas"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 Estatísticas do Model Router")
        print("="*60)
        print(f"Total de Queries: {stats['total_queries']}")
        print(f"  🦙 Ollama: {stats['ollama_queries']} ({stats.get('ollama_percentage', 0):.1f}%)")
        print(f"  🤖 OpenAI: {stats['openai_queries']} ({stats.get('openai_percentage', 0):.1f}%)")
        print(f"  🔄 Fallbacks: {stats['fallbacks']} ({stats.get('fallback_rate', 0):.1f}%)")
        print(f"  ⚠️  Erros: {stats['errors']} ({stats.get('error_rate', 0):.1f}%)")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Teste básico
    print("🧪 Testando Model Router...\n")
    
    # Criar router
    router = ModelRouter(enable_tracking=False)
    
    # Teste 1: Usuário free (deve usar Ollama por A/B test)
    print("1. Teste usuário free (A/B test):")
    decision = router.decide_routing(
        question="O que é Retail Media?",
        user_tier="free",
        user_id=123
    )
    print(f"   Decisão: {decision.provider} - {decision.reason}\n")
    
    # Teste 2: Usuário premium (deve usar OpenAI)
    print("2. Teste usuário premium:")
    decision = router.decide_routing(
        question="O que é Retail Media?",
        user_tier="premium"
    )
    print(f"   Decisão: {decision.provider} - {decision.reason}\n")
    
    # Teste 3: Pergunta simples (deve usar OpenAI pela velocidade)
    print("3. Teste pergunta simples:")
    decision = router.decide_routing(
        question="Oi",
        user_tier="free"
    )
    print(f"   Decisão: {decision.provider} - {decision.reason}\n")
    
    # Teste 4: Forçar Ollama
    print("4. Teste forçar Ollama:")
    decision = router.decide_routing(
        question="Pergunta complexa",
        force_provider="ollama"
    )
    print(f"   Decisão: {decision.provider} - {decision.reason}\n")
    
    print("✅ Testes concluídos!")
    print("\n💡 Para usar em produção:")
    print("   router = ModelRouter()")
    print("   response = router.route_query('Sua pergunta', user_tier='free', user_id=123)")


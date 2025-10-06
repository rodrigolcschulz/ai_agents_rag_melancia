import os
import re
from datetime import datetime
from keywords import CONTEXT_KEYWORDS
import config

def garantir_pasta_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)

def registrar_log(pergunta, resposta, log_path):
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{agora}] P: {pergunta}\nR: {resposta}\n{'-'*50}\n")

def normalizar_pergunta(texto):
    """Normaliza o texto removendo pontuação e convertendo para minúsculas"""
    texto = re.sub(r"[^\w\s]", "", texto.lower())
    return texto.strip()

# Contexto fixo
INFO_ML = "ML = Mercado Livre"
INFO_JOEL = "Joel Jonathan Cunha é CEO da Conecta Ads"

def is_relevant(pergunta: str) -> bool:
    """
    Verifica se a pergunta é relevante para o contexto de Retail Media
    Agora inclui logística quando relacionada a e-commerce/publicidade
    """
    pergunta_normalizada = normalizar_pergunta(pergunta)
    
    # Lista de palavras que indicam contexto comercial/publicitário
    contexto_comercial = [
        "publicidade", "anúncios", "vendas", "e-commerce", "marketplace",
        "campanha", "performance", "conversão", "negócio", "cliente"
    ]
    
    # Se a pergunta contém palavras-chave diretas, é relevante
    if any(keyword.lower() in pergunta_normalizada for keyword in CONTEXT_KEYWORDS):
        return True
    
    # Se contém "envio" ou "logística" junto com contexto comercial, também é relevante
    palavras_logistica = ["envio", "entrega", "logística", "prazo", "frete"]
    tem_logistica = any(palavra in pergunta_normalizada for palavra in palavras_logistica)
    tem_contexto_comercial = any(palavra in pergunta_normalizada for palavra in contexto_comercial)
    
    if tem_logistica and tem_contexto_comercial:
        return True
    
    return False
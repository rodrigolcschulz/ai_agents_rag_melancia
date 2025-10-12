CONTEXT_KEYWORDS = [
    # Retail Media e Publicidade
    "retail media", "publicidade", "mercado ads", "anúncios", "performance",
    "marketing digital", "mídia paga", "advertising",
    
    # E-commerce e Marketplaces
    "e-commerce", "ecommerce", "marketplace", "mercado livre", "ml", "meli",
    "e commerce",
    
    # Tipos de Anúncios
    "product ads", "pads", "produtos patrocinados", "anúncios patrocinados",
    "sponsored products", "display ads", "banner ads", "vitrine",
    
    # Métricas e Performance
    "acos", "tacos", "roas", "ctr", "cpc", "cpm", "taxa de conversão",
    "conversion rate", "impressões", "cliques", "roi", "revenue",
    
    # Estratégias
    "vendas", "estratégia", "estratégias", "campanha", "campanhas",
    "otimização", "segmentação", "palavra-chave", "keywords", "bid", "lance",
    "tráfego pago", "budget", "orçamento",
    
    # Conteúdo e Criativos
    "clips", "vídeos curtos", "creative", "criativo", "storytelling",
    "imagem", "foto", "descrição", "título",
    
    # Logística (ADICIONADO - relacionado ao seu caso)
    "envio", "entrega", "logística", "prazo", "frete", "fulfillment",
    "estoque", "disponibilidade", "full", "centro de distribuição"
]

# Função melhorada para detectar relevância
def is_context_relevant(pergunta: str, keywords_list: list) -> bool:
    """
    Verifica se a pergunta contém pelo menos uma palavra-chave do contexto
    """
    pergunta_normalizada = pergunta.lower()
    
    # Verifica se pelo menos uma keyword está presente
    for keyword in keywords_list:
        if keyword.lower() in pergunta_normalizada:
            return True
    
    return False
CONTEXT_KEYWORDS = [
    # === RETAIL MEDIA E PUBLICIDADE ===
    "retail media", "publicidade", "mercado ads", "anúncios", "performance",
    "marketing digital", "mídia paga", "advertising", "publicidade digital",
    "mídia programática", "programática", "adtech",
    
    # === E-COMMERCE E MARKETPLACES ===
    "e-commerce", "ecommerce", "marketplace", "mercado livre", "ml", "meli",
    "e commerce", "shopee", "amazon", "magalu", "b2w",
    
    # === TIPOS DE ANÚNCIOS ===
    "product ads", "pads", "produtos patrocinados", "anúncios patrocinados",
    "sponsored products", "display ads", "banner ads", "vitrine",
    
    # === MÉTRICAS E PERFORMANCE ===
    "acos", "tacos", "roas", "ctr", "cpc", "cpm", "roi", "revenue",
    "taxa de conversão", "conversion rate", "conversão",
    "impressões", "cliques", "visibilidade", "ranking", "posicionamento",
    
    # === ESTRATÉGIAS DE MARKETING ===
    "vendas", "estratégia", "estratégias", "campanha", "campanhas",
    "otimização", "segmentação", "palavra-chave", "keywords", "palavras chave",
    "bid", "lance", "lances", "tráfego pago", "budget", "orçamento",
    "full funnel", "jornada", "jornada compra", "público-alvo", "público alvo",
    "targeting", "público", "concorrência", "benchmark", "competitivo",
    
    # === CONTEÚDO VISUAL E CRIATIVOS ===
    "clips", "vídeos curtos", "vídeos", "vídeo", "creative", "criativo",
    "storytelling", "fotos", "foto", "imagens", "imagem",
    "descrição", "título", "editor fotos", "gestor fotos",
    
    # === LOGÍSTICA E ENVIOS ===
    "envio", "entrega", "logística", "prazo", "frete", "fulfillment",
    "envio flex", "envios flex", "full", "mercado envios", "me1",
    "mesmo dia", "fim semana", "cobertura", "centro de distribuição",
    "tabela contingência", "devolução",
    
    # === GESTÃO DE ESTOQUE ===
    "estoque", "disponibilidade", "ruptura", "inventário",
    
    # === CATÁLOGO E PRODUTOS ===
    "autopeças", "auto peças", "categoria", "categorias", "produto", "produtos",
    "publicação", "publicações", "anúncio", "listing", "catálogo",
    "ficha técnica", "características", "atributos", "especificações",
    "código universal", "sku", "ean", "gtin", "variações", "variação",
    
    # === FERRAMENTAS E RECURSOS ===
    "ferramenta", "ferramentas", "recurso", "funcionalidade",
    "excel", "editor", "planilha", "anunciador massa",
    "modo férias", "pausado", "sincronizado",
    
    # === COMPATIBILIDADE (AUTOPEÇAS) ===
    "compatibilidade", "compatibilidades", "veículo", "veículos",
    "carro", "carros", "marca veículo", "modelo veículo",
    
    # === BUSCA E SEO ===
    "busca", "pesquisa", "seo", "resultados busca", "algoritmo",
    
    # === ATORES DO MARKETPLACE ===
    "comprador", "compradores", "vendedor", "vendedores",
    "cliente", "clientes", "experiência",
    
    # === REPUTAÇÃO E ATENDIMENTO ===
    "reputação", "reclamações", "avaliações", "feedback", "qualidade",
    "mensagens", "atendimento", "resposta", "pré-venda", "denúncia",
    "tempo resposta", "resposta automática",
    
    # === BRANDING E MARCA ===
    "loja oficial", "branding", "marca", "inpi", "registro marca",
    
    # === PRECIFICAÇÃO E FINANCEIRO ===
    "preço", "precificar", "simulador", "custos", "tarifas", "taxas",
    "mercado pago", "crédito", "empréstimo", "pagamento",
    "fluxo financeiro", "dinheiro", "faturamento",
    
    # === EVENTOS E SAZONALIDADE ===
    "black friday", "sazonalidade", "datas comemorativas",
    
    # === TECNOLOGIA E DADOS ===
    "first party data", "dados próprios", "privacidade", "dados",
    "inteligência artificial", "ia", "automação",
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
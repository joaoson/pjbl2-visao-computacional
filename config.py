"""
Configurações do projeto.

Todos os parâmetros que podem ser ajustados ficam aqui, em um só lugar.
Os scripts das etapas apenas importam estes valores.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Pastas do projeto
# ---------------------------------------------------------------------------
PASTA_PROJETO = Path(__file__).parent

PASTA_DADOS = PASTA_PROJETO / "dados"
PASTA_IMAGENS_ORIGINAIS = PASTA_DADOS / "originais"  # dados/originais/<classe>/*.jpg
PASTA_IMAGENS_DIVIDIDAS = PASTA_DADOS / "divididos"  # dados/divididos/<conjunto>/<classe>/*.jpg

PASTA_RESULTADOS = PASTA_PROJETO / "resultados"
CAMINHO_MODELO_TREINADO = PASTA_RESULTADOS / "modelo_treinado.pt"
CAMINHO_HISTORICO_TREINO = PASTA_RESULTADOS / "historico_treinamento.json"

# ---------------------------------------------------------------------------
# Etapa 1 — Base de imagens
# ---------------------------------------------------------------------------
# Subconjunto do Food-101 com 3 categorias de comida (fotos reais, ~200 por classe).
URL_DATASET = (
    "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/"
    "pizza_steak_sushi_20_percent.zip"
)

# Nome da pasta no arquivo baixado -> nome da classe em português
TRADUCAO_CLASSES = {
    "pizza": "pizza",
    "steak": "bife",
    "sushi": "sushi",
}

# ---------------------------------------------------------------------------
# Etapa 2 — Protocolo experimental (divisão treino / validação / teste)
# ---------------------------------------------------------------------------
PROPORCAO_TREINO = 0.70
PROPORCAO_VALIDACAO = 0.15
PROPORCAO_TESTE = 0.15

# Semente aleatória fixa: garante que a divisão e o treino sejam reproduzíveis.
SEMENTE = 42

# ---------------------------------------------------------------------------
# Etapa 3 — Modelo e treinamento
# ---------------------------------------------------------------------------
TAMANHO_IMAGEM = 224  # tamanho de entrada esperado pela ResNet-18
TAMANHO_LOTE = 32  # quantas imagens por passo de treinamento (batch size)
NUMERO_EPOCAS = 25
TAXA_APRENDIZADO = 0.001

# Para o treino se nenhuma melhora na validação ocorrer por N épocas seguidas.
PACIENCIA_PARADA_ANTECIPADA = 4

# Média e desvio padrão das cores do ImageNet (base em que a ResNet foi pré-treinada).
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESVIO_IMAGENET = [0.229, 0.224, 0.225]

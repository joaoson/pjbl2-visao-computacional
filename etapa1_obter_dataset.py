"""
ETAPA 1 — Busca da base de imagens.

Baixa um subconjunto do Food-101 com 3 categorias (pizza, bife e sushi) e
organiza todas as imagens em uma pasta por classe:

    dados/originais/
        pizza/  *.jpg
        bife/   *.jpg
        sushi/  *.jpg

O arquivo baixado já vem separado em "train" e "test", mas aqui juntamos tudo:
a divisão em treino/validação/teste é feita por nós na Etapa 2.

Para usar outra base de imagens, basta pular esta etapa e colocar as fotos
manualmente em dados/originais/<nome_da_classe>/.
"""

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import config
from utils.graficos import salvar_grade_de_amostras


def baixar_e_extrair(url: str, pasta_destino: Path) -> None:
    """Baixa o arquivo .zip da URL e extrai seu conteúdo na pasta de destino."""
    caminho_zip = pasta_destino / "dataset.zip"

    print(f"Baixando {url} ...")
    urllib.request.urlretrieve(url, caminho_zip)

    print("Extraindo arquivos...")
    with zipfile.ZipFile(caminho_zip) as arquivo_zip:
        arquivo_zip.extractall(pasta_destino)


def organizar_por_classe(pasta_extraida: Path) -> None:
    """Copia as imagens de train/ e test/ para dados/originais/<classe>/."""
    for nome_original, nome_classe in config.TRADUCAO_CLASSES.items():
        pasta_classe = config.PASTA_IMAGENS_ORIGINAIS / nome_classe
        pasta_classe.mkdir(parents=True, exist_ok=True)

        imagens = sorted(pasta_extraida.glob(f"*/{nome_original}/*.jpg"))
        for imagem in imagens:
            shutil.copy(imagem, pasta_classe / imagem.name)

        print(f"  {nome_classe:<6} -> {len(imagens)} imagens")


def main() -> None:
    if config.PASTA_IMAGENS_ORIGINAIS.exists():
        print(f"A pasta {config.PASTA_IMAGENS_ORIGINAIS} já existe. Nada a fazer.")
    else:
        with tempfile.TemporaryDirectory() as pasta_temporaria:
            pasta_temporaria = Path(pasta_temporaria)
            baixar_e_extrair(config.URL_DATASET, pasta_temporaria)
            print("Organizando imagens por classe:")
            organizar_por_classe(pasta_temporaria)

    caminho_figura = config.PASTA_RESULTADOS / "1_amostras_do_dataset.png"
    salvar_grade_de_amostras(config.PASTA_IMAGENS_ORIGINAIS, caminho_figura)
    print(f"Exemplos de imagens salvos em {caminho_figura}")


if __name__ == "__main__":
    main()

"""Funções que geram as figuras salvas na pasta resultados/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos de imagem sem abrir janelas

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay

from utils.dados import listar_imagens_por_classe

NOMES_DOS_CONJUNTOS = {"treino": "treino", "validacao": "validação", "teste": "teste"}


def _abrir_quadrada(caminho: Path, lado: int = 256) -> Image.Image:
    """Abre a imagem recortada no centro em formato quadrado, para as grades ficarem alinhadas."""
    return ImageOps.fit(Image.open(caminho).convert("RGB"), (lado, lado))


def _salvar_figura(figura: plt.Figure, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.tight_layout()
    figura.savefig(caminho, dpi=120)
    plt.close(figura)


def salvar_grade_de_amostras(pasta_imagens: Path, caminho: Path, imagens_por_classe: int = 5) -> None:
    """Etapa 1: mostra algumas imagens de cada classe, uma classe por linha."""
    caminhos, classes = listar_imagens_por_classe(pasta_imagens)
    nomes_classes = sorted(set(classes))

    figura, eixos = plt.subplots(
        len(nomes_classes), imagens_por_classe, figsize=(2.2 * imagens_por_classe, 2.4 * len(nomes_classes))
    )
    for linha, nome_classe in enumerate(nomes_classes):
        imagens_da_classe = [c for c, classe in zip(caminhos, classes) if classe == nome_classe]
        for coluna in range(imagens_por_classe):
            eixo = eixos[linha, coluna]
            eixo.imshow(_abrir_quadrada(imagens_da_classe[coluna]))
            eixo.axis("off")
        eixos[linha, 0].set_title(nome_classe, loc="left", fontsize=12, fontweight="bold")

    figura.suptitle("Exemplos de imagens da base")
    _salvar_figura(figura, caminho)


def salvar_grafico_da_divisao(contagem: dict[str, dict[str, int]], nomes_classes: list[str], caminho: Path) -> None:
    """Etapa 2: gráfico de barras com a quantidade de imagens por classe em cada conjunto."""
    nomes_conjuntos = list(contagem.keys())
    posicoes = np.arange(len(nomes_conjuntos))
    largura_barra = 0.8 / len(nomes_classes)

    figura, eixo = plt.subplots(figsize=(7, 4))
    for i, nome_classe in enumerate(nomes_classes):
        quantidades = [contagem[conjunto][nome_classe] for conjunto in nomes_conjuntos]
        barras = eixo.bar(posicoes + i * largura_barra, quantidades, largura_barra, label=nome_classe)
        eixo.bar_label(barras)

    rotulos = [NOMES_DOS_CONJUNTOS[conjunto] for conjunto in nomes_conjuntos]
    eixo.set_xticks(posicoes + largura_barra * (len(nomes_classes) - 1) / 2, rotulos)
    eixo.set_ylabel("número de imagens")
    eixo.set_title("Divisão estratificada da base (70% / 15% / 15%)")
    eixo.legend()
    _salvar_figura(figura, caminho)


def salvar_curvas_de_treinamento(historico: dict[str, list[float]], caminho: Path) -> None:
    """Etapa 3: evolução da perda e da acurácia em treino e validação ao longo das épocas."""
    epocas = range(1, len(historico["perda_treino"]) + 1)

    figura, (eixo_perda, eixo_acuracia) = plt.subplots(1, 2, figsize=(11, 4))

    eixo_perda.plot(epocas, historico["perda_treino"], "o-", label="treino")
    eixo_perda.plot(epocas, historico["perda_validacao"], "o-", label="validação")
    eixo_perda.set_title("Perda (quanto menor, melhor)")
    eixo_perda.set_xlabel("época")
    eixo_perda.legend()

    eixo_acuracia.plot(epocas, historico["acuracia_treino"], "o-", label="treino")
    eixo_acuracia.plot(epocas, historico["acuracia_validacao"], "o-", label="validação")
    eixo_acuracia.set_title("Acurácia (quanto maior, melhor)")
    eixo_acuracia.set_xlabel("época")
    eixo_acuracia.set_ylim(0, 1)
    eixo_acuracia.legend()

    _salvar_figura(figura, caminho)


def salvar_matriz_de_confusao(matriz: np.ndarray, nomes_classes: list[str], caminho: Path) -> None:
    """Etapa 4: matriz de confusão em quantidade e em porcentagem de cada classe real."""
    matriz_percentual = matriz / matriz.sum(axis=1, keepdims=True)

    figura, (eixo_quantidade, eixo_percentual) = plt.subplots(1, 2, figsize=(11, 4.5))

    ConfusionMatrixDisplay(matriz, display_labels=nomes_classes).plot(
        ax=eixo_quantidade, cmap="Blues", colorbar=False
    )
    eixo_quantidade.set_title("Quantidade de imagens")

    ConfusionMatrixDisplay(matriz_percentual, display_labels=nomes_classes).plot(
        ax=eixo_percentual, cmap="Blues", colorbar=False, values_format=".0%"
    )
    eixo_percentual.set_title("Porcentagem por classe real")

    for eixo in (eixo_quantidade, eixo_percentual):
        eixo.set_xlabel("classe prevista")
        eixo.set_ylabel("classe real")

    figura.suptitle("Matriz de confusão — conjunto de teste")
    _salvar_figura(figura, caminho)


def salvar_grade_de_erros(
    caminhos_imagens: list[str],
    classes_reais: np.ndarray,
    classes_previstas: np.ndarray,
    nomes_classes: list[str],
    caminho: Path,
    maximo_de_imagens: int = 12,
) -> None:
    """Etapa 4: mostra as imagens de teste que o modelo classificou errado."""
    indices_erros = np.flatnonzero(classes_reais != classes_previstas)[:maximo_de_imagens]

    if len(indices_erros) == 0:
        figura, eixo = plt.subplots(figsize=(5, 2))
        eixo.text(0.5, 0.5, "Nenhum erro no conjunto de teste!", ha="center", va="center", fontsize=14)
        eixo.axis("off")
        _salvar_figura(figura, caminho)
        return

    colunas = min(4, len(indices_erros))
    linhas = int(np.ceil(len(indices_erros) / colunas))
    figura, eixos = plt.subplots(linhas, colunas, figsize=(3 * colunas, 3.5 * linhas), squeeze=False)

    for eixo in eixos.flat:
        eixo.axis("off")

    for eixo, indice in zip(eixos.flat, indices_erros):
        eixo.imshow(_abrir_quadrada(caminhos_imagens[indice]))
        eixo.set_title(
            f"real: {nomes_classes[classes_reais[indice]]}\n"
            f"previsto: {nomes_classes[classes_previstas[indice]]}",
            fontsize=10,
            color="darkred",
        )

    figura.suptitle("Imagens de teste classificadas errado")
    _salvar_figura(figura, caminho)

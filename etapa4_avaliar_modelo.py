"""
ETAPA 4 — Avaliação do modelo no conjunto de TESTE.

O conjunto de teste não foi usado nem no treino nem na escolha do modelo,
então ele mostra como o modelo se comporta com imagens novas.

Métricas calculadas:

  - Matriz de confusão: linhas = classe real, colunas = classe prevista.
    A diagonal são os acertos; fora dela, os erros (e com qual classe o
    modelo confundiu).

  Para cada classe C, contando:
    VP = imagens de C previstas como C            (verdadeiros positivos)
    FP = imagens de outra classe previstas como C (falsos positivos)
    FN = imagens de C previstas como outra classe (falsos negativos)
    VN = imagens de outra classe não previstas como C (verdadeiros negativos)

  - Precisão      = VP / (VP + FP)  -> das vezes que o modelo disse "C", quantas acertou
  - Revocação     = VP / (VP + FN)  -> das imagens de C, quantas o modelo encontrou
                                       (também chamada de sensibilidade ou recall)
  - Especificidade = VN / (VN + FP) -> das imagens que NÃO são C, quantas ele não chamou de C
  - F1-score      = média harmônica entre precisão e revocação
  - Acurácia      = total de acertos / total de imagens
  - Média macro   = média simples das classes (todas pesam igual)
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import config
from utils.dados import carregar_conjunto, criar_carregador
from utils.graficos import salvar_grade_de_erros, salvar_matriz_de_confusao
from utils.modelo import carregar_modelo, escolher_dispositivo


@torch.no_grad()
def prever_conjunto(modelo, carregador, dispositivo) -> tuple[np.ndarray, np.ndarray]:
    """Retorna as classes reais e as classes previstas pelo modelo para todas as imagens."""
    classes_reais, classes_previstas = [], []

    for imagens, rotulos in carregador:
        saidas = modelo(imagens.to(dispositivo))
        classes_previstas.extend(saidas.argmax(dim=1).cpu().tolist())
        classes_reais.extend(rotulos.tolist())

    return np.array(classes_reais), np.array(classes_previstas)


def calcular_especificidade(matriz: np.ndarray) -> np.ndarray:
    """Calcula a especificidade de cada classe a partir da matriz de confusão."""
    total = matriz.sum()
    verdadeiros_positivos = np.diag(matriz)
    falsos_positivos = matriz.sum(axis=0) - verdadeiros_positivos  # coluna, fora da diagonal
    falsos_negativos = matriz.sum(axis=1) - verdadeiros_positivos  # linha, fora da diagonal
    verdadeiros_negativos = total - verdadeiros_positivos - falsos_positivos - falsos_negativos

    return verdadeiros_negativos / (verdadeiros_negativos + falsos_positivos)


def montar_relatorio(classes_reais, classes_previstas, nomes_classes, matriz) -> str:
    """Monta um texto com a matriz de confusão e as métricas por classe."""
    precisao, revocacao, f1, quantidade = precision_recall_fscore_support(
        classes_reais, classes_previstas, labels=range(len(nomes_classes)), zero_division=0
    )
    especificidade = calcular_especificidade(matriz)
    acuracia = accuracy_score(classes_reais, classes_previstas)

    linhas = ["MATRIZ DE CONFUSÃO (linhas = real, colunas = previsto)", ""]
    linhas.append(f"{'':>10}" + "".join(f"{nome:>10}" for nome in nomes_classes))
    for nome, linha_matriz in zip(nomes_classes, matriz):
        linhas.append(f"{nome:>10}" + "".join(f"{valor:>10}" for valor in linha_matriz))

    linhas += ["", "MÉTRICAS POR CLASSE", ""]
    cabecalho = f"{'classe':<12}{'precisão':>10}{'revocação':>11}{'especif.':>10}{'F1':>8}{'imagens':>9}"
    linhas += [cabecalho, "-" * len(cabecalho)]
    for i, nome in enumerate(nomes_classes):
        linhas.append(
            f"{nome:<12}{precisao[i]:>10.3f}{revocacao[i]:>11.3f}"
            f"{especificidade[i]:>10.3f}{f1[i]:>8.3f}{quantidade[i]:>9}"
        )
    linhas.append("-" * len(cabecalho))
    linhas.append(
        f"{'média macro':<12}{precisao.mean():>10.3f}{revocacao.mean():>11.3f}"
        f"{especificidade.mean():>10.3f}{f1.mean():>8.3f}{quantidade.sum():>9}"
    )

    acertos = int(np.trace(matriz))
    linhas += ["", f"ACURÁCIA: {acuracia:.3f} ({acertos} de {len(classes_reais)} imagens corretas)"]

    return "\n".join(linhas)


def main() -> None:
    dispositivo = escolher_dispositivo()
    modelo, nomes_classes = carregar_modelo(config.CAMINHO_MODELO_TREINADO, dispositivo)

    conjunto_teste = carregar_conjunto("teste")
    carregador_teste = criar_carregador(conjunto_teste, embaralhar=False)
    print(f"Avaliando {len(conjunto_teste)} imagens de teste...\n")

    classes_reais, classes_previstas = prever_conjunto(modelo, carregador_teste, dispositivo)
    matriz = confusion_matrix(classes_reais, classes_previstas, labels=range(len(nomes_classes)))

    relatorio = montar_relatorio(classes_reais, classes_previstas, nomes_classes, matriz)
    print(relatorio)

    caminho_relatorio = config.PASTA_RESULTADOS / "4_metricas.txt"
    caminho_relatorio.write_text(relatorio + "\n")

    caminho_matriz = config.PASTA_RESULTADOS / "4_matriz_de_confusao.png"
    salvar_matriz_de_confusao(matriz, nomes_classes, caminho_matriz)

    # Com shuffle desligado, a ordem das previsões é a mesma de conjunto_teste.samples.
    caminhos_imagens = [caminho for caminho, _ in conjunto_teste.samples]
    caminho_erros = config.PASTA_RESULTADOS / "4_erros_de_classificacao.png"
    salvar_grade_de_erros(caminhos_imagens, classes_reais, classes_previstas, nomes_classes, caminho_erros)

    print(f"\nMétricas salvas em {caminho_relatorio}")
    print(f"Matriz de confusão salva em {caminho_matriz}")
    print(f"Exemplos de erros salvos em {caminho_erros}")


if __name__ == "__main__":
    main()

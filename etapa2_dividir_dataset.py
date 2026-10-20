"""
ETAPA 2 — Protocolo experimental: divisão em treino, validação e teste.

Protocolo adotado: holdout estratificado 70% / 15% / 15%.

  - Treino    (70%): imagens usadas para ajustar os pesos do modelo.
  - Validação (15%): usadas durante o treino para escolher a melhor época e
                     decidir quando parar. O modelo nunca aprende com elas.
  - Teste     (15%): usadas uma única vez, no final (Etapa 4), para medir o
                     desempenho em imagens que o modelo nunca viu.

"Estratificado" = cada conjunto mantém a mesma proporção de classes da base
original. A semente aleatória fixa faz a divisão ser sempre a mesma.

Resultado:

    dados/divididos/
        treino/     pizza/ bife/ sushi/
        validacao/  pizza/ bife/ sushi/
        teste/      pizza/ bife/ sushi/
"""

import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

import config
from utils.dados import listar_imagens_por_classe
from utils.graficos import salvar_grafico_da_divisao


def dividir_estratificado(caminhos: list[Path], classes: list[str]) -> dict[str, list]:
    """Divide as imagens em treino, validação e teste mantendo a proporção das classes."""

    # 1º corte: separa o conjunto de teste (15%) do restante (85%).
    caminhos_resto, caminhos_teste, classes_resto, classes_teste = train_test_split(
        caminhos,
        classes,
        test_size=config.PROPORCAO_TESTE,
        stratify=classes,
        random_state=config.SEMENTE,
    )

    # 2º corte: do restante, separa a validação.
    # A validação deve ser 15% do TOTAL, ou seja, 0.15 / 0.85 ≈ 17.6% do restante.
    proporcao_validacao_no_resto = config.PROPORCAO_VALIDACAO / (
        config.PROPORCAO_TREINO + config.PROPORCAO_VALIDACAO
    )
    caminhos_treino, caminhos_validacao, classes_treino, classes_validacao = train_test_split(
        caminhos_resto,
        classes_resto,
        test_size=proporcao_validacao_no_resto,
        stratify=classes_resto,
        random_state=config.SEMENTE,
    )

    return {
        "treino": list(zip(caminhos_treino, classes_treino)),
        "validacao": list(zip(caminhos_validacao, classes_validacao)),
        "teste": list(zip(caminhos_teste, classes_teste)),
    }


def copiar_para_pastas(divisao: dict[str, list]) -> None:
    """Copia cada imagem para dados/divididos/<conjunto>/<classe>/."""

    # Recria a pasta do zero para não misturar com uma divisão antiga.
    if config.PASTA_IMAGENS_DIVIDIDAS.exists():
        shutil.rmtree(config.PASTA_IMAGENS_DIVIDIDAS)

    for nome_conjunto, imagens in divisao.items():
        for caminho_imagem, classe in imagens:
            pasta_destino = config.PASTA_IMAGENS_DIVIDIDAS / nome_conjunto / classe
            pasta_destino.mkdir(parents=True, exist_ok=True)
            shutil.copy(caminho_imagem, pasta_destino / caminho_imagem.name)


def contar_por_classe(divisao: dict[str, list], nomes_classes: list[str]) -> dict[str, dict[str, int]]:
    """Retorna, para cada conjunto, quantas imagens existem de cada classe."""
    contagem = {}
    for nome_conjunto, imagens in divisao.items():
        classes_do_conjunto = [classe for _, classe in imagens]
        contagem[nome_conjunto] = {
            classe: classes_do_conjunto.count(classe) for classe in nomes_classes
        }
    return contagem


def imprimir_tabela(contagem: dict[str, dict[str, int]], nomes_classes: list[str]) -> None:
    """Mostra no terminal uma tabela conjunto x classe."""
    cabecalho = f"{'conjunto':<10}" + "".join(f"{classe:>8}" for classe in nomes_classes) + f"{'total':>8}"
    print(cabecalho)
    print("-" * len(cabecalho))
    for nome_conjunto, quantidades in contagem.items():
        linha = f"{nome_conjunto:<10}"
        linha += "".join(f"{quantidades[classe]:>8}" for classe in nomes_classes)
        linha += f"{sum(quantidades.values()):>8}"
        print(linha)


def main() -> None:
    caminhos, classes = listar_imagens_por_classe(config.PASTA_IMAGENS_ORIGINAIS)
    nomes_classes = sorted(set(classes))
    print(f"{len(caminhos)} imagens encontradas em {len(nomes_classes)} classes: {nomes_classes}\n")

    divisao = dividir_estratificado(caminhos, classes)
    copiar_para_pastas(divisao)

    contagem = contar_por_classe(divisao, nomes_classes)
    imprimir_tabela(contagem, nomes_classes)

    caminho_figura = config.PASTA_RESULTADOS / "2_divisao_do_dataset.png"
    salvar_grafico_da_divisao(contagem, nomes_classes, caminho_figura)
    print(f"\nGráfico da divisão salvo em {caminho_figura}")


if __name__ == "__main__":
    main()

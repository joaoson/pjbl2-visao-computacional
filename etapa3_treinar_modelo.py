"""
ETAPA 3 — Treinamento do modelo de visão computacional.

Modelo: ResNet-18 pré-treinada no ImageNet (transfer learning).
Apenas a última camada é treinada para as nossas 3 classes.

A cada época:
  1. o modelo é treinado com o conjunto de TREINO;
  2. o modelo é avaliado no conjunto de VALIDAÇÃO;
  3. se a perda na validação for a menor até agora, o modelo é salvo.

Se a validação não melhorar por várias épocas seguidas, o treino para
antes do fim (parada antecipada), evitando overfitting.
O conjunto de TESTE não é usado aqui: ele fica reservado para a Etapa 4.
"""

import json

import torch
from torch import nn

import config
from utils.dados import carregar_conjunto, criar_carregador
from utils.graficos import salvar_curvas_de_treinamento
from utils.modelo import criar_modelo, escolher_dispositivo, fixar_semente, salvar_modelo


def treinar_uma_epoca(modelo, carregador, funcao_perda, otimizador, dispositivo) -> tuple[float, float]:
    """Passa uma vez por todas as imagens de treino ajustando os pesos. Retorna (perda, acurácia)."""
    modelo.train()
    soma_perdas, acertos, total = 0.0, 0, 0

    for imagens, rotulos in carregador:
        imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)

        saidas = modelo(imagens)  # 1. previsão
        perda = funcao_perda(saidas, rotulos)  # 2. quão errada foi a previsão

        otimizador.zero_grad()
        perda.backward()  # 3. calcula como cada peso influenciou o erro
        otimizador.step()  # 4. ajusta os pesos para errar menos

        soma_perdas += perda.item() * len(rotulos)
        acertos += (saidas.argmax(dim=1) == rotulos).sum().item()
        total += len(rotulos)

    return soma_perdas / total, acertos / total


@torch.no_grad()  # na avaliação não é preciso calcular gradientes
def avaliar_uma_epoca(modelo, carregador, funcao_perda, dispositivo) -> tuple[float, float]:
    """Mede perda e acurácia sem alterar os pesos do modelo."""
    modelo.eval()
    soma_perdas, acertos, total = 0.0, 0, 0

    for imagens, rotulos in carregador:
        imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)

        saidas = modelo(imagens)
        perda = funcao_perda(saidas, rotulos)

        soma_perdas += perda.item() * len(rotulos)
        acertos += (saidas.argmax(dim=1) == rotulos).sum().item()
        total += len(rotulos)

    return soma_perdas / total, acertos / total


def main() -> None:
    fixar_semente(config.SEMENTE)
    dispositivo = escolher_dispositivo()
    print(f"Dispositivo de treinamento: {dispositivo}")

    conjunto_treino = carregar_conjunto("treino")
    conjunto_validacao = carregar_conjunto("validacao")
    carregador_treino = criar_carregador(conjunto_treino, embaralhar=True)
    carregador_validacao = criar_carregador(conjunto_validacao, embaralhar=False)

    nomes_classes = conjunto_treino.classes
    print(f"Classes: {nomes_classes}")
    print(f"Imagens de treino: {len(conjunto_treino)} | validação: {len(conjunto_validacao)}\n")

    modelo = criar_modelo(len(nomes_classes)).to(dispositivo)
    funcao_perda = nn.CrossEntropyLoss()
    # Só a última camada (modelo.fc) é passada ao otimizador, pois é a única treinável.
    otimizador = torch.optim.Adam(modelo.fc.parameters(), lr=config.TAXA_APRENDIZADO)

    historico = {"perda_treino": [], "acuracia_treino": [], "perda_validacao": [], "acuracia_validacao": []}
    melhor_perda_validacao = float("inf")
    epocas_sem_melhora = 0

    for epoca in range(1, config.NUMERO_EPOCAS + 1):
        perda_treino, acuracia_treino = treinar_uma_epoca(
            modelo, carregador_treino, funcao_perda, otimizador, dispositivo
        )
        perda_validacao, acuracia_validacao = avaliar_uma_epoca(
            modelo, carregador_validacao, funcao_perda, dispositivo
        )

        historico["perda_treino"].append(perda_treino)
        historico["acuracia_treino"].append(acuracia_treino)
        historico["perda_validacao"].append(perda_validacao)
        historico["acuracia_validacao"].append(acuracia_validacao)

        print(
            f"Época {epoca:2d}/{config.NUMERO_EPOCAS} | "
            f"treino: perda {perda_treino:.3f}, acurácia {acuracia_treino:.1%} | "
            f"validação: perda {perda_validacao:.3f}, acurácia {acuracia_validacao:.1%}",
            end="",
        )

        if perda_validacao < melhor_perda_validacao:
            melhor_perda_validacao = perda_validacao
            epocas_sem_melhora = 0
            salvar_modelo(modelo, nomes_classes, config.CAMINHO_MODELO_TREINADO)
            print("  <- melhor até agora, modelo salvo")
        else:
            epocas_sem_melhora += 1
            print()

        if epocas_sem_melhora >= config.PACIENCIA_PARADA_ANTECIPADA:
            print(f"\nParada antecipada: {epocas_sem_melhora} épocas sem melhora na validação.")
            break

    config.CAMINHO_HISTORICO_TREINO.write_text(json.dumps(historico, indent=2))

    caminho_figura = config.PASTA_RESULTADOS / "3_curvas_de_treinamento.png"
    salvar_curvas_de_treinamento(historico, caminho_figura)

    print(f"\nMelhor modelo salvo em {config.CAMINHO_MODELO_TREINADO}")
    print(f"Curvas de treinamento salvas em {caminho_figura}")


if __name__ == "__main__":
    main()

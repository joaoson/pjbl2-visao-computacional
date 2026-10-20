"""Criação, salvamento e carregamento do modelo de visão computacional."""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models


def fixar_semente(semente: int) -> None:
    """Fixa as sementes aleatórias para que os resultados sejam reproduzíveis."""
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


def escolher_dispositivo() -> torch.device:
    """Usa a GPU se houver uma disponível (NVIDIA ou Apple Silicon); senão, a CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def criar_modelo(numero_classes: int, pre_treinado: bool = True) -> nn.Module:
    """
    Cria uma ResNet-18 adaptada para o nosso problema (transfer learning).

    A ResNet-18 é uma rede neural convolucional (CNN) já treinada no ImageNet,
    com 1,2 milhão de imagens de 1000 categorias. Ela já sabe reconhecer
    bordas, texturas e formas, então aproveitamos esse conhecimento e
    treinamos apenas a última camada para distinguir as nossas classes.
    """
    pesos = models.ResNet18_Weights.DEFAULT if pre_treinado else None
    modelo = models.resnet18(weights=pesos)

    # Congela todas as camadas pré-treinadas: seus pesos não serão alterados.
    for parametro in modelo.parameters():
        parametro.requires_grad = False

    # Troca a última camada (que classificava 1000 classes do ImageNet)
    # por uma nova camada com uma saída para cada uma das nossas classes.
    # Por ser nova, apenas esta camada é treinada.
    modelo.fc = nn.Linear(modelo.fc.in_features, numero_classes)

    return modelo


def salvar_modelo(modelo: nn.Module, nomes_classes: list[str], caminho: Path) -> None:
    """Salva os pesos do modelo junto com os nomes das classes."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pesos": modelo.state_dict(), "classes": nomes_classes}, caminho)


def carregar_modelo(caminho: Path, dispositivo: torch.device) -> tuple[nn.Module, list[str]]:
    """Carrega um modelo salvo por salvar_modelo() e o deixa pronto para avaliação."""
    checkpoint = torch.load(caminho, map_location=dispositivo)
    nomes_classes = checkpoint["classes"]

    modelo = criar_modelo(len(nomes_classes), pre_treinado=False)
    modelo.load_state_dict(checkpoint["pesos"])
    modelo.to(dispositivo)
    modelo.eval()

    return modelo, nomes_classes

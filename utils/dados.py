"""Funções para listar, transformar e carregar as imagens."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config

EXTENSOES_DE_IMAGEM = {".jpg", ".jpeg", ".png"}


def listar_imagens_por_classe(pasta: Path) -> tuple[list[Path], list[str]]:
    """
    Percorre pasta/<classe>/ e devolve duas listas alinhadas:
    os caminhos das imagens e a classe (nome da subpasta) de cada uma.
    """
    caminhos, classes = [], []
    pastas_de_classe = sorted(p for p in pasta.iterdir() if p.is_dir())

    for pasta_classe in pastas_de_classe:
        for arquivo in sorted(pasta_classe.iterdir()):
            if arquivo.suffix.lower() in EXTENSOES_DE_IMAGEM:
                caminhos.append(arquivo)
                classes.append(pasta_classe.name)

    return caminhos, classes


def transformacoes_de_treino() -> transforms.Compose:
    """
    Transformações aplicadas nas imagens de treino.

    Inclui "data augmentation" (recortes e espelhamentos aleatórios): a cada
    época o modelo vê versões um pouco diferentes da mesma foto, o que reduz
    o risco de ele decorar as imagens (overfitting).
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(config.TAMANHO_IMAGEM, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.MEDIA_IMAGENET, config.DESVIO_IMAGENET),
    ])


def transformacoes_de_avaliacao() -> transforms.Compose:
    """
    Transformações aplicadas na validação e no teste.

    Sem nada aleatório: a mesma imagem sempre gera a mesma entrada,
    para que a avaliação seja justa e reproduzível.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.TAMANHO_IMAGEM),
        transforms.ToTensor(),
        transforms.Normalize(config.MEDIA_IMAGENET, config.DESVIO_IMAGENET),
    ])


def carregar_conjunto(nome_conjunto: str) -> datasets.ImageFolder:
    """
    Carrega "treino", "validacao" ou "teste" a partir de dados/divididos/.

    O ImageFolder usa o nome de cada subpasta como rótulo da classe.
    """
    if nome_conjunto == "treino":
        transformacoes = transformacoes_de_treino()
    else:
        transformacoes = transformacoes_de_avaliacao()

    pasta = config.PASTA_IMAGENS_DIVIDIDAS / nome_conjunto
    return datasets.ImageFolder(pasta, transform=transformacoes)


def criar_carregador(conjunto: datasets.ImageFolder, embaralhar: bool) -> DataLoader:
    """Cria um DataLoader, que entrega as imagens ao modelo em lotes (batches)."""
    return DataLoader(conjunto, batch_size=config.TAMANHO_LOTE, shuffle=embaralhar)

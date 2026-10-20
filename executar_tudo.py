"""Executa as 4 etapas do projeto em sequência."""

import etapa1_obter_dataset
import etapa2_dividir_dataset
import etapa3_treinar_modelo
import etapa4_avaliar_modelo

ETAPAS = [
    ("ETAPA 1 — Obter a base de imagens", etapa1_obter_dataset),
    ("ETAPA 2 — Dividir em treino, validação e teste", etapa2_dividir_dataset),
    ("ETAPA 3 — Treinar o modelo", etapa3_treinar_modelo),
    ("ETAPA 4 — Avaliar o modelo no teste", etapa4_avaliar_modelo),
]

if __name__ == "__main__":
    for titulo, etapa in ETAPAS:
        print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")
        etapa.main()

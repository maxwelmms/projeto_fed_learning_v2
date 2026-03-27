#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importa o classificador linear incremental do scikit-learn.
# Escolhemos SGDClassifier porque ele suporta partial_fit(),
# o que é importante no cenário federado com clientes possivelmente
# vendo subconjuntos diferentes das classes.
from sklearn.linear_model import SGDClassifier


def create_model(random_state: int = 42):
    """
    Cria e retorna o modelo base usado por cada cliente e pelo servidor.

    Por que usamos uma função:
    - centraliza a definição do modelo em um único lugar
    - facilita trocar o algoritmo depois sem mexer no restante do pipeline
    - garante que clientes e servidor usem a mesma arquitetura

    Parâmetro:
    - random_state: controla a reprodutibilidade do algoritmo

    Retorno:
    - uma instância de SGDClassifier configurada para classificação logística
    """

    return SGDClassifier(
        # loss="log_loss" faz o SGDClassifier se comportar como
        # uma regressão logística treinada via descida de gradiente estocástica.
        # Isso nos dá um classificador probabilístico linear adequado
        # para tarefas de classificação multiclasse.
        loss="log_loss",

        # penalty="l2" aplica regularização L2.
        # Por que usamos:
        # - ajuda a evitar sobreajuste
        # - estabiliza os pesos do modelo
        # - é uma escolha padrão e robusta para classificadores lineares
        penalty="l2",

        # alpha é a força da regularização.
        # Um valor pequeno mantém regularização moderada.
        # Aqui usamos 1e-4 como valor padrão estável.
        alpha=1e-4,

        # max_iter=1 é intencional.
        # Por que:
        # - no cenário federado, cada rodada local funciona como um pequeno passo de atualização
        # - com SGDClassifier, queremos atualizações incrementais curtas por round
        # - isso combina com o uso de partial_fit() no analyze.py
        max_iter=1,

        # tol=None desabilita critério interno de parada por tolerância.
        # Por que:
        # - como estamos usando max_iter=1 e partial_fit incremental,
        #   não queremos que o modelo pare "cedo" por convergência local
        # - o controle de rounds já é feito pelo pipeline federado
        tol=None,

        # random_state fixa a aleatoriedade do modelo.
        # Isso ajuda na reprodutibilidade dos experimentos.
        random_state=random_state,
    )
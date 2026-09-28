import numpy as np
import matplotlib.pyplot as plt

import selecao
import mutacao
import cruzamento
import funcoes_teste


def fitness (avaliacoes: np.ndarray) -> np.ndarray:
    return avaliacoes


def criar_populacao_inicial(N: int, n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low, high, size=(N, n))


def algoritmo_genetico ():
    ...
    



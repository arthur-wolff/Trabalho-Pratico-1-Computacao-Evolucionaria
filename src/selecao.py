import numpy as np


def roleta(populacao: np.ndarray, avaliacoes: np.ndarray, n_selecionados: int,
           rng: np.random.Generator) -> np.ndarray:
    
    pesos = 1.0 / (1.0 + avaliacoes)
    probabilidade = pesos / np.sum(pesos)
    
    idx = rng.choice(len(populacao), size=n_selecionados, p=probabilidade)
    
    return populacao[idx]
    
    
def torneio(populacao: np.ndarray, avaliacoes: np.ndarray, n_selecionados: int,
            rng: np.random.Generator, k: int = 3) -> np.ndarray:
    parentes = rng.integers(len(populacao), size=(n_selecionados, k))
    
    nota = avaliacoes[parentes]
    coluna_vencedores = np.argmin(nota, axis=1)
    vencedores = parentes[np.arange(n_selecionados), coluna_vencedores]
    
    return populacao[vencedores]
    
    
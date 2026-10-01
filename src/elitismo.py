import numpy as np


def elitismo(populacao_at: np.ndarray, aval_atual: np.ndarray, filhos: np.ndarray, aval_filhos: np.ndarray, n_elite: int) -> tuple[np.ndarray, np.ndarray]:
    
    populacao_nv = filhos.copy()
    aval_nv = aval_filhos.copy()
    
    if n_elite <= 0:
        return populacao_nv, aval_nv

    idx_elite = np.argsort(aval_atual)[:n_elite]
    idx_piores = np.argsort(aval_filhos)[-n_elite:]
    
    populacao_nv[idx_piores] = populacao_at[idx_elite]
    aval_nv[idx_piores] = aval_atual[idx_elite]
    
    return populacao_nv, aval_nv
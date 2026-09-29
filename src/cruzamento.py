import numpy as np


def blx_alpha(parentes: np.ndarray, pc: float, alpha: float, low: float, high: float,
              rng: np.random.Generator) -> np.ndarray:
    
    if len(parentes) % 2 != 0:
       raise ValueError("O numero de pais dever ser par")
    
    p1 = parentes[0::2]
    p2 = parentes[1::2]
    
    d = np.abs(p1 - p2)
    min = np.minimum(p1, p2) - alpha * d
    max = np.maximum(p1, p2) + alpha * d
    
    filho1 = rng.uniform(min, max)
    filho2 = rng.uniform(min, max)
    
    cruza = rng.random(len(p1)) < pc
    filho1 = np.where(cruza[:, None], filho1, p1)
    filho2 = np.where(cruza[:, None], filho2, p2)
    
    filhos = np.empty_like(parentes)
    filhos[0::2] = filho1
    filhos[1::2] = filho2
    
    return np.clip(filhos, low, high)

def single_point(parentes: np.ndarray, pc: float, rng: np.random.Generator) -> np.ndarray:
    
    if len(parentes) % 2 != 0:
        raise ValueError("O nuemro de pais deve ser par")
    
    p1 = parentes[0::2]
    p2 = parentes[1::2]
    m, n = p1.shape
    
    ponto = rng.integers(1, n, size=m)
    antes_corte = np.arange(n)[None, :] < ponto[:, None]
    
    filho1 = np.where(antes_corte, p1, p2)
    filho2 = np.where(antes_corte, p2, p1)
    
    cruza = rng.random(m)< pc
    filho1 = np.where(cruza[:, None], filho1, p1)
    filho2 = np.where(cruza[:, None], filho2, p2)
    
    filhos = np.empty_like(parentes)
    filhos[0::2] = filho1
    filhos[1::2] = filho2
    
    return filhos


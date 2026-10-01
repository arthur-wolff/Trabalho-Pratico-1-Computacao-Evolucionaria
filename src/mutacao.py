import numpy as np


def gaussiana(populacao: np.ndarray, pm: float, sigma: float, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    
    mask = rng.random(populacao.shape) < pm
    
    ruido = rng.normal(0.0, sigma, size=populacao.shape)
    
    return np.clip(populacao + mask * ruido, low, high)

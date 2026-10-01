import numpy as np

# 1: Funcao Esferica


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)

# 2 Funcao rastrigin


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return 10*n+np.sum(x**2-10 * np.cos(2*np.pi * x))

# 3 Funcao rosenbrock


def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1]-1)**2)

import numpy as np
import matplotlib.pyplot as plt
import time

from src import selecao
from src import mutacao
from src import cruzamento
from src import funcoes_teste
from src import elitismo


def fitness(avaliacoes: np.ndarray) -> np.ndarray:
    return avaliacoes


def avaliar(populacao: np.ndarray, funcao_teste) -> np.ndarray:
    
    return np.array([funcao_teste(x) for x in populacao])


def criar_populacao_inicial(N: int, n: int, low: float, high: float,
                            rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low, high, size=(N, n))


def algoritmo_genetico(funcao_teste, n: int, *, N: int, pc: float, pm: float,
                       sigma: float, alpha: float, k: int, n_elite: int, 
                       metodo_selecao: str, metodo_cruzamento: str, low: float,
                       high: float, nfe_max: int, seed: int) -> dict:

    if N % 2 != 0:
        raise ValueError("N deve ser par")
    
    inicio = time.perf_counter()
    rng = np.random.default_rng(seed)
    
    populacao = criar_populacao_inicial(N, n,  low, high, rng)
    avaliacao = avaliar(populacao, funcao_teste)
    nfe = N
    
    historico = [avaliacao.min()]
    
    while nfe + N <= nfe_max:
        
        if metodo_selecao == "torneio":
            pais = selecao.torneio(populacao, fitness(avaliacao), N, rng, k)
            
        elif metodo_selecao == "roleta":
            pais = selecao.roleta(populacao, fitness(avaliacao), N, rng)
        
        else:
            raise ValueError(f"Metodo de selecao desconhecido : {metodo_selecao}")
        
        if metodo_cruzamento == "blx":
            filhos = cruzamento.blx_alpha(pais, pc, alpha, low, high, rng)
        
        elif metodo_cruzamento == "single_point":
            filhos = cruzamento.single_point(pais, pc, rng)
        
        else:
            raise ValueError(f"Metodo de cruzamento desconhecido: {metodo_cruzamento}")

        filhos = mutacao.gaussiana(filhos, pm, sigma, low, high, rng)
        
        avaliacao_filhos = avaliar(filhos, funcao_teste)
        nfe += N
        
        populacao, avaliacao = elitismo.elitismo(populacao, avaliacao, filhos, avaliacao_filhos, n_elite)
        
        historico.append(avaliacao.min())
        
    i_melhor = int(np.argmin(avaliacao))
        
    return {
        "melhor_x": populacao[i_melhor].copy(),
        "melhor_f": float(avaliacao[i_melhor]),
        "historico": np.array(historico),
        "nfe": nfe,
        "tempo": time.perf_counter() - inicio,
    }
    



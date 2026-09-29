import numpy as np
import matplotlib.pyplot as plt
import os

from src import algoritmo_genetico as ag
from src import funcoes_teste as ft

CONFIGURACAO_TESTES = dict(
    N=100, pc=0.9, pm=0.1, sigma=0.5,alpha=0.5, k=3,n_elite=2,
    metodo_selecao = "torneio", metodo_cruzamento="blx",
    low=-5.0, high=5.0, nfe_max = 20000,

)

FUNCOES = {
    "esferica": ft.sphere,
    "rastrigin": ft.rastrigin,
    "rosenbrock": ft.rosenbrock,
}

DIMENSOES = (2, 5, 10)
N_EXECUCOES = 10


def rodar_config(funcao, n: int, n_execucoes: int, **config) -> dict:
    
    melhor_f, nfes, tempos, historicos = [], [], [], []
    
    for seed in range(n_execucoes):
        r = ag.algoritmo_genetico(funcao, n, seed=seed, **config)
        melhor_f.append(r["melhor_f"])
        nfes.append(r["nfe"])
        tempos.append(r["tempo"])
        historicos.append(r["historico"])
    
    melhor_f = np.array(melhor_f)
    historicos = np.array(historicos)
    
    return {
        "melhor": melhor_f.min(),
        "media": melhor_f.mean(),
        "desvio": melhor_f.std(),
        "nfe": int(np.mean(nfes)),
        "tempo_medio": float(np.mean(tempos)),
        "historico_medio": historicos.mean(axis=0),
        "eixo_nfe": config["N"] * np.arange(1, historicos.shape[1] + 1),       
    }
    

def imprimir_tabela(titulo: str, linhas: list[tuple[str,dict]])-> None:
    
    print(f"\n{titulo}")
    print(f"{'':14s}{'melhor':>14s}{'media':>14s}{'desvio':>14s}{'nfe':>10s}{'tempo(s)':>10s}")
    for nome, r in linhas:
        print(f"{nome:14s}{r['melhor']:14.6g}{r['media']:14.6g}{r['desvio']:14.6g}"
        f"{r['nfe']:10d}{r['tempo_medio']:10.4f}")

def plotar_convergencia(titulo: str, linhas: List[tuple[str,dict]], arquivo: str) -> None:
    
    destino = "graficos"
    os.makedirs(destino, exist_ok=True)
    caminho = os.path.join(destino, arquivo)
    
    plt.figure(figsize=(6,4))
    
    for nome, r in linhas:
        plt.plot(r["eixo_nfe"],r["historico_medio"], label = nome)
    
    plt.yscale("log")
    plt.xlabel("NFE")
    plt.ylabel("Melhor f (media de 10 execucoes)")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()

def experimento_basico()->None:
    for nome_funcao, funcao in FUNCOES.items():
        linhas = []
        for n in DIMENSOES:
            r = rodar_config(funcao, n, N_EXECUCOES, **CONFIGURACAO_TESTES)
            linhas.append((f"n={n}",r))
        imprimir_tabela(f"Config Basica - {nome_funcao}",linhas)
        plotar_convergencia(f"Convergencia - {nome_funcao}", linhas,f"convergencia_{nome_funcao}.png")
        

if __name__ == "__main__":
    experimento_basico()
    
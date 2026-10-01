import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from src import algoritmo_genetico as ag
from src import funcoes_teste as ft

# Define a parametrizacao de execucao do AG
PARAMETRIZACAO = dict(
    N=100, pc=0.9, pm=0.1, sigma=0.5,alpha=0.5, k=3,n_elite=2,
    metodo_selecao = "torneio", metodo_cruzamento="blx",
    low=-5.0, high=5.0, nfe_max = 20000,

)
DIMENSOES = (2, 5, 10)
N_EXECUCOES = 10
EPS = 1e-16

# Define as funcoes de teste 
FUNCOES = {
    "esferica": ft.sphere,
    "rastrigin": ft.rastrigin,
    "rosenbrock": ft.rosenbrock,
}


# Define algumas configuracoes para gerar os graficos e tabelas 
PASTA_TABELAS = "tabelas"
PASTA_GRAFICOS = "graficos"


# Pega a parametrizaçao e roda o AG retornando os dados gerados 
def rodar_config(funcao, n: int, n_execucoes: int, **config) -> dict:
    melhor_f, nfes, tempos, historicos, divs = [], [], [], [], []

    for seed in range(n_execucoes):
        r = ag.algoritmo_genetico(funcao, n, seed=seed, **config)
        melhor_f.append(r["melhor_f"])
        nfes.append(r["nfe"])
        tempos.append(r["tempo"])
        historicos.append(r["historico"])
        divs.append(r["diversidade"])

    melhor_f = np.array(melhor_f)
    historicos = np.array(historicos)
    divs = np.array(divs)

    return {
        "melhor": melhor_f.min(),
        "media": melhor_f.mean(),
        "desvio": melhor_f.std(),
        "nfe": int(np.mean(nfes)),
        "tempo_medio": float(np.mean(tempos)),
        "historico_medio": historicos.mean(axis=0),
        "diversidade_media": divs.mean(axis=0),
        "eixo_nfe": config["N"] * np.arange(1, historicos.shape[1] + 1),
    }
    
    
# Funcoes destinadas a plotagem dos graficos e tabelas

# Definicao do cabecalho das tabelas 
CABECALHO = ["Parametros", "Melhor", "Media", "Desvio", "NFE", "Tempo medio (s)"]


# Formatacao das linhas da tabela
def _linhas_formatadas(linhas):
    return [[nome, f"{r['melhor']:.6g}", f"{r['media']:.6g}", f"{r['desvio']:.6g}",
             f"{r['nfe']}", f"{r['tempo_medio']:.4f}"] for nome, r in linhas]


# Imprime a as tabelas na CL
def imprimir_tabela(titulo, linhas):
    print(f"\n{titulo}")
    print(f"{'':14s}{'melhor':>14s}{'media':>14s}{'desvio':>14s}{'nfe':>10s}{'tempo(s)':>10s}")
    for nome, r in linhas:
        print(f"{nome:14s}{r['melhor']:14.6g}{r['media']:14.6g}{r['desvio']:14.6g}"
              f"{r['nfe']:10d}{r['tempo_medio']:10.4f}")


# Salva a tabela em um arquivo .CSV e as Plota em .png
def salvar_tabela(titulo, linhas, arquivo):
    os.makedirs(PASTA_TABELAS, exist_ok=True)
    dados = _linhas_formatadas(linhas)
 
    with open(os.path.join(PASTA_TABELAS, arquivo + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CABECALHO)
        w.writerows(dados)
 
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.4 * (len(dados) + 1)))
    ax.axis("off")
    tab = ax.table(cellText=dados, colLabels=CABECALHO, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.4)
    for j in range(len(CABECALHO)):
        tab[0, j].set_facecolor("#dfe6f0")
        tab[0, j].set_text_props(weight="bold")
    ax.set_title(titulo, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_TABELAS, arquivo + ".png"), dpi=200)
    plt.close(fig)


# Plota os graficos
def plotar(titulo, linhas, arquivo, chave="historico_medio", ylabel=None, log=True):
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for nome, r in linhas:
        y = np.maximum(r[chave], EPS) if log else r[chave]
        plt.plot(r["eixo_nfe"], y, label=nome)
    if log:
        plt.yscale("log")
    plt.xlabel("NFE")
    plt.ylabel(ylabel or "Melhor f (media de 10 execucoes)")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, arquivo), dpi=150)
    plt.close()


# Experimento padrao com a parametrizaçao inicial de testes 
def experimento_basico():
    for nome_funcao, funcao in FUNCOES.items():
        linhas = []
        for n in DIMENSOES:
            linhas.append((f"n={n}", rodar_config(funcao, n, N_EXECUCOES, **PARAMETRIZACAO)))
        titulo = f"Config Basica - {nome_funcao}"
        imprimir_tabela(titulo, linhas)
        salvar_tabela(titulo, linhas, f"tabela_{nome_funcao}")
        plotar(f"Convergencia - {nome_funcao}", linhas, f"convergencia_{nome_funcao}.png")
        plotar(f"Diversidade - {nome_funcao}", linhas, f"diversidade_{nome_funcao}.png",
               chave="diversidade_media", ylabel="Diversidade (desvio medio dos genes)")


# Experimento comparando os metodos de selecao de roleta e torneio
def experimento_selecao(nome_funcao="rastrigin", n=5):
    funcao = FUNCOES[nome_funcao]
    linhas = []
    
    for metodo in ("torneio", "roleta"):
        par = dict(PARAMETRIZACAO, metodo_selecao=metodo)
        linhas.append((metodo, rodar_config(funcao, n, N_EXECUCOES, **par)))
        titulo = f"Selecao: torneio x roleta - {nome_funcao} (n={n})"
        imprimir_tabela(titulo, linhas)
        salvar_tabela(titulo, linhas, f"tabela_selecao_{nome_funcao}_n{n}")
        plotar(f"Convergencia - {titulo}", linhas, f"selecao_covergencia_{nome_funcao}_n{n}.png")
        plotar(f"Diversidade - {titulo}", linhas, f"Selecao_diversidade_{nome_funcao}_n{n}.png", chave="diversidade_media", ylabel="Diversidade (desvio medio dos genes)")


if __name__ == "__main__":
    experimento_basico()
    experimento_selecao()
    
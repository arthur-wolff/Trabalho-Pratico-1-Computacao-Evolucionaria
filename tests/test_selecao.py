import numpy as np
from src import selecao
from src import funcoes_teste


def contar(selecionados: np.ndarray, individuo: np.ndarray) -> int:
    return int(np.sum(np.all(selecionados == individuo, axis=1)))


if __name__ == "__main__":
    # População fixa de exemplo (N=20, n=2) avaliada com a esférica
    rng = np.random.default_rng(0)
    pop = rng.uniform(-5, 5, size=(20, 2))
    aval = np.array([funcoes_teste.sphere(x) for x in pop])
    melhor = pop[np.argmin(aval)]
    pior = pop[np.argmax(aval)]

    # --- Teste 1: shape (n_selecionados diferente de k e de N) ---
    sel_t = selecao.torneio(pop, aval, 7, np.random.default_rng(1), k=3)
    sel_r = selecao.roleta(pop, aval, 7, np.random.default_rng(1))
    assert sel_t.shape == (7, 2)
    assert sel_r.shape == (7, 2)

    # --- Teste 2: todo selecionado é um indivíduo da população original ---
    for sel in (sel_t, sel_r):
        for linha in sel:
            assert np.any(np.all(pop == linha, axis=1))

    # --- Teste 3: reprodutibilidade com a mesma seed ---
    a = selecao.torneio(pop, aval, 10, np.random.default_rng(42))
    b = selecao.torneio(pop, aval, 10, np.random.default_rng(42))
    assert np.array_equal(a, b)
    c = selecao.roleta(pop, aval, 10, np.random.default_rng(42))
    d = selecao.roleta(pop, aval, 10, np.random.default_rng(42))
    assert np.array_equal(c, d)

    # --- Teste 4: pressão seletiva do torneio (k maior => melhor vence mais) ---
    n_sel = 5000
    t_k2 = selecao.torneio(pop, aval, n_sel, np.random.default_rng(3), k=2)
    t_k10 = selecao.torneio(pop, aval, n_sel, np.random.default_rng(3), k=10)
    assert contar(t_k10, melhor) > contar(t_k2, melhor)
    assert contar(t_k2, melhor) > contar(t_k2, pior)

    # --- Teste 5: roleta favorece o melhor sobre o pior ---
    r = selecao.roleta(pop, aval, n_sel, np.random.default_rng(4))
    assert contar(r, melhor) > contar(r, pior)

    # --- Teste 6: torneio com k = N (sorteio com reposição) ---
    # Probabilidade do melhor estar no torneio: 1 - (1 - 1/N)^N, ~64% para N=20
    t_kn = selecao.torneio(pop, aval, n_sel, np.random.default_rng(5), k=len(pop))
    esperado = 1 - (1 - 1 / len(pop)) ** len(pop)
    assert abs(contar(t_kn, melhor) / n_sel - esperado) < 0.05

    print("Todos os testes de selecao passaram")
import numpy as np
from src import algoritmo_genetico as ag
from src import funcoes_teste

if __name__ == "__main__":
    # Teste 1: respeita o orçamento de NFE e devolve os campos esperados
    r = ag.algoritmo_genetico(funcoes_teste.sphere, 5, N=100, pc=0.9, pm=0.2, sigma=0.5, alpha=0.5, k=3, n_elite=2, metodo_selecao="torneio", metodo_cruzamento="blx", low=-5.0, high=5.0, nfe_max=20000, seed=1)
    assert r["nfe"] <= 20000
    assert r["nfe"] == 20000  # N=100 divide 20000, então usa o orçamento inteiro
    assert r["melhor_x"].shape == (5,)
    assert r["tempo"] > 0

    # Teste 2: NFE respeitado também quando N não divide o orçamento
    r = ag.algoritmo_genetico(funcoes_teste.sphere, 3, N=60, pc=0.9, pm=0.2, sigma=0.5, alpha=0.5, k=3, n_elite=2, metodo_selecao="torneio", metodo_cruzamento="blx", low=-5.0, high=5.0, nfe_max=1000, seed=1)
    assert r["nfe"] <= 1000

    # Teste 3: com elitismo, o melhor valor nunca piora entre gerações
    r = ag.algoritmo_genetico(funcoes_teste.rastrigin, 5, N=100, pc=0.9, pm=0.2, sigma=0.5, alpha=0.5, k=3, n_elite=2, metodo_selecao="torneio", metodo_cruzamento="blx", low=-5.0, high=5.0, nfe_max=20000, seed=2)
    assert np.all(np.diff(r["historico"]) <= 1e-12)

    # Teste 4: melhor_f é consistente com melhor_x e está dentro do domínio
    r = ag.algoritmo_genetico(funcoes_teste.rosenbrock, 5, N=100, pc=0.9, pm=0.2, sigma=0.5, alpha=0.5, k=3, n_elite=2, metodo_selecao="torneio", metodo_cruzamento="blx", low=-5.0, high=5.0, nfe_max=20000, seed=3)
    assert np.isclose(r["melhor_f"], funcoes_teste.rosenbrock(r["melhor_x"]))
    assert np.all(r["melhor_x"] >= -5) and np.all(r["melhor_x"] <= 5)

    # Teste 5: reprodutibilidade com a mesma seed, e seeds diferentes divergem
    a = ag.algoritmo_genetico(funcoes_teste.sphere, 5, N=100, pc=0.9, pm=0.2, sigma=0.5, alpha=0.5, k=3, n_elite=2, metodo_selecao="torneio", metodo_cruzamento="blx", low=-5.0, high=5.0, nfe_max=20000, seed=42)
    b = ag.algoritmo_genetico(funcoes_teste.sphere, 5, N=100, pc=0.9, pm=0.2, sigma=0.5, alpha=0.5, k=3, n_elite=2, metodo_selecao="torneio", metodo_cruzamento="blx", low=-5.0, high=5.0, nfe_max=20000, seed=42)
    c = ag.algoritmo_genetico(funcoes_teste.sphere, 5, N=100, pc=0.9, pm=0.2, sigma=0.5, alpha=0.5, k=3, n_elite=2, metodo_selecao="torneio", metodo_cruzamento="blx", low=-5.0, high=5.0, nfe_max=20000, seed=43)
    assert a["melhor_f"] == b["melhor_f"]
    assert np.array_equal(a["historico"], b["historico"])
    assert a["melhor_f"] != c["melhor_f"]

    # Teste 6: converge na esférica e melhora bastante em relação à geração inicial
    r = ag.algoritmo_genetico(funcoes_teste.sphere, 2, N=100, pc=0.9, pm=0.2, sigma=0.5, alpha=0.5, k=3, n_elite=2, metodo_selecao="torneio", metodo_cruzamento="blx", low=-5.0, high=5.0, nfe_max=20000, seed=4)
    assert r["melhor_f"] < 1e-3
    assert r["historico"][-1] < r["historico"][0]

    # Teste 7: todas as combinações de seleção e cruzamento rodam
    for sel in ("torneio", "roleta"):
        for cruz in ("blx", "single_point"):
            r = ag.algoritmo_genetico(funcoes_teste.sphere, 5, N=100, pc=0.9, pm=0.2, sigma=0.5,
                                      alpha=0.5, k=3, n_elite=2, metodo_selecao=sel,
                                      metodo_cruzamento=cruz, low=-5.0, high=5.0,
                                      nfe_max=2000, seed=5)
            assert r["nfe"] <= 2000

    # Teste 8: entradas inválidas levantam ValueError
    base = dict(N=100, pc=0.9, pm=0.2, sigma=0.5, alpha=0.5, k=3, n_elite=2,
                metodo_selecao="torneio", metodo_cruzamento="blx", low=-5.0, high=5.0,
                nfe_max=2000, seed=0)
    for campo, valor in (("N", 51), ("metodo_selecao", "x"), ("metodo_cruzamento", "x")):
        kwargs = dict(base)
        kwargs[campo] = valor
        try:
            ag.algoritmo_genetico(funcoes_teste.sphere, 2, **kwargs)
            assert False, f"deveria falhar com {campo}={valor}"
        except ValueError:
            pass

    # Teste 9: avaliar devolve um valor por indivíduo, igual à função objetivo
    pop = np.random.default_rng(6).uniform(-5, 5, size=(8, 4))
    aval = ag.avaliar(pop, funcoes_teste.rastrigin)
    assert aval.shape == (8,)
    for x, v in zip(pop, aval):
        assert np.isclose(v, funcoes_teste.rastrigin(x))

    print("Todos os testes do algoritmo genetico passaram")
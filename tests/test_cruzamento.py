import numpy as np
from src import cruzamento

if __name__ == "__main__":
    low, high = -5, 5
    rng = np.random.default_rng(0)
    pais = rng.uniform(low, high, size=(10, 5))

    # ---------- BLX-alpha ----------
    # Teste 1: shape e domínio
    filhos = cruzamento.blx_alpha(pais, 0.9, 0.5, low, high, np.random.default_rng(1))
    assert filhos.shape == pais.shape
    assert np.all(filhos >= low) and np.all(filhos <= high)

    # Teste 2: pc=0 devolve os pais; pc=1 altera
    f0 = cruzamento.blx_alpha(pais, 0.0, 0.5, low, high, np.random.default_rng(2))
    assert np.array_equal(f0, pais)
    f1 = cruzamento.blx_alpha(pais, 1.0, 0.5, low, high, np.random.default_rng(2))
    assert not np.array_equal(f1, pais)

    # Teste 3: filhos ficam no intervalo [min - a*d, max + a*d] (limites largos, sem clipping)
    alpha = 0.5
    filhos_l = cruzamento.blx_alpha(pais, 1.0, alpha, -100, 100, np.random.default_rng(3))
    p1, p2 = pais[0::2], pais[1::2]
    d = np.abs(p1 - p2)
    minimo = np.minimum(p1, p2) - alpha * d
    maximo = np.maximum(p1, p2) + alpha * d
    for f in (filhos_l[0::2], filhos_l[1::2]):
        assert np.all(f >= minimo - 1e-12) and np.all(f <= maximo + 1e-12)

    # Teste 4: reprodutibilidade e número ímpar de pais
    a = cruzamento.blx_alpha(pais, 0.9, 0.5, low, high, np.random.default_rng(42))
    b = cruzamento.blx_alpha(pais, 0.9, 0.5, low, high, np.random.default_rng(42))
    assert np.array_equal(a, b)
    try:
        cruzamento.blx_alpha(pais[:9], 0.9, 0.5, low, high, np.random.default_rng(0))
        assert False, "deveria levantar ValueError"
    except ValueError:
        pass

    # ---------- Um ponto ----------
    # Teste 5: shape e domínio
    filhos = cruzamento.single_point(pais, 1.0, np.random.default_rng(4))
    assert filhos.shape == pais.shape
    assert np.all(filhos >= low) and np.all(filhos <= high)

    # Teste 6: todo gene do filho vem do mesmo índice de um dos pais do casal
    for i in range(len(pais) // 2):
        for j in range(pais.shape[1]):
            opcoes = (pais[2 * i, j], pais[2 * i + 1, j])
            assert filhos[2 * i, j] in opcoes
            assert filhos[2 * i + 1, j] in opcoes

    # Teste 7: pc=0 devolve os pais; funciona com n=2; reprodutibilidade
    assert np.array_equal(cruzamento.single_point(pais, 0.0, np.random.default_rng(5)), pais)
    pais2 = rng.uniform(low, high, size=(6, 2))
    assert cruzamento.single_point(pais2, 1.0, np.random.default_rng(6)).shape == (6, 2)
    a = cruzamento.single_point(pais, 0.9, np.random.default_rng(42))
    b = cruzamento.single_point(pais, 0.9, np.random.default_rng(42))
    assert np.array_equal(a, b)

    print("Todos os testes de cruzamento passaram")
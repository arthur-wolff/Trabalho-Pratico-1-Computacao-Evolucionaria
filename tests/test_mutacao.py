import numpy as np
from src import mutacao

if __name__ == "__main__":
    low, high = -5, 5
    rng = np.random.default_rng(0)
    pop = rng.uniform(low, high, size=(1000, 5))
    copia = pop.copy()

    # Teste 1: shape e domínio (com sigma grande para forçar o clipping)
    m = mutacao.gaussiana(pop, 0.5, 5.0, low, high, np.random.default_rng(1))
    assert m.shape == pop.shape
    assert np.all(m >= low) and np.all(m <= high)

    # Teste 2: a população original não é alterada
    assert np.array_equal(pop, copia)

    # Teste 3: pm=0 não muda nada; sigma=0 também não
    assert np.array_equal(mutacao.gaussiana(pop, 0.0, 1.0, low, high, np.random.default_rng(2)), pop)
    assert np.array_equal(mutacao.gaussiana(pop, 1.0, 0.0, low, high, np.random.default_rng(2)), pop)

    # Teste 4: fração de genes alterados ~ pm
    for pm in (0.1, 0.5):
        m = mutacao.gaussiana(pop, pm, 1.0, low, high, np.random.default_rng(3))
        frac = np.mean(m != pop)
        assert abs(frac - pm) < 0.03

    # Teste 5: pm=1 altera praticamente todos os genes
    m = mutacao.gaussiana(pop, 1.0, 1.0, low, high, np.random.default_rng(4))
    assert np.mean(m != pop) > 0.95

    # Teste 6: reprodutibilidade
    a = mutacao.gaussiana(pop, 0.2, 1.0, low, high, np.random.default_rng(42))
    b = mutacao.gaussiana(pop, 0.2, 1.0, low, high, np.random.default_rng(42))
    assert np.array_equal(a, b)

    print("Todos os testes de mutacao passaram")
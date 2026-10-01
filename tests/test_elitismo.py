import numpy as np
from src import elitismo
from src import funcoes_teste

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pop = rng.uniform(-5, 5, size=(20, 3))
    filhos = rng.uniform(-5, 5, size=(20, 3))
    aval_pop = np.array([funcoes_teste.sphere(x) for x in pop])
    aval_filhos = np.array([funcoes_teste.sphere(x) for x in filhos])

    # Teste 1: shapes preservados
    nova_pop, nova_aval = elitismo.elitismo(pop, aval_pop, filhos, aval_filhos, 2)
    assert nova_pop.shape == filhos.shape
    assert nova_aval.shape == aval_filhos.shape

    # Teste 2: os melhores da geração atual sobrevivem
    for i in np.argsort(aval_pop)[:2]:
        assert np.any(np.all(nova_pop == pop[i], axis=1))

    # Teste 3: avaliações continuam correspondendo aos indivíduos (sem reavaliar)
    recalc = np.array([funcoes_teste.sphere(x) for x in nova_pop])
    assert np.allclose(nova_aval, recalc)

    # Teste 4: o melhor da nova geração nunca é pior que o da anterior
    assert nova_aval.min() <= aval_pop.min()

    # Teste 5: n_elite=0 devolve os filhos sem mudança; entradas não são alteradas
    p0, a0 = elitismo.elitismo(pop, aval_pop, filhos, aval_filhos, 0)
    assert np.array_equal(p0, filhos) and np.array_equal(a0, aval_filhos)
    assert not np.shares_memory(p0, filhos)

    # Teste 6: só os n_elite piores filhos são substituídos
    diferentes = np.sum(np.any(nova_pop != filhos, axis=1))
    assert diferentes <= 2

    print("Todos os testes de elitismo passaram")
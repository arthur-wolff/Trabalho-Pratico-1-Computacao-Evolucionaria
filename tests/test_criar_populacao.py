from src.algoritmo_genetico import criar_populacao_inicial
import numpy as np

if __name__ == "__main__":
    low, high = -5, 5
    
    rng = np.random.default_rng(1)
    pop_a = criar_populacao_inicial(5, 2, low, high, rng)
    assert pop_a.shape == (5,2)
    
    pop_b = criar_populacao_inicial(10, 5, low, high, rng)
    assert pop_b.shape == (10, 5)
 
    # --- Teste 2: todos os valores dentro do domínio [low, high] ---
    rng = np.random.default_rng(2)
    pop_c = criar_populacao_inicial(50, 10, low, high, rng)
    assert np.all(pop_c >= low) and np.all(pop_c <= high)
 
    # --- Teste 3: reprodutibilidade com a mesma seed ---
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    pop_d = criar_populacao_inicial(5, 2, low, high, rng1)
    pop_e = criar_populacao_inicial(5, 2, low, high, rng2)
    assert np.array_equal(pop_d, pop_e)
 
    # --- Teste 4: seeds diferentes geram populações diferentes ---
    rng3 = np.random.default_rng(1)
    rng4 = np.random.default_rng(2)
    pop_f = criar_populacao_inicial(5, 2, low, high, rng3)
    pop_g = criar_populacao_inicial(5, 2, low, high, rng4)
    assert not np.array_equal(pop_f, pop_g)
 
    print("Todos os testes de criar_populacao_inicial passaram")


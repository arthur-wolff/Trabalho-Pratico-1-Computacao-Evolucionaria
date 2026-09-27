from src import funcoes_teste
import numpy as np

if __name__ == "__main__":
    otimo_gl_esfera = np.zeros(5)
    valor_esfera = funcoes_teste.sphere(otimo_gl_esfera)

    otimo_gl_rastrigin = np.zeros(5)
    valor_rastrigin = funcoes_teste.rastrigin(otimo_gl_rastrigin)

    otimo_gl_rosenbrock = np.ones(5)
    valor_rosenbrock = funcoes_teste.rosenbrock(otimo_gl_rosenbrock)
    
    assert np.isclose(valor_esfera, 0)
    assert np.isclose(valor_rastrigin, 0)
    assert np.isclose(valor_rosenbrock, 0)
    
    assert np.isclose(funcoes_teste.sphere(np.array([1,1,1,1,1])),5)
    assert np.isclose(funcoes_teste.rastrigin(np.array([1,1,1,1,1])), 5)
    assert np.isclose(funcoes_teste.rosenbrock(np.array([0,0,0,0,0])), 4)
    
    print("Todas as funcoes passaram no teste")
import numpy as np

def mutacion_gaussiana(hijos, prob_mutacion, sigma=0.1):
    for hijo in hijos:
        for i in range(len(hijo)):
            if np.random.rand() < prob_mutacion:
                hijo[i] += np.random.normal(0, sigma)
    return hijos
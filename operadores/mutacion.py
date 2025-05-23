import numpy as np

def mutacion_gaussiana(hijos, prob_mutacion, sigma=0.1):
    mut= 0
    for hijo in hijos:
        for i in range(len(hijo)):
            if np.random.rand() < prob_mutacion:
                mut+= 1
                hijo[i] += np.random.normal(0, sigma)
    return hijos, mut


def mutacion_intercambio(hijos, prob_mutacion):
    mut= 0
    for hijo in hijos:
        for i in range(len(hijo)):
            if np.random.rand() < prob_mutacion:
                mut+= 1
                c1 = np.random.randint(0, len(hijo))  # Cambié 9 por len(hijo)
                c2 = np.random.randint(0, len(hijo))  # Cambié 9 por len(hijo)
                # Se escoge un valor aleatorio en el intervalo [-10.0, 10.0]
                aux = hijo[c1]
                hijo[c1] = hijo[c2]
                hijo[c2] = aux
    return hijos, mut


def mutacion_uniforme(hijos, prob_mutacion, valor_min=-10.0, valor_max=10.0):
    mut= 0
    for hijo in hijos:
        for i in range(len(hijo)):
            if np.random.rand() < prob_mutacion:
                mut+= 1
                gen = np.random.randint(0, len(hijo))  # Cambié 9 por len(hijo)
                # Se escoge un valor aleatorio en el intervalo [-10.0, 10.0]
                valor = np.random.uniform(valor_min, valor_max)
                hijo[gen] = valor
    return hijos, mut

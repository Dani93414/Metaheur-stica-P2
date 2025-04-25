import numpy as np

def seleccion_torneo(poblacion, fitness, tamaño):
    seleccionados = []
    for _ in range(tamaño):
        i, j = np.random.randint(0, len(poblacion), 2)
        ganador = poblacion[i] if fitness[i] < fitness[j] else poblacion[j]
        seleccionados.append(ganador)
    return seleccionados

def seleccion_ruleta(poblacion, fitness, tamaño):
    inv_fitness = 1 / (np.array(fitness) + 1e-8)  # evitar división por cero
    prob = inv_fitness / np.sum(inv_fitness)
    indices = np.random.choice(len(poblacion), size=tamaño, p=prob)
    return [poblacion[i] for i in indices]

def seleccion_muestreo_estocastico(poblacion, fitness, tamaño):
    inv_fitness = 1 / (np.array(fitness) + 1e-8)
    prob = inv_fitness / np.sum(inv_fitness)
    puntos = np.linspace(0, 1, tamaño, endpoint=False) + np.random.uniform(0, 1 / tamaño)
    acumuladas = np.cumsum(prob)
    seleccionados = []
    i = 0
    for p in puntos:
        while p > acumuladas[i]:
            i += 1
        seleccionados.append(poblacion[i])
    return seleccionados

def seleccion_emparejamiento_inverso(poblacion, fitness, tamaño):
    # Selección aleatoria de parejas, elige el PEOR de cada par
    seleccionados = []
    for _ in range(tamaño):
        i, j = np.random.randint(0, len(poblacion), 2)
        perdedor = poblacion[i] if fitness[i] > fitness[j] else poblacion[j]
        seleccionados.append(perdedor)
    return seleccionados

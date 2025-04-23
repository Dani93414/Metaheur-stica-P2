import numpy as np

# ---------- INICIALIZACIÓN ----------
def initialize_population(size, bounds=(-5, 5)):
    return [np.random.uniform(bounds[0], bounds[1], 8) for _ in range(size)]

# ---------- EVALUACIÓN ----------
def evaluate(individual, x_vals, y_vals):
    a, b, c, d, e, g, h, i = individual
    preds = np.exp(a + b*x_vals + c*x_vals**2 + d*x_vals**3 + e*x_vals**4 +
                   g*x_vals**5 + h*x_vals**6 + i*x_vals**7)
    return np.mean((preds - y_vals)**2)

# ---------- CRUCE ----------
def cruce_uniforme(padres, prob_cruce):
    hijos = []
    for i in range(0, len(padres), 2):
        p1 = padres[i]
        p2 = padres[i+1]
        if np.random.rand() < prob_cruce:
            mask = np.random.rand(len(p1)) < 0.5
            hijo1 = np.where(mask, p1, p2)
            hijo2 = np.where(mask, p2, p1)
        else:
            hijo1, hijo2 = p1.copy(), p2.copy()
        hijos.extend([hijo1, hijo2])
    return hijos

# (a implementar)
# def cruce_blx(padres, prob_cruce): ...
# def cruce_ejemplo() ...

# ---------- MUTACIÓN ----------
def mutacion_gaussiana(hijos, prob_mutacion, sigma=0.1):
    for hijo in hijos:
        for i in range(len(hijo)):
            if np.random.rand() < prob_mutacion:
                hijo[i] += np.random.normal(0, sigma)
    return hijos

# (a implementar)
# def mutacion_intercambio(hijos, prob_mutacion): ...
# def mutacion_uniforme(hijos, prob_mutacion): ...

# ---------- ALGORITMO PRINCIPAL ----------
def algoritmo_genetico(n_generaciones, tamaño_poblacion, prob_cruce, prob_mutacion,
                       x_vals, y_vals, op_cruce, op_mutacion):
    
    population = initialize_population(tamaño_poblacion)

    for gen in range(n_generaciones):
        fitness = np.array([evaluate(ind, x_vals, y_vals) for ind in population])

        # Selección: torneo binario simple
        padres = []
        for _ in range(tamaño_poblacion):
            i, j = np.random.randint(0, tamaño_poblacion, 2)
            ganador = population[i] if fitness[i] < fitness[j] else population[j]
            padres.append(ganador)

        # Operador de cruce según selección
        if op_cruce == 1:
            hijos = cruce_uniforme(padres, prob_cruce)
        elif op_cruce == 2:
            raise NotImplementedError("Cruce por un punto (a implementar)")
        elif op_cruce == 3:
            raise NotImplementedError("Cruce BLX-alpha (a implementar)")
        else:
            raise ValueError("Opción de cruce inválida")

        # Operador de mutación según selección
        if op_mutacion == 1:
            hijos_mutados = mutacion_gaussiana(hijos, prob_mutacion)
        elif op_mutacion == 2:
            raise NotImplementedError("Mutación por intercambio (a implementar)")
        elif op_mutacion == 3:
            raise NotImplementedError("Mutación uniforme (a implementar)")
        else:
            raise ValueError("Opción de mutación inválida")

        # Evaluación de descendientes
        hijos_fitness = np.array([evaluate(ind, x_vals, y_vals) for ind in hijos_mutados])

        # Elitismo: conservar el mejor de la población anterior
        elite_idx = np.argmin(fitness)
        elite = population[elite_idx]
        elite_fit = fitness[elite_idx]

        # Reemplazo con elite
        population = hijos_mutados
        population[np.random.randint(len(population))] = elite

        # Estadísticas
        print(f"Generación {gen}: Mejor error = {elite_fit:.5f}")

    # Resultado final
    best_idx = np.argmin([evaluate(ind, x_vals, y_vals) for ind in population])
    best_ind = population[best_idx]
    best_fit = evaluate(best_ind, x_vals, y_vals)

    print("\nMejor individuo encontrado:")
    print(f"Coeficientes: {np.round(best_ind, 4)}")
    print(f"Error: {best_fit:.5f}")

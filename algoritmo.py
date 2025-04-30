import numpy as np
from operadores import cruce as cr
from operadores import mutacion as mt
from operadores import seleccion as sel

# ---------- INICIALIZACIÓN ----------
def initialize_population(size, bounds=(-5, 5)):
    return [np.random.uniform(bounds[0], bounds[1], 8) for _ in range(size)]

# ---------- EVALUACIÓN ----------
def evaluate(individual, x_vals, y_vals):
    a, b, c, d, e, g, h, i = individual
    preds = np.exp(a) + b*x_vals + c*x_vals**2 + d*x_vals**3 + e*x_vals**4 + \
            g*x_vals**5 + h*x_vals**6 + i*x_vals**7
    return np.mean((preds - y_vals)**2)

# ---------- ALGORITMO PRINCIPAL ----------
def algoritmo_genetico(n_generaciones, tamaño_poblacion, prob_cruce, prob_mutacion,
                       x_vals, y_vals, op_cruce, op_mutacion, op_seleccion):

    population = initialize_population(tamaño_poblacion)

    # Estadísticas por generación
    historico_mejores = []
    historico_promedios = []

    # Registro del mejor individuo global
    best_global = None
    best_global_fit = float('inf')

    for gen in range(n_generaciones):
        fitness = np.array([evaluate(ind, x_vals, y_vals) for ind in population])

        # Guardar estadísticas
        mejor_fitness = np.min(fitness)
        promedio_fitness = np.mean(fitness)
        historico_mejores.append(mejor_fitness)
        historico_promedios.append(promedio_fitness)

        # Actualizar el mejor individuo global
        if mejor_fitness < best_global_fit:
            best_global_fit = mejor_fitness
            best_global = population[np.argmin(fitness)]

        # Mensajes informativos
        print(f"Generación {gen}: Mejor error = {mejor_fitness:.5f}, Error promedio = {promedio_fitness:.5f}")

        # Selección
        if op_seleccion == 1:
            padres = sel.seleccion_torneo(population, fitness, tamaño_poblacion)
        elif op_seleccion == 2:
            padres = sel.seleccion_ruleta(population, fitness, tamaño_poblacion)
        elif op_seleccion == 3:
            padres = sel.seleccion_muestreo_estocastico(population, fitness, tamaño_poblacion)
        elif op_seleccion == 4:
            padres = sel.seleccion_emparejamiento_inverso(population, fitness, tamaño_poblacion)
        else:
            raise ValueError("Opción de selección no válida")

        # Cruce
        if op_cruce == 1:
            hijos = cr.cruce_uniforme(padres, prob_cruce)
        elif op_cruce == 2:
            hijos = cr.cruce_un_punto(padres, prob_cruce)
        elif op_cruce == 3:
            hijos = cr.cruce_blx(padres, prob_cruce)
        elif op_cruce == 4:
            hijos = cr.cruce_aritmetico_simple(padres, prob_cruce)
        else:
            raise ValueError("Opción de cruce no válida")

        # Mutación adaptativa
        prob_mut_actual = prob_mutacion * (1 - gen / n_generaciones)

        if op_mutacion == 1:
            hijos_mutados = mt.mutacion_gaussiana(hijos, prob_mut_actual)
        elif op_mutacion == 2:
            hijos_mutados = mt.mutacion_intercambio(hijos, prob_mut_actual)
        elif op_mutacion == 3:
            hijos_mutados = mt.mutacion_uniforme(hijos, prob_mut_actual)
        else:
            raise ValueError("Opción de mutación no válida")

        # Evaluación descendientes
        hijos_fitness = np.array([evaluate(ind, x_vals, y_vals) for ind in hijos_mutados])

        # Elitismo
        elite_idx = np.argmin(fitness)
        elite = population[elite_idx]
        elite_fit = fitness[elite_idx]

        # Reemplazar al peor de los hijos con la élite
        worst_idx = np.argmax(hijos_fitness)
        hijos_mutados[worst_idx] = elite
        hijos_fitness[worst_idx] = elite_fit

        # Actualizar población
        population = hijos_mutados

    # Resultado final: usar el mejor individuo global
    print("\nMejor individuo encontrado globalmente:")
    print(f"Coeficientes: {np.round(best_global, 4)}")
    print(f"Error: {best_global_fit:.5f}")

    # Devolver estadísticas por si quieres graficar
    return best_global, historico_mejores, historico_promedios

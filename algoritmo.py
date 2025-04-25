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
    preds = np.exp(a + b*x_vals + c*x_vals**2 + d*x_vals**3 + e*x_vals**4 +
                   g*x_vals**5 + h*x_vals**6 + i*x_vals**7)
    return np.mean((preds - y_vals)**2)

# ---------- ALGORITMO PRINCIPAL ----------
def algoritmo_genetico(n_generaciones, tamaño_poblacion, prob_cruce, prob_mutacion,
                       x_vals, y_vals, op_cruce, op_mutacion, op_seleccion):
    
    population = initialize_population(tamaño_poblacion)

    for gen in range(n_generaciones):
        fitness = np.array([evaluate(ind, x_vals, y_vals) for ind in population])

        # Operadores de selección
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

        # Operadores de cruce
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

        # Operador de mutación
        if op_mutacion == 1:
            hijos_mutados = mt.mutacion_gaussiana(hijos, prob_mutacion)
        elif op_mutacion == 2:
            hijos_mutados = mt.mutacion_intercambio(hijos, prob_mutacion)
        elif op_mutacion == 3:
            hijos_mutados = mt.mutacion_uniforme(hijos, prob_mutacion)
        else:
            raise ValueError("Opción de mutación no válida")

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

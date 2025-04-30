import numpy as np
import os
import time
from algoritmo import algoritmo_genetico
import graficas as gr
from graficas import (
    plot_mejor_fitness,
    plot_boxplot_fitness_final,
    plot_tiempos,
    plot_fitness_vs_poblacion,
    plot_generaciones_convergencia,
    plot_error_medio_generacion,
    plot_real_vs_predicho
)

# --- MENÚ de selección de operadores ---

print(" Selección de operadores\n")

print("¿Qué deseas hacer?")
print("1. Elegir combinación manualmente")
print("2. Usar combinación predefinida")

opcion_general = int(input("Opción [1-2]: "))

if opcion_general == 1:
    # --- Manual ---
    print("\nSelecciona método de selección:")
    print("1. Torneo binario")
    print("2. Ruleta")
    print("3. Muestreo Estocástico Universal")
    print("4. Emparejamiento variado inverso")
    op_seleccion = int(input("Opción [1-4]: "))

    print("\nSelecciona operador de cruce:")
    print("1. Cruce uniforme")
    print("2. Cruce por un punto")
    print("3. Cruce BLX-alpha")
    print("4. Cruce aritmético simple")
    op_cruce = int(input("Opción [1-4]: "))

    print("\nSelecciona operador de mutación:")
    print("1. Mutación gaussiana")
    print("2. Mutación por intercambio")
    print("3. Mutación uniforme")
    op_mutacion = int(input("Opción [1-3]: "))

elif opcion_general == 2:
    # --- Predefinidas ---
    print("\nSelecciona combinación predefinida:")
    print("1. Torneo + BLX-alpha + Gaussiana (Muy recomendado)")
    print("2. Torneo + Uniforme + Gaussiana")
    print("3. Muestreo Estocástico + BLX-alpha + Uniforme")
    print("4. Emparejamiento inverso + Aritmético simple + Gaussiana")
    print("5. Ruleta + Cruce por un punto + Mutación por intercambio (Exploración más agresiva)")
    op_combo = int(input("Opción [1-5]: "))

    if op_combo == 1:
        op_seleccion = 1
        op_cruce = 3
        op_mutacion = 1
    elif op_combo == 2:
        op_seleccion = 1
        op_cruce = 1
        op_mutacion = 1
    elif op_combo == 3:
        op_seleccion = 3
        op_cruce = 3
        op_mutacion = 3
    elif op_combo == 4:
        op_seleccion = 4
        op_cruce = 4
        op_mutacion = 1
    elif op_combo == 5:
        op_seleccion = 2
        op_cruce = 2
        op_mutacion = 2
    else:
        raise ValueError("Opción predefinida no válida")
else:
    raise ValueError("Opción general no válida")

# --- Parámetros globales ---
np.random.seed(42)

x_vals = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.3, -0.1, -1.6, -1.7, -0.83, -0.82, -1.98, -1.99])

y_vals = np.array([
    3.490342957, 3.649406057, 3.850310157, 4.110680257, 4.444613357, 4.864490457,
    8.945762957, 14.907555257, 3.3508758574, -10.443986642, -10.134869742,
    -0.0700854481, 0.0372406176, -0.2501897336, 0.4626335969
])

# --- Configuración de ejecución ---
n_ejecuciones = 10
n_generaciones = 100
tamaño_poblacion = 30
prob_cruce = 0.9
prob_mutacion = 0.1

# --- Variables para almacenar resultados ---
resultados_fitness = []
fitness_final = []
tiempos = []
generaciones_convergencia = []
mejores_individuos = []

# --- Ejecuciones múltiples ---
for ejec in range(n_ejecuciones):
    start = time.time()
    best_ind, historico_mejores, historico_promedios = algoritmo_genetico(
        n_generaciones=n_generaciones,
        tamaño_poblacion=tamaño_poblacion,
        prob_cruce=prob_cruce,
        prob_mutacion=prob_mutacion,
        x_vals=x_vals,
        y_vals=y_vals,
        op_cruce=op_cruce,
        op_mutacion=op_mutacion,
        op_seleccion=op_seleccion
    )
    end = time.time()

    resultados_fitness.append(historico_mejores)
    fitness_final.append(historico_mejores[-1])
    tiempos.append(end - start)
    mejores_individuos.append(best_ind)

    # Detectar convergencia (cuando mejora menos de un umbral pequeño)
    for i in range(1, len(historico_mejores)):
        if abs(historico_mejores[i] - historico_mejores[i-1]) < 1e-6:
            generaciones_convergencia.append(i)
            break
    else:
        generaciones_convergencia.append(n_generaciones)

# --- Guardar resultados ---
nombre_combo = f"Sel{op_seleccion}_Cru{op_cruce}_Mut{op_mutacion}"
directorio = f"resultados/{nombre_combo}"
os.makedirs(directorio, exist_ok=True)

# --- Gráficas principales ---

gr.plot_mejor_fitness({nombre_combo: resultados_fitness}, save_path=directorio+"/mejor_fitness.png")
gr.plot_boxplot_fitness_final({nombre_combo: fitness_final}, save_path=directorio+"/boxplot_fitness_final.png")
gr.plot_tiempos({nombre_combo: tiempos}, save_path=directorio+"/tiempos_ejecucion.png")

# Opcional: Real vs Predicho con mejor individuo promedio
mejor_individuo = mejores_individuos[np.argmin(fitness_final)]
a, b, c, d, e, g, h, i = mejor_individuo
y_pred = np.exp(a) + b*x_vals + c*x_vals**2 + d*x_vals**3 + e*x_vals**4 + g*x_vals**5 + h*x_vals**6 + i*x_vals**7

gr.plot_real_vs_predicho(x_vals, y_vals, y_pred, save_path=directorio+"/real_vs_predicho.png")

# --- Nuevas gráficas pedidas ---
# Fitness vs Tamaño de Población (aqui solo un punto)
poblaciones_vs_fitness = {tamaño_poblacion: np.min(fitness_final)}
gr.plot_fitness_vs_poblacion(poblaciones_vs_fitness, save_path=directorio+"/fitness_vs_poblacion.png")

# Generaciones hasta convergencia
convergencia_por_combo = {nombre_combo: generaciones_convergencia}
gr.plot_generaciones_convergencia(convergencia_por_combo, save_path=directorio+"/generaciones_convergencia.png")

# Error medio por generación
gr.plot_error_medio_generacion({nombre_combo: resultados_fitness}, save_path=directorio+"/error_medio_generacion.png")
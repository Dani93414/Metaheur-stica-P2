import numpy as np
from algoritmo import algoritmo_genetico

# --- MENÚ de selección de operadores ---

print(" Selección de operadores")
# --- Selección ---
print("\nSelecciona método de selección:")
print("1. Torneo binario")
print("2. Ruleta")
print("3. Muestreo Estocástico Universal")
print("4. Emparejamiento variado inverso")
op_seleccion = int(input("Opción [1-4]: "))

# --- Cruce ---
print("\nSelecciona operador de cruce:")
print("1. Cruce uniforme (implementado)")
print("2. Cruce por un punto (a implementar)")
print("3. Cruce BLX-alpha (a implementar)")
op_cruce = int(input("Opción [1-3]: "))

# --- Mutación ---
print("\nSelecciona operador de mutación:")
print("1. Mutación gaussiana (implementado)")
print("2. Mutación por intercambio (a implementar)")
print("3. Mutación uniforme (a implementar)")
op_mutacion = int(input("Opción [1-3]: "))

# --- Parámetros globales ---
np.random.seed(42)

x_vals = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.3, -0.1, -1.6, -1.7, -0.83, -0.82, -1.98, -1.99])

y_vals = np.array([
    3.490342957, 3.649406057, 3.850310157, 4.110680257, 4.444613357, 4.864490457,
    8.945762957, 14.907555257, 3.3508758574, -10.443986642, -10.134869742,
    -0.0700854481, 0.0372406176, -0.2501897336, 0.4626335969
])

# Ejecutar el algoritmo
algoritmo_genetico(
    n_generaciones=100,
    tamaño_poblacion=30,
    prob_cruce=0.9,
    prob_mutacion=0.1,
    x_vals=x_vals,
    y_vals=y_vals,
    op_cruce=op_cruce,
    op_mutacion=op_mutacion,
    op_seleccion=op_seleccion
)
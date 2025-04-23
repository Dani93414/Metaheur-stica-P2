import numpy as np
from algoritmo import algoritmo_genetico

# --- MENÚ de selección de operadores ---

print(" Selección de operadores")

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

# Simulación de datos (puedes cargar los tuyos si ya los tienes)
x_vals = np.linspace(-1, 1, 100)
y_vals = np.exp(1 + 2*x_vals + 0.5*x_vals**2 - 0.2*x_vals**3 + x_vals**4 - 0.3*x_vals**5 + 0.1*x_vals**6 - 0.05*x_vals**7)

# Ejecutar el algoritmo
algoritmo_genetico(
    n_generaciones=100,
    tamaño_poblacion=30,
    prob_cruce=0.9,
    prob_mutacion=0.1,
    x_vals=x_vals,
    y_vals=y_vals,
    op_cruce=op_cruce,
    op_mutacion=op_mutacion
)

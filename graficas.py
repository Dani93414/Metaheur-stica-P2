import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Datos esperados:
# - resultados_fitness: dict["combo"] = lista del mejor fitness por generación (por ejecución)
# - fitness_final: dict["combo"] = lista del fitness final por ejecución
# - tiempos: dict["combo"] = lista de tiempos de ejecución por ejecución
# - generaciones_convergencia: dict["combo"] = lista de generaciones necesarias para converger
# - poblaciones_vs_fitness: dict[tamaño_poblacion] = mejor fitness encontrado
# - y_true: array de valores reales
# - y_pred: array de valores predichos (por el mejor individuo)


def plot_fitness_y_error_juntos(resultados_fitness, save_path=None):
    plt.figure(figsize=(10,6))
    for combo, listas_fitness in resultados_fitness.items():
        mejor_fitness = np.min(listas_fitness, axis=0)
        error_medio = np.mean(listas_fitness, axis=0)

        plt.plot(mejor_fitness, label=f'{combo} - Mejor Fitness', color='blue')
        plt.plot(error_medio, label=f'{combo} - Error Medio', color='orange', linestyle='--')

    plt.xlabel('Generaciones')
    plt.ylabel('Valor')
    plt.title('Mejor Fitness y Error Medio por Generación')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_boxplot_fitness_final(fitness_final, save_path=None):
    data = []
    labels = []
    for combo, fitness_list in fitness_final.items():
        data.append(fitness_list)
        labels.append(combo)
    plt.figure(figsize=(10,6))
    sns.boxplot(data=data)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.ylabel('Fitness Final')
    plt.title('Distribución del Fitness Final por Combinación')
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_tiempos(tiempos, save_path=None):
    promedio_tiempos = {combo: np.mean(tiempos_list) for combo, tiempos_list in tiempos.items()}
    combos = list(promedio_tiempos.keys())
    valores = list(promedio_tiempos.values())
    plt.figure(figsize=(10,6))
    sns.barplot(x=combos, y=valores)
    plt.xticks(rotation=45)
    plt.ylabel('Tiempo Promedio (s)')
    plt.title('Tiempo Promedio de Ejecución por Combinación')
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_fitness_vs_poblacion(poblaciones_vs_fitness, save_path=None):
    poblaciones = list(poblaciones_vs_fitness.keys())
    fitness = list(poblaciones_vs_fitness.values())
    plt.figure(figsize=(10,6))
    plt.plot(poblaciones, fitness, marker='o')
    plt.xlabel('Tamaño de Población')
    plt.ylabel('Mejor Fitness Encontrado')
    plt.title('Fitness vs Tamaño de Población')
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_generaciones_convergencia(generaciones_convergencia, save_path=None):
    promedio_gen = {combo: np.mean(lista_gen) for combo, lista_gen in generaciones_convergencia.items()}
    combos = list(promedio_gen.keys())
    valores = list(promedio_gen.values())
    plt.figure(figsize=(10,6))
    sns.barplot(x=combos, y=valores)
    plt.xticks(rotation=45)
    plt.ylabel('Generaciones hasta Convergencia')
    plt.title('Generaciones Necesarias hasta Convergencia')
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_real_vs_predicho(x_vals, y_true, y_pred, save_path=None):
    plt.figure(figsize=(10,6))
    plt.scatter(x_vals, y_true, color='red', label='Valores reales', zorder=5)
    sorted_indices = np.argsort(x_vals)
    plt.plot(x_vals[sorted_indices], y_pred[sorted_indices], color='blue', label='Predicción modelo')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Comparación de Reales vs Predicho')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def graficar_poblacion_por_generacion(poblaciones, save_path=None):
    generaciones = list(range(1, len(poblaciones) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(generaciones, poblaciones, marker='o', linestyle='-', color='blue')
    plt.title('Evolución del Tamaño de Población por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Cantidad de Individuos')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def graficar_cruces_y_mutaciones(cruces, mutaciones, save_path=None):
    generaciones = list(range(1, len(cruces) + 1))
    
    plt.figure(figsize=(10, 6))
    
    # Líneas con menor grosor y color más estilizado
    plt.plot(generaciones, cruces, marker='o', linestyle='-', linewidth=1.5, color='#1f77b4', label='Cruces')
    plt.plot(generaciones, mutaciones, marker='s', linestyle='--', linewidth=1.5, color='#ff7f0e', label='Mutaciones')
    
    plt.title('Cruces y Mutaciones por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Cantidad')
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

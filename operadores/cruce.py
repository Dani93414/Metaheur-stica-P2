import numpy as np

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


def cruce_aritmetico_simple(padres, prob_cruce):
    hijos = []
    for i in range(0, len(padres), 2):
        p1 = padres[i]
        p2 = padres[i+1]
        if np.random.rand() < prob_cruce:
            hijo1 = []
            for j in range(0, len(p1)):
                hijo1[j]= (p1[j] + p2[j])/ 2

            # Se mantiene uno de los dos padres de manera aleatoria
            padre= np.random.randint(0,2)
            h2= p1.copy() if padre == 0 else p2.copy
        else:
            hijo1, hijo2 = p1.copy(), p2.copy()
        hijos.extend([hijo1, hijo2])
    return hijos
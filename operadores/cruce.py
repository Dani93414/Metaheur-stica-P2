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


def cruce_un_punto(padres, prob_cruce, pto= 4):
    hijos = []
    for i in range(0, len(padres), 2):
        p1 = padres[i]
        p2 = padres[i+1]
        if np.random.rand() < prob_cruce:
            hijo1, hijo2 = []
            for j in range(0, len(p1)):
                if j < pto:
                    hijo1[j]= p1[j]
                    hijo2[j]= p2[j]
                else:
                    hijo1[j]= p2[j]
                    hijo2[j]= p1[j]
        else:
            hijo1, hijo2 = p1.copy(), p2.copy()
        hijos.extend([hijo1, hijo2])
    return hijos


def cruce_blx(padres, prob_cruce, alpha=0.1):
    hijos = []
    for i in range(0, len(padres), 2):
        p1 = padres[i]
        p2 = padres[i+1]
        if np.random.rand() < prob_cruce:
            hijo1, hijo2 = []

            for j in range(0, len(p1)):
                c_max= max(hijo1[j], hijo2[j])
                c_min= min(hijo1[j], hijo2[j])

                I= c_max - c_min

                cota_inf= c_min - (I * alpha)
                cota_sup= c_max + (I * alpha)

                c1= np.random.uniform(cota_inf, cota_sup)
                c2= np.random.uniform(cota_inf, cota_sup)

                hijo1[j]= c1
                hijo2[j]= c2
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
            hijo2= p1.copy() if padre == 0 else p2.copy
        else:
            hijo1, hijo2 = p1.copy(), p2.copy()
        hijos.extend([hijo1, hijo2])
    return hijos
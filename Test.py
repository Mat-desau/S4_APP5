import numpy as np

Frequence = ((np.pi/1000) / (2 * np.pi))* 41000
Valeur_N = 0


for N in range(0, 20):
    Val = (2*np.pi) / (N + 1)
    ValDb = 20 * np.log10(Val)
    print(Val, ValDb)




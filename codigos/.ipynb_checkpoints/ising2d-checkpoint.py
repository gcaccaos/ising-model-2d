from numpy import exp
from numpy.random import choice, permutation, uniform
from numba import jit

@jit
def metropolis(L, N, T, J = 1, h = 0, kB = 1):
    spins = choice([-1, +1], (L, L, N)) # gera uma configuracao aleatoria de spins
    
    # Faz N varreduras sobre o grid
    for n in range(N):
        # Ordem aleatoria para varrer os spins do grid
        ordemi = permutation(L)
        ordemj = permutation(L)
        
        # Varredura sobre os spins e evolucao (flips aleatorios) devido a temperatura
        for i in ordemi:
            for j in ordemj:
                # Calculo da energia para flipar o spin na posicao i
                Eflip = 2*spins[i, j, n]*(J*(spins[(i - 1)%L, j, n] + spins[(i + 1)%L, j, n] +
                                             spins[i, (j - 1)%L, n] + spins[i, (j + 1)%L, n]) + h)

                # Se Eflip < 0, flipa o spin; caso contrario, aplica-se o passo de Monte Carlo
                if Eflip < 0:
                    spins[i, j, n] = - spins[i, j, n]
                else:
                    Pflip = exp(- Eflip/(kB*T)) # probabilidade de flipar o spin na posicao i
                    if uniform(0, 1) < Pflip:
                        spins[i, j, n] = - spins[i, j, n]
    return spins

@jit
def H(spins, J, h):
    H = 0
    return H
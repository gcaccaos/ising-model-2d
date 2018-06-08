from numpy import exp
from numpy.random import choice, permutation, uniform
from numba import jit

@jit
def metropolis(L, N, T, J = 1, h = 0, kB = 1):
    '''Simula configuracoes de spin atraves do algoritmo de Metropolis. Sao realizados passos de Monte Carlo sobre os spins, simulando a interacao com o banho termico. Esse processo e repetido N vezes.
    
    Parametros
    ----------
    L: int
        Largura do grid de spins.
    N: int
        Numero de varreduras de Monte Carlo.
    T: float
        Temperatura do banho termico.
    J: float
        Constante que caracteriza a interacao entre dois spins vizinhos.
    h: float
        Constante que caracteriza a interacao dos spins com o campo magnetico externo.
    kB: float
        Constante de Boltzmann.
    
    Retorna
    -------
    ndarray
        Array de N configuracoes de spin para o grid de tamanho L por L.'''
    
    spins = choice([-1, +1], (L, L, N)) # gera uma configuracao aleatoria de spins
    
    # Faz N varreduras sobre o grid
    for n in range(N):
        # Ordem aleatoria para varrer os spins do grid
        ordemi = permutation(L)
        ordemj = permutation(L)
        
        # Varredura sobre os spins
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
        # Fim de um passo de Monte Carlo (representa uma interacao dos spins com o banho termico)
    return spins

@jit
def H(spins, J = 1, h = 0):
    '''Calcula a energia de uma dada configuracao de spins.'''
    L = spins.shape[0]
    E = 0
    
    for i in range(L):
        for j in range(L):
            E += - spins[i, j]*(J*(spins[(i - 1)%L, j] + spins[(i + 1)%L, j] +
                                  spins[i, (j - 1)%L] + spins[i, (j + 1)%L]) + h)
    return E
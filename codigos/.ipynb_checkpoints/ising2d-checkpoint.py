from numpy import exp, zeros
from numpy.random import choice, permutation, uniform
from numba import jit

@jit
def ising2d(L, N, T, J = 1, h = 0, kB = 1):
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
    
    spins = choice([-1, +1], (L, L, N)) # gera N configuracoes aleatorias de spins
    M = zeros(N)
    E = zeros(N)
    
    # Faz N varreduras sobre o grid
    for n in range(N):
        # Ordem aleatoria para varrer os spins do grid
        ordemi = permutation(L)
        ordemj = permutation(L)
        
        # Varredura sobre os spins
        for i in ordemi:
            for j in ordemj:
                # Calculo da energia para flipar o spin na posicao i
                DeltaE = 2*spins[i, j, n]*(J*(spins[(i - 1)%L, j, n] + spins[(i + 1)%L, j, n] +
                                             spins[i, (j - 1)%L, n] + spins[i, (j + 1)%L, n]) + h)

                # Se Eflip < 0, flipa o spin; caso contrario, aplica-se o passo de Monte Carlo
                if DeltaE < 0:
                    spins[i, j, n] = - spins[i, j, n]
                else:
                    PTrans = exp(- DeltaE/(kB*T)) # probabilidade de flipar o spin
                    if uniform(0, 1) < PTrans:
                        spins[i, j, n] = - spins[i, j, n]
                
                # Atualiza a energia total da configuracao
                E[n] -= DeltaE/2 # dividido por 2 para nao contar duas vezes cada spin
                # Atualiza a magnetizacao total da configuracao
                M[n] += spins[i, j, n]
        
        # Fim de um passo de Monte Carlo (representa uma interacao dos spins com o banho termico)
    return spins, M.mean(), E.mean()
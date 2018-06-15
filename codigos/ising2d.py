from numpy import exp, zeros
from numpy.random import choice, permutation, uniform
from numba import jit
from ipywidgets import interact
from matplotlib.pyplot import subplots, imshow, axis, show

@jit
def ising_update(spins, i, j, T, J, h):
    kB = 1 # constante de Boltzmann (assim, a temperatura e dada em [T]/[kB])
    L = spins.shape[0]
    
    # Calculo da energia para flipar o spin na posicao i, j
    DeltaE = 2*spins[i, j]*(J*(spins[(i - 1)%L, j] + spins[(i + 1)%L, j] +
                               spins[i, (j - 1)%L] + spins[i, (j + 1)%L]) + h)

    # Se Eflip < 0, flipa o spin; caso contrario, aplica-se o passo de Monte Carlo
    if DeltaE < 0:
        spins[i, j] *= -1
    else:
        Pflip = exp(- DeltaE/(kB*T)) # probabilidade de flipar o spin
        if uniform(0, 1) < Pflip:
            spins[i, j] *= -1

@jit
def ising_step(spins, T, J, h):
    '''Simula configuracoes de spin atraves do algoritmo de Metropolis. Sao realizados passos de Monte Carlo sobre os spins, simulando a interacao com o banho termico. Esse processo e repetido N vezes.
    
    Parametros
    ----------
    spins: ndarray
        Configuracao de spins.
    T: float
        Temperatura do banho termico.
    J: float
        Constante que caracteriza a interacao entre dois spins vizinhos.
    h: float
        Constante que caracteriza a interacao dos spins com o campo magnetico externo.
    
    Retorna
    -------
    ndarray
        Array de N configuracoes de spin para o grid de tamanho L por L.'''
    
    # Ordem aleatoria para varrer os spins do grid
    L = spins.shape[0]
    ordemi = permutation(L)
    ordemj = permutation(L)

    # Varredura sobre os spins
    for i in ordemi:
        for j in ordemj:
            ising_update(spins, i, j, T, J, h)
    
    return spins

@jit
def magnetization_per_site(spins):
    L = spins.shape[0]
    M = spins.sum()
    m = M/(L**2)
    
    return m

@jit
def energy_per_site(spins, J, h):
    L = spins.shape[0]
    E = 0
    
    for i in range(L):
        for j in range(L):
            DeltaE = 2*spins[i, j]*(J/4*(spins[(i - 1)%L, j] + spins[(i + 1)%L, j] + # dividido por 4 para nao constar o mesmo spin duas vezes
                                       spins[i, (j - 1)%L] + spins[i, (j + 1)%L]) + h)
            E -= DeltaE
    
    e = E/(L**2)
    
    return e

@jit
def display_spins(ax, spins):
    ax.imshow(spins, cmap = 'binary')
    ax.axis('off')
    show()

def display_ising_sequence(images):
    def update_display(step = (0, len(images) - 1)):
        fig, ax = subplots()
        display_spins(ax, images[step])
    return interact(update_display)
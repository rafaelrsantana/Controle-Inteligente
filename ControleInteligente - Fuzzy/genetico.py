import numpy as np
import random

def otimizar(func, limites, n_pop=20, n_gen=30, mutation_rate=0.2, x0=None):

    print(f"\n--- INICIANDO ALGORITMO GENÉTICO ({n_gen} gerações) ---")

    n_vars = len(limites)
    
    populacao = np.zeros((n_pop, n_vars))
    for i in range(n_pop):
        for j in range(n_vars):
            low, high = limites[j]
            populacao[i, j] = random.uniform(low, high)

    if x0 is not None:
        populacao[0] = np.array(x0)
    custos = np.array([func(ind) for ind in populacao])
    
    history_best_cost = []
    
    for gen in range(n_gen):
    
        indices_ordenados = np.argsort(custos)
        populacao = populacao[indices_ordenados]
        custos = custos[indices_ordenados]
        
        melhor_custo = custos[0]
        melhor_ind = populacao[0]
        history_best_cost.append(melhor_custo)
        
        print(f"Geração {gen+1}/{n_gen}: Melhor J = {melhor_custo:.6f} | Params: {melhor_ind}")
        
        nova_populacao = [melhor_ind.copy()] 
        
        while len(nova_populacao) < n_pop:
            
            pai1 = selecao_torneio(populacao, custos)
            pai2 = selecao_torneio(populacao, custos)
            
            
            filho = crossover_aritmetico(pai1, pai2)
            
            
            filho = mutacao(filho, limites, mutation_rate)
            
            nova_populacao.append(filho)
            
        populacao = np.array(nova_populacao)
        
        custos = np.array([func(ind) for ind in populacao])

    
    best_idx = np.argmin(custos)
    return populacao[best_idx], history_best_cost



def selecao_torneio(pop, custos, k=3):
  
    n_ind = len(pop)
    competidores_idx = np.random.choice(n_ind, k, replace=False)
    
    
    melhor_idx_local = competidores_idx[0]
    melhor_custo_local = custos[melhor_idx_local]
    
    for idx in competidores_idx[1:]:
        if custos[idx] < melhor_custo_local:
            melhor_custo_local = custos[idx]
            melhor_idx_local = idx
            
    return pop[melhor_idx_local]

def crossover_aritmetico(pai1, pai2):

    alpha = np.random.random() 
    filho = alpha * pai1 + (1 - alpha) * pai2
    return filho

def mutacao(ind, limites, rate):

    ind_mutado = ind.copy()
    for i in range(len(ind)):
        if np.random.random() < rate:
            
            low, high = limites[i]
            amplitude = (high - low) * 0.1 
            perturbacao = np.random.uniform(-amplitude, amplitude)
            
            ind_mutado[i] += perturbacao
            
            ind_mutado[i] = np.clip(ind_mutado[i], low, high)
            
    return ind_mutado
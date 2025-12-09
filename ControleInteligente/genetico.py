"""
    Arquivo: genetico.py
    Algoritmo: Algoritmo Genético (AG)
    Descrição: Otimizador inspirado na evolução natural.
    
    Funcionalidades:
    - Representação Real (Float)
    - Seleção por Torneio
    - Crossover Aritmético (Média ponderada)
    - Mutação Gaussiana
    - Elitismo (Preserva o melhor de cada geração)
"""

import numpy as np
import random

def otimizar(func, limites, n_pop=20, n_gen=30, mutation_rate=0.2):
    """
    Executa a otimização por Algoritmo Genético.
    
    Args:
        func (callable): Função custo (run_simulation).
        limites (list of tuples): Ex: [(0, 100), (0, 100), (0, 100)] para Kp, Ki, Kd.
        n_pop (int): Tamanho da população (número de indivíduos).
        n_gen (int): Número de gerações (iterações).
        mutation_rate (float): Probabilidade de um gene sofrer mutação (0.0 a 1.0).
        
    Returns:
        best_ind (array): Melhores parâmetros encontrados.
        history (list): Histórico do melhor custo por geração.
    """
    
    print(f"\n--- INICIANDO ALGORITMO GENÉTICO ({n_gen} gerações) ---")
    
    n_vars = len(limites)
    
    # 1. Inicialização da População
    # Cria matriz [n_pop x n_vars] com valores aleatórios dentro dos limites
    populacao = np.zeros((n_pop, n_vars))
    for i in range(n_pop):
        for j in range(n_vars):
            low, high = limites[j]
            populacao[i, j] = random.uniform(low, high)
            
    # Avalia a população inicial
    custos = np.array([func(ind) for ind in populacao])
    
    history_best_cost = []
    
    for gen in range(n_gen):
        # Classifica do melhor (menor custo) para o pior
        indices_ordenados = np.argsort(custos)
        populacao = populacao[indices_ordenados]
        custos = custos[indices_ordenados]
        
        melhor_custo = custos[0]
        melhor_ind = populacao[0]
        history_best_cost.append(melhor_custo)
        
        print(f"Geração {gen+1}/{n_gen}: Melhor J = {melhor_custo:.6f} | Params: {melhor_ind}")
        
        # ELITISMO: O melhor indivíduo passa direto para a próxima geração
        nova_populacao = [melhor_ind.copy()] 
        
        # Loop para criar o restante da nova população
        while len(nova_populacao) < n_pop:
            
            # 2. Seleção (Torneio)
            pai1 = selecao_torneio(populacao, custos)
            pai2 = selecao_torneio(populacao, custos)
            
            # 3. Crossover (Cruzamento)
            filho = crossover_aritmetico(pai1, pai2)
            
            # 4. Mutação
            filho = mutacao(filho, limites, mutation_rate)
            
            nova_populacao.append(filho)
            
        populacao = np.array(nova_populacao)
        
        # Recalcula custos da nova geração
        # (Nota: poderia otimizar não recalculando o elite, mas mantive simples)
        custos = np.array([func(ind) for ind in populacao])

    # Pega o melhor da última geração
    best_idx = np.argmin(custos)
    return populacao[best_idx], history_best_cost

# --- FUNÇÕES AUXILIARES DO AG ---

def selecao_torneio(pop, custos, k=3):
    """
    Escolhe k indivíduos aleatórios e retorna o melhor deles.
    Simula a 'luta pela sobrevivência'.
    """
    n_ind = len(pop)
    competidores_idx = np.random.choice(n_ind, k, replace=False)
    
    # Acha quem tem o menor custo entre os sorteados
    melhor_idx_local = competidores_idx[0]
    melhor_custo_local = custos[melhor_idx_local]
    
    for idx in competidores_idx[1:]:
        if custos[idx] < melhor_custo_local:
            melhor_custo_local = custos[idx]
            melhor_idx_local = idx
            
    return pop[melhor_idx_local]

def crossover_aritmetico(pai1, pai2):
    """
    Combina os pais fazendo uma média ponderada aleatória.
    Filho = alpha * Pai1 + (1-alpha) * Pai2
    """
    alpha = np.random.random() # Valor entre 0 e 1
    filho = alpha * pai1 + (1 - alpha) * pai2
    return filho

def mutacao(ind, limites, rate):
    """
    Introduz pequenas variações aleatórias para manter diversidade.
    """
    ind_mutado = ind.copy()
    for i in range(len(ind)):
        if np.random.random() < rate:
            # Perturbação: soma um valor aleatório de +/- 10% do range do parametro
            low, high = limites[i]
            amplitude = (high - low) * 0.1 
            perturbacao = np.random.uniform(-amplitude, amplitude)
            
            ind_mutado[i] += perturbacao
            
            # Garante que não fugiu dos limites (Clamping)
            ind_mutado[i] = np.clip(ind_mutado[i], low, high)
            
    return ind_mutado
"""
    Arquivo: poliedro.py
    Algoritmo: Método de Nelder-Mead (Poliedros Flexíveis)
    Descrição: Otimizador numérico para funções não-lineares sem uso de derivadas.
    
    Este módulo é genérico e pode otimizar qualquer função que receba uma lista 
    de números e retorne um custo (float).
"""

import numpy as np

def otimizar(func, x0, tol=1e-4, max_iter=50):
    """
    Executa a otimização por Poliedros Flexíveis.
    
    Args:
        func (callable): A função custo a ser minimizada (ex: run_simulation).
        x0 (list/array): Chute inicial dos parâmetros [p1, p2, ...].
        tol (float): Tolerância para parada antecipada.
        max_iter (int): Número máximo de iterações.
        
    Returns:
        best_params (array): Os melhores parâmetros encontrados.
        history (list): Histórico do melhor custo a cada iteração.
    """
    
    print("\n--- INICIANDO POLIEDRO (Nelder-Mead) ---")
    
    # Parâmetros padrão do método
    alpha = 1.0  # Reflexão
    gamma = 2.0  # Expansão
    beta = 0.5   # Contração
    delta = 0.5  # Redução (Shrink)
    
    x0 = np.array(x0)
    N_dim = len(x0)
    
    # Passo 0: Criação do Simplex Inicial
    # Cria N+1 vértices perturbando o chute inicial
    simplex = [x0]
    perturbation = 0.05 # 5% de perturbação para criar o volume inicial
    
    for i in range(N_dim):
        point = np.array(x0)
        if point[i] == 0:
            point[i] = 0.001
        else:
            point[i] = point[i] * (1 + perturbation)
        simplex.append(point)
    
    # Avalia o custo inicial de todos os vértices
    # Aqui chamamos a função 'func' que veio do sistema.py
    costs = [func(v) for v in simplex]
    
    history_best_cost = []
    
    for k in range(max_iter):
        # Passo 1: Ordenação
        indices = np.argsort(costs)
        simplex = [simplex[i] for i in indices]
        costs = [costs[i] for i in indices]
        
        # Melhores e Piores pontos
        w_L = simplex[0]   # Best (Low cost)
        J_L = costs[0]
        w_H = simplex[-1]  # Worst (High cost)
        J_H = costs[-1]
        w_SH = simplex[-2] # Second Worst
        J_SH = costs[-2]

        history_best_cost.append(J_L)
        
        # Log de progresso
        print(f"Iter {k+1}/{max_iter}: Melhor J = {J_L:.6f} | Params: {w_L}")
        
        # Critério de parada (tamanho do simplex)
        simplex_size = np.sum([np.linalg.norm(w - w_L) for w in simplex])
        if simplex_size < tol:
            print(">>> Convergência atingida (tolerância).")
            break
            
        # Passo 2: Centróide (média de todos menos o pior)
        c = np.mean(simplex[:-1], axis=0)
        
        # Passo 3: Reflexão
        w_R = c + alpha * (c - w_H)
        J_R = func(w_R)
        
        if J_R < J_L:
            # Passo 4: Expansão (A reflexão foi excelente, vamos esticar)
            w_E = c + gamma * (w_R - c)
            J_E = func(w_E)
            
            if J_E < J_R:
                simplex[-1] = w_E
                costs[-1] = J_E
            else:
                simplex[-1] = w_R
                costs[-1] = J_R
                
        elif J_R >= J_SH:
            # A reflexão foi ruim (pior que o segundo pior) -> Contração
            should_shrink = False
            
            if J_R < J_H:
                # Contração Externa
                w_C = c + beta * (w_R - c)
                J_C = func(w_C)
                if J_C <= J_R:
                    simplex[-1] = w_C
                    costs[-1] = J_C
                else:
                    should_shrink = True
            else:
                # Contração Interna
                w_C = c - beta * (w_R - c)
                J_C = func(w_C)
                if J_C < J_H:
                    simplex[-1] = w_C
                    costs[-1] = J_C
                else:
                    should_shrink = True
            
            # Passo 5: Redução (Shrink) se a contração falhou
            if should_shrink:
                for i in range(1, len(simplex)):
                    simplex[i] = w_L + delta * (simplex[i] - w_L)
                    costs[i] = func(simplex[i])
        else:
            # Reflexão normal (aceita w_R)
            simplex[-1] = w_R
            costs[-1] = J_R

    return simplex[0], history_best_cost
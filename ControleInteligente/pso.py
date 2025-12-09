"""
    Arquivo: pso.py
    Algoritmo: Otimização por Enxame de Partículas (PSO)
    Descrição: Otimizador inspirado no comportamento social de pássaros/peixes.
    
    Cada "partícula" voa pelo espaço de busca ajustando sua rota com base em:
    1. Sua própria melhor experiência (pbest)
    2. A melhor experiência do grupo (gbest)
"""

import numpy as np

def otimizar(func, limites, n_particulas=30, n_iter=50, w=0.7, c1=1.5, c2=1.5):
    """
    Executa a otimização PSO.
    
    Args:
        func (callable): Função custo (run_simulation).
        limites (list of tuples): Limites para Kp, Ki, Kd [(min, max), ...].
        n_particulas (int): Número de partículas no enxame.
        n_iter (int): Número de iterações.
        w (float): Peso de inércia (o quanto a partícula tende a manter sua velocidade).
        c1 (float): Coeficiente cognitivo (atração pela própria melhor memória).
        c2 (float): Coeficiente social (atração pelo melhor líder do grupo).
        
    Returns:
        gbest_position (array): Melhores parâmetros encontrados.
        history (list): Histórico do melhor custo a cada iteração.
    """
    
    print(f"\n--- INICIANDO PSO ({n_iter} iterações) ---")
    
    n_vars = len(limites)
    limites = np.array(limites)
    min_vals = limites[:, 0]
    max_vals = limites[:, 1]
    
    # 1. Inicialização
    # Posições aleatórias dentro dos limites
    positions = np.random.uniform(low=min_vals, high=max_vals, size=(n_particulas, n_vars))
    
    # Velocidades iniciais (geralmente zeradas ou pequenas aleatórias)
    velocities = np.zeros((n_particulas, n_vars))
    
    # Melhores posições pessoais (pbest) e seus custos
    pbest_positions = positions.copy()
    pbest_costs = np.full(n_particulas, float('inf'))
    
    # Melhor posição global (gbest)
    gbest_position = np.zeros(n_vars)
    gbest_cost = float('inf')
    
    history_best_cost = []
    
    # Avaliação Inicial
    for i in range(n_particulas):
        cost = func(positions[i])
        
        # Atualiza Pbest (pessoal)
        if cost < pbest_costs[i]:
            pbest_costs[i] = cost
            pbest_positions[i] = positions[i]
            
        # Atualiza Gbest (global)
        if cost < gbest_cost:
            gbest_cost = cost
            gbest_position = positions[i].copy()
            
    history_best_cost.append(gbest_cost)
    print(f"Iteração 0/{n_iter}: Melhor J = {gbest_cost:.6f}")
    
    # 2. Loop Principal
    for it in range(n_iter):
        for i in range(n_particulas):
            # A. Atualização da Velocidade
            # v = w*v + c1*rand*(pbest - x) + c2*rand*(gbest - x)
            r1 = np.random.random(n_vars)
            r2 = np.random.random(n_vars)
            
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest_positions[i] - positions[i]) + 
                             c2 * r2 * (gbest_position - positions[i]))
            
            # B. Atualização da Posição
            positions[i] = positions[i] + velocities[i]
            
            # C. Restrição de Limites (Clamping)
            # Se a partícula sair do mapa, trazemos ela de volta para a borda
            positions[i] = np.clip(positions[i], min_vals, max_vals)
            
            # D. Avaliação
            cost = func(positions[i])
            
            # Atualiza Pbest
            if cost < pbest_costs[i]:
                pbest_costs[i] = cost
                pbest_positions[i] = positions[i]
                
                # Atualiza Gbest
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_position = positions[i].copy()
        
        history_best_cost.append(gbest_cost)
        print(f"Iteração {it+1}/{n_iter}: Melhor J = {gbest_cost:.6f}")
        
    return gbest_position, history_best_cost
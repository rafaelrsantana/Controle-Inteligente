import numpy as np

def otimizar(func, limites, n_particulas=30, n_iter=50, w=0.7, c1=1.5, c2=1.5, x0 = None):

    print(f"\n--- INICIANDO PSO ({n_iter} iterações) ---")
    
    n_vars = len(limites)
    limites = np.array(limites)
    min_vals = limites[:, 0]
    max_vals = limites[:, 1]
    
    positions = np.random.uniform(low=min_vals, high=max_vals, size=(n_particulas, n_vars))
    
    if x0 is not None:
        positions[0] = np.array(x0)

    velocities = np.zeros((n_particulas, n_vars))
    
    pbest_positions = positions.copy()
    pbest_costs = np.full(n_particulas, float('inf'))
    
    gbest_position = np.zeros(n_vars)
    gbest_cost = float('inf')
    
    history_best_cost = []
    
    for i in range(n_particulas):
        cost = func(positions[i])
        
        if cost < pbest_costs[i]:
            pbest_costs[i] = cost
            pbest_positions[i] = positions[i]
            
        if cost < gbest_cost:
            gbest_cost = cost
            gbest_position = positions[i].copy()
            
    history_best_cost.append(gbest_cost)
    print(f"Iteração 0/{n_iter}: Melhor J = {gbest_cost:.6f}")
    
    for it in range(n_iter):
        for i in range(n_particulas):
            
            r1 = np.random.random(n_vars)
            r2 = np.random.random(n_vars)
            
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest_positions[i] - positions[i]) + 
                             c2 * r2 * (gbest_position - positions[i]))
            

            positions[i] = positions[i] + velocities[i]
            
            positions[i] = np.clip(positions[i], min_vals, max_vals)
            
            cost = func(positions[i])
            
            if cost < pbest_costs[i]:
                pbest_costs[i] = cost
                pbest_positions[i] = positions[i]
                  
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_position = positions[i].copy()
        
        history_best_cost.append(gbest_cost)
        print(f"Iteração {it+1}/{n_iter}: Melhor J = {gbest_cost:.6f}")
        
    return gbest_position, history_best_cost
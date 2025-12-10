import numpy as np

def otimizar(func, x0, tol=1e-4, max_iter=50):

    print("\n--- INICIANDO POLIEDRO (Nelder-Mead) ---")
    
    alpha = 1.0  
    gamma = 2.0  
    beta = 0.5   
    delta = 0.5  
    
    x0 = np.array(x0)
    N_dim = len(x0)
    
    simplex = [x0]
    perturbation = 0.05 
    
    for i in range(N_dim):
        point = np.array(x0)
        if point[i] == 0:
            point[i] = 0.001
        else:
            point[i] = point[i] * (1 + perturbation)
        simplex.append(point)
    
    costs = [func(v) for v in simplex]
    
    history_best_cost = []
    
    for k in range(max_iter):
        
        indices = np.argsort(costs)
        simplex = [simplex[i] for i in indices]
        costs = [costs[i] for i in indices]
            
        w_L = simplex[0]   
        J_L = costs[0]
        w_H = simplex[-1]  
        J_H = costs[-1]
        w_SH = simplex[-2] 
        J_SH = costs[-2]

        history_best_cost.append(J_L)
        
        print(f"Iter {k+1}/{max_iter}: Melhor J = {J_L:.6f} | Params: {w_L}")
        
        simplex_size = np.sum([np.linalg.norm(w - w_L) for w in simplex])
        if simplex_size < tol:
            print(">>> Convergência atingida (tolerância).")
            break
            
        c = np.mean(simplex[:-1], axis=0)
        
        w_R = c + alpha * (c - w_H)
        J_R = func(w_R)
        
        if J_R < J_L:
            
            w_E = c + gamma * (w_R - c)
            J_E = func(w_E)
            
            if J_E < J_R:
                simplex[-1] = w_E
                costs[-1] = J_E
            else:
                simplex[-1] = w_R
                costs[-1] = J_R
                
        elif J_R >= J_SH:
            
            should_shrink = False
            
            if J_R < J_H:
                
                w_C = c + beta * (w_R - c)
                J_C = func(w_C)
                if J_C <= J_R:
                    simplex[-1] = w_C
                    costs[-1] = J_C
                else:
                    should_shrink = True
            else:
                
                w_C = c - beta * (w_R - c)
                J_C = func(w_C)
                if J_C < J_H:
                    simplex[-1] = w_C
                    costs[-1] = J_C
                else:
                    should_shrink = True
            
            
            if should_shrink:
                for i in range(1, len(simplex)):
                    simplex[i] = w_L + delta * (simplex[i] - w_L)
                    costs[i] = func(simplex[i])
        else:
            
            simplex[-1] = w_R
            costs[-1] = J_R

    return simplex[0], history_best_cost
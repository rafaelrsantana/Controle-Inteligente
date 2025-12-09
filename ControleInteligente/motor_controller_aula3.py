"""
    This code has been developed by Juan Sandubete Lopez and all the rights
    belongs to him.
    Distribution or commercial use of the code is not allowed without previous
    agreement with the author.
"""

import numpy as np
import matplotlib.pyplot as plt
# odeint removido do loop, não precisamos mais dele para o passo-a-passo
import time

# --- CONFIGURAÇÕES GERAIS ---
tf = 10.0
ts_ms = 0.01
n = int((1 / (ts_ms / 1000.0))*tf + 1)
time_vector = np.linspace(0, tf, n)
t_sim_step = time_vector[1] - time_vector[0] # Passo de tempo (dt)

# Referências
torque_ref = np.sin(time_vector)
torquep_ref = np.cos(time_vector)

# Parâmetros do Modelo (Fixos)
a = 1
k = 1
a_model_error = 0.1
k_model_error = 0.3
k_c1 = 1

# Pesos do Índice Goodhart
c1, c2, c3 = 0.33, 0.33, 0.34

# --- DEFINIÇÃO DOS MODELOS ---
def dc_motor_model(x1_m, u):
    dx1_m = -a*k*x1_m + k*u
    return dx1_m

def connected_systems_model_pid(states, t, tau_ref_val, taup_ref_val, Kp, Ki, Kd):
    x1_m, integral_state = states
    
    # PID Implementation
    error = tau_ref_val - x1_m
    
    # Nota: Seu modelo original usa apenas PI aqui (Kd não está sendo usado na conta abaixo)
    dc_volts = (Kp * error) + (Ki * integral_state)
    
    # Compute motor torque derivative
    taup = dc_motor_model(x1_m, dc_volts)
    
    # Retorna [derivada do estado, derivada da integral (que é o erro)]
    return np.array([taup, error])

# --- FUNÇÃO DE SIMULAÇÃO (OTIMIZADA) ---
def run_simulation(params):
    Kp, Ki, Kd = params
    
    # Penalidade para ganhos negativos
    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e9 

    # Inicialização rápida usando numpy
    states0 = np.array([0.0, 0.0]) # [x1_m, integral_state]
    states = np.zeros((n-1, 2))
    
    # --- OTIMIZAÇÃO AQUI ---
    # Removemos o odeint de dentro do loop. 
    # Usamos o método de Euler: Novo = Velho + (Derivada * Passo_de_Tempo)
    
    # Pré-carregar vetor para acesso rápido
    t_ref_local = torque_ref
    tp_ref_local = torquep_ref
    
    # Loop de integração manual
    for i in range(n-1):
        # 1. Calcular as derivadas (dx/dt)
        # O argumento 't' é 0 pois o sistema não depende explicitamente de t na função,
        # mas sim das referências que passamos
        derivatives = connected_systems_model_pid(
            states0, 0, 
            t_ref_local[i], tp_ref_local[i], 
            Kp, Ki, Kd
        )
        
        # 2. Atualizar estados (Integração de Euler)
        # x[k+1] = x[k] + dx/dt * dt
        states0 += derivatives * t_sim_step
        
        # 3. Salvar no histórico
        states[i] = states0

    # --- CÁLCULO DOS ÍNDICES ---
    y_real = states[:, 0]
    # Ajuste de slicing para bater com o tamanho de y_real (n-1)
    y_ref = torque_ref[:len(y_real)] 
    error_vector = y_ref - y_real
    
    integral_state = states[:, 1]
    control_signal = (Kp * error_vector) + (Ki * integral_state)
    
    mse = np.mean(error_vector**2)
    
    eps1 = np.mean(control_signal)
    eps2 = np.mean((control_signal - eps1)**2)
    eps3 = mse
    
    J = (c1 * eps1) + (c2 * eps2) + (c3 * eps3)
    
    if np.isnan(J) or np.isinf(J):
        return 1e9
        
    return J

# --- ALGORITMO DE OTIMIZAÇÃO: POLIEDROS FLEXÍVEIS (NELDER-MEAD) ---
def flexible_polyhedron_optimize(func, x0, tol=1e-4, max_iter=5):
    print("\n--- INICIANDO OTIMIZAÇÃO (POLIEDROS FLEXÍVEIS) ---")
    
    alpha = 1.0; gamma = 2.0; beta = 0.5; delta = 0.5
    N_dim = len(x0)
    
    simplex = [np.array(x0)]
    perturbation = 0.5
    
    for i in range(N_dim):
        point = np.array(x0)
        if point[i] == 0: point[i] = 0.1
        else: point[i] = point[i] * (1 + perturbation)
        simplex.append(point)
    
    costs = [func(v) for v in simplex]
    history_best_cost = []
    
    for k in range(max_iter):
        indices = np.argsort(costs)
        simplex = [simplex[i] for i in indices]
        costs = [costs[i] for i in indices]
        
        w_L = simplex[0]; J_L = costs[0]
        w_H = simplex[-1]; J_H = costs[-1]
        w_SH = simplex[-2]; J_SH = costs[-2]

        history_best_cost.append(J_L)
        
        P = np.sum([np.linalg.norm(w - w_L) for w in simplex])
        print(f"Iter {k}: Melhor J = {J_L:.6f} | Params: {w_L}")
        
        if P < tol:
            print("Critério de parada atingido.")
            break
            
        c = np.mean(simplex[:-1], axis=0)
        
        # Reflexão
        w_R = c + alpha * (c - w_H)
        J_R = func(w_R)
        
        if J_R < J_L:
            # Expansão
            w_E = c + gamma * (w_R - c)
            J_E = func(w_E)
            if J_E < J_R:
                simplex[-1] = w_E; costs[-1] = J_E
            else:
                simplex[-1] = w_R; costs[-1] = J_R
                
        elif J_R >= J_SH:
            # Contração
            should_shrink = False
            if J_R < J_H: # Externa
                w_C = c + beta * (w_R - c)
                J_C = func(w_C)
                if J_C <= J_R:
                    simplex[-1] = w_C; costs[-1] = J_C
                else: should_shrink = True
            else: # Interna
                w_C = c - beta * (w_R - c)
                J_C = func(w_C)
                if J_C < J_H:
                    simplex[-1] = w_C; costs[-1] = J_C
                else: should_shrink = True
            
            if should_shrink: # Redução
                for i in range(1, len(simplex)):
                    simplex[i] = w_L + delta * (simplex[i] - w_L)
                    costs[i] = func(simplex[i])
        else:
            simplex[-1] = w_R; costs[-1] = J_R

    return simplex[0], history_best_cost

# --- EXECUÇÃO PRINCIPAL ---

initial_guess = [18.0, 40.0, 10.0]
print(f"Parâmetros Iniciais: {initial_guess}")
# Teste rápido de custo inicial
print(f"Custo Inicial: {run_simulation(initial_guess):.6f}")

start_time = time.time()
best_params, cost_history = flexible_polyhedron_optimize(run_simulation, initial_guess, max_iter=20)
end_time = time.time()

print("\n" + "="*50)
print("RESULTADO DA OTIMIZAÇÃO")
print("="*50)
print(f"Tempo de Execução: {end_time - start_time:.2f} s")
print(f"Melhores Parâmetros: Kp={best_params[0]:.4f}, Ki={best_params[1]:.4f}, Kd={best_params[2]:.4f}")
print(f"Custo Final: {cost_history[-1]:.6f}")
print("="*50)

# Simulação final para plotagem
Kp_opt, Ki_opt, Kd_opt = best_params
states0 = np.array([0.0, 0.0])
states_opt = np.zeros((n-1, 2))

# Loop de simulação final (Mesma lógica Euler)
for i in range(n-1):
    derivs = connected_systems_model_pid(states0, 0, torque_ref[i], torquep_ref[i], Kp_opt, Ki_opt, Kd_opt)
    states0 += derivs * t_sim_step
    states_opt[i] = states0

# Plotagem
plt.rcParams['axes.grid'] = True
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(time_vector[:-1], torque_ref[:-1], 'k--', linewidth=2, label='Referência')
ax1.plot(time_vector[:-1], states_opt[:, 0], 'r', linewidth=2, label='Torque Otimizado')
ax1.set_ylabel('Torque [Nm]')
ax1.legend()

ax2.plot(cost_history, 'b-o', linewidth=2)
ax2.set_ylabel('Custo (J)')
ax2.set_xlabel('Iteração')

plt.tight_layout()
plt.show()
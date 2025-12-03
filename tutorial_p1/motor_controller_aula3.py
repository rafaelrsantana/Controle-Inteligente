"""
    This code has been developed by Juan Sandubete Lopez and all the rights
    belongs to him.
    Distribution or commercial use of the code is not allowed without previous
    agreement with the author.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import pandas as pd

# --- CONFIGURAÇÕES GERAIS ---
tf = 10.0
ts_ms = 0.01
n = int((1 / (ts_ms / 1000.0))*tf + 1)
time_vector = np.linspace(0, tf, n)
t_sim_step = time_vector[1] - time_vector[0]

# Referências (pré-calculadas para otimizar o loop)
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
    
    
    dc_volts = (Kp * error) + (Ki * integral_state)
    
    # Compute motor torque derivative
    taup = dc_motor_model(x1_m, dc_volts)
    
    return [taup, error]

# --- FUNÇÃO DE SIMULAÇÃO (A FUNÇÃO CUSTO J) ---
def run_simulation(params):
    """
    Executa a simulação com os parâmetros PID fornecidos.
    Retorna o Custo (J) para o otimizador.
    """
    Kp, Ki, Kd = params
    
    # Penalidade para ganhos negativos (Restrição)
    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e9 # Custo altíssimo para descartar

    states0 = [0, 0]
    states = np.zeros((n-1, len(states0)))
    
    # Loop de integração
    # Para otimizar velocidade, não vamos imprimir progresso aqui
    for i in range(n-1):
        # Passamos Kp, Ki, Kd via args
        out_states = odeint(connected_systems_model_pid, states0, [0.0, tf/n],
                            args=(torque_ref[i], torquep_ref[i], Kp, Ki, Kd))
        states0 = out_states[-1,:]
        states[i] = out_states[-1,:]

    # --- CÁLCULO DOS ÍNDICES ---
    N = len(states[:, 0])
    y_real = states[:, 0]
    y_ref = torque_ref[:N]
    error_vector = y_ref - y_real
    
    # Reconstrução do Sinal de Controle para Goodhart
    integral_state = states[:, 1]
    control_signal = (Kp * error_vector) + (Ki * integral_state)
    
    # MSE
    mse = np.mean(error_vector**2)
    
    # Goodhart
    eps1 = np.mean(control_signal)
    eps2 = np.mean((control_signal - eps1)**2)
    eps3 = mse
    
    J = (c1 * eps1) + (c2 * eps2) + (c3 * eps3)
    
    # Retornamos J. Se der NaN (instabilidade), retorna alto.
    if np.isnan(J) or np.isinf(J):
        return 1e9
        
    return J

# --- ALGORITMO DE OTIMIZAÇÃO: POLIEDROS FLEXÍVEIS (NELDER-MEAD) ---
def flexible_polyhedron_optimize(func, x0, tol=1e-4, max_iter=5):
    """
    Implementação do Método dos Poliedros Flexíveis (Seção 7.4.2)
    """
    print("\n--- INICIANDO OTIMIZAÇÃO (POLIEDROS FLEXÍVEIS) ---")
    
    # Parâmetros do método (Passo 0)
    alpha = 1.0  # Reflexão
    gamma = 2.0  # Expansão
    beta = 0.5   # Contração
    delta = 0.5  # Redução
    
    N_dim = len(x0) # Dimensão (3 para PID)
    
    # Passo 0: Escolher n+1 pontos (Simplex inicial)
    # Criamos vértices perturbando o chute inicial x0
    simplex = [np.array(x0)]
    perturbation = 0.5 # 50% de perturbação inicial
    
    for i in range(N_dim):
        point = np.array(x0)
        if point[i] == 0:
            point[i] = 0.1
        else:
            point[i] = point[i] * (1 + perturbation)
        simplex.append(point)
    
    # Avaliar função custo para cada vértice
    costs = [func(v) for v in simplex]
    
    history_best_cost = []
    
    for k in range(max_iter):
        # Passo 1: Ordenar vértices pelo custo
        # Indices ordenados: indices[0] é o melhor (L), indices[-1] é o pior (H)
        indices = np.argsort(costs)
        simplex = [simplex[i] for i in indices]
        costs = [costs[i] for i in indices]
        
        w_L = simplex[0]   # Melhor vértice
        J_L = costs[0]
        w_H = simplex[-1]  # Pior vértice
        J_H = costs[-1]
        w_SH = simplex[-2] # Segundo pior (necessário para lógica padrão)
        J_SH = costs[-2]

        history_best_cost.append(J_L)
        
        # Passo 3: Critério de Parada (Baseado no tamanho do simplex ou variância)
        # Usando a fórmula do PDF: P = sum(|w_i - w_L|^2)
        P = np.sum([np.linalg.norm(w - w_L) for w in simplex])
        
        print(f"Iter {k}: Melhor J = {J_L:.6f} | Params: {w_L}")
        
        if P < tol:
            print("Critério de parada atingido (Tolerância).")
            break
            
        # Passo 4: Centróide da face oposta a w_H
        # Média de todos exceto o pior
        c = np.mean(simplex[:-1], axis=0)
        
        # Passo 5: Reflexão
        w_R = c + alpha * (c - w_H)
        J_R = func(w_R)
        
        # Passo 6: Expansão
        if J_R < J_L:
            w_E = c + gamma * (w_R - c)
            J_E = func(w_E)
            
            if J_E < J_R:
                simplex[-1] = w_E
                costs[-1] = J_E
                action = "Expansão"
            else:
                simplex[-1] = w_R
                costs[-1] = J_R
                action = "Reflexão (após tentar expansão)"
                
        # Passo 8 (Lógica padrão): Contração
        elif J_R >= J_SH: # Pior que o segundo pior
            # Contração Externa ou Interna?
            # O PDF simplifica dizendo "Se J_R >= J_H então Redução", mas geralmente tenta contração antes.
            # Vamos seguir a lógica robusta que se encaixa no PDF:
            
            should_shrink = False
            
            if J_R < J_H:
                # Contração Externa (o ponto refletido é melhor que o pior, mas ainda ruim)
                w_C = c + beta * (w_R - c)
                J_C = func(w_C)
                if J_C <= J_R:
                    simplex[-1] = w_C
                    costs[-1] = J_C
                    action = "Contração Externa"
                else:
                    should_shrink = True
            else:
                # Contração Interna (o ponto refletido é pior que o pior original)
                w_C = c - beta * (w_R - c) # Ou c + beta*(w_H - c)
                J_C = func(w_C)
                if J_C < J_H:
                    simplex[-1] = w_C
                    costs[-1] = J_C
                    action = "Contração Interna"
                else:
                    should_shrink = True
            
            # Passo 7: Redução (Shrink) se contração falhar
            if should_shrink:
                action = "Redução (Shrink)"
                for i in range(1, len(simplex)):
                    simplex[i] = w_L + delta * (simplex[i] - w_L)
                    costs[i] = func(simplex[i])
                    
        else:
            # Aceita Reflexão simples (J_L <= J_R < J_SH)
            simplex[-1] = w_R
            costs[-1] = J_R
            action = "Reflexão"
            
        # print(f"   Ação: {action}")

    return simplex[0], history_best_cost

# --- EXECUÇÃO PRINCIPAL ---

# 1. Chute Inicial [Kp, Ki, Kd]
initial_guess = [18.0, 40.0, 10.0] # Valores do seu código original
print(f"Parâmetros Iniciais: {initial_guess}")
print(f"Custo Inicial: {run_simulation(initial_guess):.6f}")

# 2. Rodar Otimização
start_time = time.time()
best_params, cost_history = flexible_polyhedron_optimize(run_simulation, initial_guess, max_iter=20)
end_time = time.time()

print("\n" + "="*50)
print("RESULTADO DA OTIMIZAÇÃO")
print("="*50)
print(f"Tempo de Execução: {end_time - start_time:.2f} s")
print(f"Melhores Parâmetros Encontrados:")
print(f"Kp = {best_params[0]:.4f}")
print(f"Ki = {best_params[1]:.4f}")
print(f"Kd = {best_params[2]:.4f}")
print(f"Custo Final (Goodhart): {cost_history[-1]:.6f}")
print("="*50)

# 3. Simular Novamente com os Melhores Parâmetros para Plotar
print("\nSimulando sistema com parâmetros otimizados...")
Kp_opt, Ki_opt, Kd_opt = best_params

states0 = [0, 0]
states_opt = np.zeros((n-1, len(states0)))

for i in range(n-1):
    out_states = odeint(connected_systems_model_pid, states0, [0.0, tf/n],
                        args=(torque_ref[i], torquep_ref[i], Kp_opt, Ki_opt, Kd_opt))
    states0 = out_states[-1,:]
    states_opt[i] = out_states[-1,:]

# 4. Plotagem
plt.rcParams['axes.grid'] = True
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Gráfico 1: Resposta no Tempo
ax1.plot(time_vector[:-1], torque_ref[:-1], 'k--', linewidth=2, label='Referência')
ax1.plot(time_vector[:-1], states_opt[:, 0], 'r', linewidth=2, label='Torque Otimizado')
ax1.set_ylabel('Torque [Nm]')
ax1.set_title(f'Resposta Otimizada (Kp={Kp_opt:.1f}, Ki={Ki_opt:.1f}, Kd={Kd_opt:.1f})')
ax1.legend()

# Gráfico 2: Convergência do Custo
ax2.plot(cost_history, 'b-o', linewidth=2)
ax2.set_ylabel('Custo (Goodhart Index)')
ax2.set_xlabel('Iteração')
ax2.set_title('Convergência da Otimização')

plt.tight_layout()
plt.show()

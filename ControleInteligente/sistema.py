"""
    Arquivo: sistema.py
    Descrição: Contém a definição do sistema físico (Motor + PID), 
    parâmetros de simulação e a função custo (J).
    Atua como o orquestrador principal para rodar as otimizações.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import poliedro
import genetico
import pso
# --- IMPORTAÇÃO DOS ALGORITMOS (Crie estes arquivos na próxima etapa) ---
# Descomente as linhas abaixo quando tiver criado os arquivos poliedro.py e genetico.py
# import poliedro
# import genetico

# --- CONFIGURAÇÕES GERAIS E GLOBAIS ---
tf = 10.0
ts_ms = 0.1
n = int((1 / (ts_ms / 1000.0))*tf + 1)
time_vector = np.linspace(0, tf, n)
t_sim_step = time_vector[1] - time_vector[0]

# Referências (Arrays Globais)
torque_ref = np.sin(time_vector)
torquep_ref = np.cos(time_vector)

# Parâmetros do Modelo (Fixos)
a = 1
k = 1
# Pesos do Índice Goodhart
c1, c2, c3 = 0.04, 0.01, 0.90

# --- MODELOS FÍSICOS ---
def dc_motor_model(x1_m, u):
    dx1_m = -a*k*x1_m + k*u
    return dx1_m

def connected_systems_model_pid(states, t, tau_ref_val, taup_ref_val, Kp, Ki, Kd):
    x1_m, integral_state = states
    
    # 1. Cálculo do Erro
    error = tau_ref_val - x1_m
    
    # 2. Parte PI (Proporcional + Integral)
    u_pi = (Kp * error) + (Ki * integral_state)
    
    # 3. Parte Derivativa (Solução do Loop Algébrico)
    # A equação original é: dx = -a*k*x + k*(u_pi + Kd*(d_ref - dx))
    # Isolando dx, temos:
    numerator = -a*k*x1_m + k*u_pi + k*Kd*taup_ref_val
    denominator = 1 + k*Kd
    
    taup = numerator / denominator
    
    # Retorna [derivada_estado, derivada_integral (erro)]
    return np.array([taup, error])

# --- FUNÇÃO CUSTO (OTIMIZADA COM EULER) ---
def run_simulation(params):
    """
    Esta é a função que os algoritmos de otimização vão chamar.
    Recebe: [Kp, Ki, Kd]
    Retorna: Custo J (float)
    """
    Kp, Ki, Kd = params
    
    # Penalidade para ganhos negativos
    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e9 
    
    if Kp > 100 or Ki > 100 or Kd > 100:
        return 1e9  # Evita saturação do atuador
    
    # Inicialização rápida
    states0 = np.array([0.0, 0.0]) # [x1_m, integral_state]
    states = np.zeros((n-1, 2))
    
    t_ref_local = torque_ref
    tp_ref_local = torquep_ref
    
    # Loop de Integração de Euler (Rápido)
    for i in range(n-1):
        derivatives = connected_systems_model_pid(
            states0, 0, 
            t_ref_local[i], tp_ref_local[i], 
            Kp, Ki, Kd
        )
        states0 += derivatives * t_sim_step
        states[i] = states0

    # --- CÁLCULO DOS ÍNDICES (CORRIGIDO) ---
    y_real = states[:, 0]
    y_ref = torque_ref[:len(y_real)]
    error_vector = y_ref - y_real
    
    integral_state = states[:, 1]
    
    # [CORREÇÃO] Calcular a derivada do erro para incluir o custo do Kd
    # np.gradient calcula a variação ponto a ponto
    derivative_error = np.gradient(error_vector, t_sim_step)
    
    # Agora o sinal de controle inclui o esforço do Kd!
    control_signal = (Kp * error_vector) + (Ki * integral_state) + (Kd * derivative_error)
    
    # MSE
    mse = np.mean(error_vector**2)
    
    # Goodhart (Agora vai penalizar o Kd alto pois control_signal será alto)
    eps1 = np.mean(np.abs(control_signal)) # Usei abs para pegar magnitude média
    eps2 = np.mean((control_signal - np.mean(control_signal))**2) # Variância
    eps3 = mse
    
    J = (c1 * eps1) + (c2 * eps2) + (c3 * eps3)
    
    if np.isnan(J) or np.isinf(J):
        return 1e9
        
    return J

# --- FUNÇÃO AUXILIAR PARA PLOTAGEM ---
def simular_e_retornar_dados(params):
    """Roda a simulação uma vez e retorna os dados para o gráfico"""
    Kp, Ki, Kd = params
    states0 = np.array([0.0, 0.0])
    states_hist = np.zeros((n-1, 2))
    
    for i in range(n-1):
        derivatives = connected_systems_model_pid(states0, 0, torque_ref[i], torquep_ref[i], Kp, Ki, Kd)
        states0 += derivatives * t_sim_step
        states_hist[i] = states0
    return states_hist[:, 0] # Retorna apenas o torque

# --- EXECUÇÃO PRINCIPAL (MAIN) ---
if __name__ == "__main__":
    print("--- INICIANDO SISTEMA DE CONTROLE ---")
    
    # Chute inicial padrão
    initial_guess = [18.0, 40.0, 10.0]
    custo_inicial = run_simulation(initial_guess)
    print(f"Custo com parâmetros iniciais {initial_guess}: {custo_inicial:.6f}")

    # Configuração dos Plots
    plt.rcParams['axes.grid'] = True
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot da Referência (Sempre igual)
    ax1.plot(time_vector[:-1], torque_ref[:-1], 'k--', linewidth=2, label='Referência')

    # ---------------------------------------------------------
    # ÁREA PARA RODAR OS ALGORITMOS
    # (Descomente abaixo quando os arquivos existirem)
    # ---------------------------------------------------------

    # --- 1. RODAR NELDER-MEAD (POLIEDRO) ---
    print("\n>>> Rodando Poliedros Flexíveis...")
    start_nm = time.time()
    params_nm, hist_nm = poliedro.otimizar(run_simulation, initial_guess)
    end_nm = time.time()
    print(f"Tempo NM: {end_nm - start_nm:.2f}s | Melhor J: {hist_nm[-1]:.6f}")
    
    #Plotar resultado NM
    dados_nm = simular_e_retornar_dados(params_nm)
    ax1.plot(time_vector[:-1], dados_nm, 'r', label=f'Nelder-Mead (J={hist_nm[-1]:.4f})')
    ax2.plot(hist_nm, 'r-o', label='Nelder-Mead')

    # --- 2. RODAR ALGORITMO GENÉTICO ---
    print("\n>>> Rodando Algoritmo Genético...")
    limites_ga = [(0, 100), (0, 100), (0, 100)] # Limites para Kp, Ki, Kd
    start_ga = time.time()
    params_ga, hist_ga = genetico.otimizar(run_simulation, limites_ga, n_pop=30, n_gen=50)
    end_ga = time.time()
    print(f"Tempo GA: {end_ga - start_ga:.2f}s | Melhor J: {hist_ga[-1]:.6f}")

    # Plotar resultado GA
    dados_ga = simular_e_retornar_dados(params_ga)
    ax1.plot(time_vector[:-1], dados_ga, 'g--', label=f'Genético (J={hist_ga[-1]:.4f})')
    ax2.plot(hist_ga, 'g-x', label='Genético')

    # --- 3. RODAR PSO (ENXAME DE PARTÍCULAS) ---
    print("\n>>> Rodando PSO...")
    limites_pso = [(0, 100), (0, 100), (0, 100)]
    start_pso = time.time()
    
    # Chama o otimizador
    params_pso, hist_pso = pso.otimizar(run_simulation, limites_pso, n_particulas=30, n_iter=50)
    
    end_pso = time.time()
    print(f"Tempo PSO: {end_pso - start_pso:.2f}s | Melhor J: {hist_pso[-1]:.6f}")

    # Plotar resultado PSO
    dados_pso = simular_e_retornar_dados(params_pso)
    ax1.plot(time_vector[:-1], dados_pso, 'b-.', label=f'PSO (J={hist_pso[-1]:.4f})') # Cor azul pontilhada
    ax2.plot(hist_pso, 'b-.', label='PSO')

    # ---------------------------------------------------------

    ax1.set_ylabel('Torque [Nm]')
    ax1.set_title('Resposta no Tempo: Comparativo')
    ax1.legend()

    ax2.set_ylabel('Custo J')
    ax2.set_xlabel('Iterações / Gerações')
    ax2.set_title('Convergência dos Algoritmos')
    ax2.legend()

    print("\n" + "="*50)
    print("🏆 RELATÓRIO FINAL: MELHORES PARÂMETROS 🏆")
    print("="*50)

    # 1. Resultado Nelder-Mead
    # (Certifique-se que você descomentou e rodou o código do NM para essa variável existir)
    if 'params_nm' in locals():
        print(f"🔷 NELDER-MEAD:")
        print(f"   Kp = {params_nm[0]:.4f}")
        print(f"   Ki = {params_nm[1]:.4f}")
        print(f"   Kd = {params_nm[2]:.4f}")
        print(f"   Custo Final (J) = {hist_nm[-1]:.6f}")
        print("-" * 30)

    # 2. Resultado Genético
    if 'params_ga' in locals():
        print(f"🟢 GENÉTICO:")
        print(f"   Kp = {params_ga[0]:.4f}")
        print(f"   Ki = {params_ga[1]:.4f}")
        print(f"   Kd = {params_ga[2]:.4f}")
        print(f"   Custo Final (J) = {hist_ga[-1]:.6f}")
        print("-" * 30)

    # 3. Resultado PSO
    if 'params_pso' in locals():
        print(f"🔵 PSO:")
        print(f"   Kp = {params_pso[0]:.4f}")
        print(f"   Ki = {params_pso[1]:.4f}")
        print(f"   Kd = {params_pso[2]:.4f}")
        print(f"   Custo Final (J) = {hist_pso[-1]:.6f}")
        print("="*50)

    plt.tight_layout()
    plt.show()
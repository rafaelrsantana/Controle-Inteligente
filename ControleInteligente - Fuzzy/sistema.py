import numpy as np
import matplotlib.pyplot as plt
import time
import controlador_fuzzy

tf = 10.0
ts_ms = 0.01
n = int((1 / (ts_ms / 1000.0))*tf + 1)
time_vector = np.linspace(0, tf, n)
t_sim_step = time_vector[1] - time_vector[0]

torque_ref = np.sin(time_vector)

a = 1
k = 1

def dc_motor_model(x1_m, u):
    dx1_m = -a*k*x1_m + k*u
    return dx1_m

    
print("\n--- COMEÇANDO A SIMULAÇÂO FUZZY ---\n")

states0 = [0.0]
states = np.zeros((n-1, 2))

meu_fuzzy = controlador_fuzzy.FuzzyPD()

erro_anterior = 0.0
start_time = time.time()

for i in range(n-1):

    torque_real = states0[0]
    ref_atual = torque_ref[i]

    erro_atual = torque_ref[i] - states0[0]

    if i == 0:
        d_erro = 0.0
    else:
        d_erro = (erro_atual - erro_anterior) / t_sim_step
        
    u_fuzzy = meu_fuzzy.calcular(erro_atual, d_erro)

    derivada_motor = dc_motor_model(torque_real, u_fuzzy)

    states0[0] += derivada_motor* t_sim_step

    states[i, 0] = states0[0]
    states[i, 1] = u_fuzzy

    erro_anterior = erro_atual
    if i % int((n-1) / 10) == 0:
        progresso = int((i / (n-1)) * 100)
        print(f"Progresso: {progresso}% concluído...")

end_time = time.time()
print(f"Simulação concluída em {end_time - start_time:.4f} segundos.")

y_real = states[:, 0]
y_ref = torque_ref[:-1]
erro = y_ref - y_real
mse = np.mean(erro**2)
mae = np.mean(np.abs(erro))

print(f"\nRESULTADOS DO FUZZY:")
print(f"MSE (Erro Quadrático):  {mse:.6f}")
print(f"MAE (Erro Absoluto):    {mae:.6f}")

plt.rcParams['axes.grid'] = True
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (10,10), sharex = True)

ax1.plot(time_vector[:-1], y_ref, 'k--', label = 'Referência', linewidth = 2)
ax1.plot(time_vector[:-1], y_real, 'b', label = 'Fuzzy PD', linewidth = 2)
ax1.set_ylabel('Torque [Nm]')
ax1.set_title(f'Controle Fuzzy - Rastreamento (MSE = {mse: .5f})')
ax1.legend()

ax2.plot(time_vector[:-1], erro, 'r')
ax2.set_ylabel('Erro [Nm]')
ax2.set_title('Erro de Rastreamento')

ax3.plot(time_vector[:-1], states[:, 1], 'g')
ax3.set_ylabel('Tensão [V]')
ax3.set_xlabel('Tempo [s]')
ax3.set_title('Esforço de Controle (Saída do Fuzzy)')

plt.tight_layout()
plt.show()
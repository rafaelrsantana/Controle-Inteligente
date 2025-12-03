"""
    This code has been developed by Juan Sandubete Lopez and all the rights
    belongs to him.
    Distribution or commercial use of the code is not allowed without previous
    agreement with the author.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import time
import pandas as pd

# Simulation parametrs
tf = 10.0  # final time
ts_ms = 0.01  # 0.001 = 1us, 1 = 1ms
save_data = False  # Attention: CSV file can be very big
show_fig = True
save_fig = False  # If False, figure is showed but not saved
title = "motor_control_a_with_error_wtf2"

print("Starting motor simulation.")

# Models Parameters
# Motor
a = 1
k = 1
a_model_error = 0.1
k_model_error = 0.3
# Motor Controller (Control1)
k_c1 = 1

# PID
USE_PID = True
Kp = 18.0
Ki = 40.0
Kd = 10.0

print("\n--- PARAMETERS --- \n ")
print("Motor Parameters: a = {}, k = {}".format(a, k))
print("Motor Induced Model Errors: a_error = {}, k_error = {}".
      format(a_model_error, k_model_error))
print("Motor Controller: kc1 = {}".format(k_c1))
if USE_PID:
    print("PID Controller Active: Kp={}, Ki={}, Kd={}".format(Kp, Ki, Kd))

# Define models
def dc_motor_model(x1_m, u):
    # DC motor model:
    # taup + a*k*tau = k*u
    # With the change: x1_m = tau (x1_motor)
    dx1_m = -a*k*x1_m + k*u
    y1_m = x1_m
    return dx1_m

def motor_controller(tau, tau_ref, taup_ref):
    # Non-Linear control for DC Motor following Dyn ecs: taup + a*k*tau = k*u
    # The controller returns dc_volts
    v = taup_ref - k_c1*(tau - tau_ref)
    return (a+a_model_error)*tau + v/(k+k_model_error)


# The following function puts all ecuations together
def connected_systems_model(states, t, tau_ref, taup_ref):
    # Input values. Check this with the out_states list
    x1_m, integral_state = states

    if USE_PID:
        # PID Implementation
        error = tau_ref - x1_m
        
        # Control Law: u = Kp*e + Ki*int_e
        dc_volts = (Kp * error) + (Ki * integral_state)
        
        # Compute motor torque derivative
        taup = dc_motor_model(x1_m, dc_volts)
        
        # Return derivatives: [d(tau)/dt, d(int_error)/dt]
        # The derivative of the integral of error is the error itself.
        return [taup, error]
    
    else:
        # Original Controller Implementation
        
        # Compute motor controller
        dc_volts = motor_controller(x1_m, tau_ref, taup_ref)
        # Compute motor torque
        taup = dc_motor_model(x1_m, dc_volts)

        # Output
        out_states = [taup, dc_volts]
        return out_states


# Initial conditions
states0 = [0, 0]
n = int((1 / (ts_ms / 1000.0))*tf + 1) # number of time points

# time span for the simulation, cycle every tf/n seconds
time_vector = np.linspace(0,tf,n)
t_sim_step = time_vector[1] - time_vector[0]

# Reference signal and its differentiations
torque_ref = np.sin(time_vector)
print("Max ref: {}".format(max(torque_ref)))
print("Min ref: {}".format(min(torque_ref)))
torquep_ref = np.cos(time_vector)
print("Max ref: {}".format(max(torquep_ref)))
print("Min ref: {}".format(min(torquep_ref)))
# Output arrays
states = np.zeros( (n-1, len(states0)) ) # States for each timestep

print("\n--- SIMULATION CONFIG. ---\n")
print("Simulation time: {} sec".format(tf))
print("Time granulatiry: {}".format(t_sim_step))
print("Initial states: {}".format(states0))

print("\n--- SIMULATION Begins ---\n")

initial_time = time.time()
# Simulate with ODEINT
t_counter = 0
for i in range(n-1):
    out_states = odeint(connected_systems_model, states0, [0.0, tf/n],
                        args=(torque_ref[i], torquep_ref[i]))
    states0 = out_states[-1,:]
    states[i] = out_states[-1,:]
    if i >= t_counter * int((n-1)/10):
        print("Simulation at {}%".format(t_counter*10))
        t_counter += 1

elapsed_time = time.time() - initial_time
print("\nElapsed time: {} sec.".format(elapsed_time))
print("\n--- SIMULATION Finished. ---\n")

# --- PERFORMANCE INDICES CALCULATION ---
print("\n--- CALCULATING PERFORMANCE INDICES ---\n")

# 1. Prepare Data
# We need to slice arrays to match the simulation steps (n-1)
N = len(states[:, 0])
y_real = states[:, 0]
y_ref = torque_ref[:N]
t_vec = time_vector[:N]
y_ref_dot = torquep_ref[:N] # Needed for original controller reconstruction

# Error Vector
error_vector = y_ref - y_real

# 2. Reconstruct Control Signal (SC_k)
# We need to recalculate 'u' (Volts) for each step to calculate Goodhart's Index
control_signal = np.zeros(N)

if USE_PID:
    integral_state = states[:, 1]
    # u = Kp*e + Ki*int_e
    # Note: We use the error vector calculated above
    control_signal = (Kp * error_vector) + (Ki * integral_state)
else:
    # Reconstruct Original Controller Signal
    for i in range(N):
        # motor_controller(tau, tau_ref, taup_ref)
        control_signal[i] = motor_controller(y_real[i], y_ref[i], y_ref_dot[i])

# 3. Calculate Indices using Discrete Formulas

# MAE (Mean Absolute Error): (1/N) * sum(|e|)
mae = np.mean(np.abs(error_vector))

# ITAE Discrete: (1/N) * sum(t * |e|)
itae = np.mean(t_vec * np.abs(error_vector))

# MSE (Mean Squared Error): (1/N) * sum(e^2)
mse = np.mean(error_vector**2)

# Goodhart's Index
# Weights (c1, c2, c3). Assuming 1.0 for all as standard if not specified.
c1, c2, c3 = 0.33, 0.33, 0.34

# epsilon 1: Mean of Control Signal (SC)
eps1 = np.mean(control_signal)

# epsilon 2: Variance of Control Signal (SC - mean)^2
eps2 = np.mean((control_signal - eps1)**2)

# epsilon 3: MSE (already calculated)
eps3 = mse

goodhart_index = (c1 * eps1) + (c2 * eps2) + (c3 * eps3)

# Store in a vector
performance_indices = [mae, itae, mse, goodhart_index]

# 4. Print Results
print("-" * 50)
print(f"PERFORMANCE METRICS ({'PID' if USE_PID else 'ORIGINAL'}):")
print("-" * 50)
print(f"MAE (Mean Abs Error)       : {mae:.6f}")
print(f"ITAE (Discrete)            : {itae:.6f}")
print(f"MSE (Mean Squared Error)   : {mse:.6f}")
print(f"Goodhart Index             : {goodhart_index:.6f}")
print("-" * 50)
print("Goodhart Components:")
print(f"  e1 (Mean Control Effort) : {eps1:.6f}")
print(f"  e2 (Control Variance)    : {eps2:.6f}")
print(f"  e3 (MSE)                 : {eps3:.6f}")
print("-" * 50)

# --- COMPARISON LOGIC (ADDED) ---
# Runs a secondary simulation for the "Original" controller to compare against the current PID run
if USE_PID:
    print("\n--- RUNNING COMPARISON SIMULATION (Original Controller) ---\n")
    
    # 1. Save current global state and switch to Original
    original_use_pid_flag = USE_PID
    USE_PID = False 
    
    # 2. Run temporary simulation for Original Controller
    states_comp = np.zeros((n-1, len(states0)))
    states0_comp = [0, 0] # Reset initial conditions
    
    for i in range(n-1):
        # Re-use the connected_systems_model which now sees USE_PID = False
        out_states = odeint(connected_systems_model, states0_comp, [0.0, tf/n],
                            args=(torque_ref[i], torquep_ref[i]))
        states0_comp = out_states[-1,:]
        states_comp[i] = out_states[-1,:]

    # 3. Calculate Indices for Original
    y_real_comp = states_comp[:, 0]
    error_vector_comp = y_ref - y_real_comp
    
    # Reconstruct Control Signal for Original
    control_signal_comp = np.zeros(N)
    for i in range(N):
        control_signal_comp[i] = motor_controller(y_real_comp[i], y_ref[i], y_ref_dot[i])
        
    mae_orig = np.mean(np.abs(error_vector_comp))
    itae_orig = np.mean(t_vec * np.abs(error_vector_comp))
    mse_orig = np.mean(error_vector_comp**2)
    
    eps1_orig = np.mean(control_signal_comp)
    eps2_orig = np.mean((control_signal_comp - eps1_orig)**2)
    eps3_orig = mse_orig
    goodhart_orig = (c1 * eps1_orig) + (c2 * eps2_orig) + (c3 * eps3_orig)
    
    # 4. Restore global state
    USE_PID = original_use_pid_flag
    
    # 5. Print Comparison Table
    print("\n" + "="*80)
    print(f"{'METRIC':<15} | {'ORIGINAL':<15} | {'PID (Current)':<15} | {'IMPROVEMENT':<15}")
    print("="*80)
    
    def print_row(name, val_orig, val_pid):
        if val_orig != 0:
            # Positive % means PID error is smaller (Better)
            imp = ((val_orig - val_pid) / val_orig) * 100
        else:
            imp = 0.0
        print(f"{name:<15} | {val_orig:.6f}        | {val_pid:.6f}        | {imp:+.2f}%")

    print_row("MAE", mae_orig, mae)
    print_row("ITAE", itae_orig, itae)
    print_row("MSE", mse_orig, mse)
    print_row("Goodhart", goodhart_orig, goodhart_index)
    print("="*80)
    print("* Positive improvement % means the PID value is lower (better error/cost).")

if save_data:
    print("Saving simulation data...")
    sim_df = pd.DataFrame(states)
    sim_df = sim_df.transpose()
    sim_df.rename({0: 'tau', 1: 'tau_ref', 2: 'taup_ref',
                   3: 'dc_volts'}, inplace=True)
    sim_df.to_csv('sim_data/ex4_motor_control.csv')

# Plot results
# States are: tau, tau_ref, taup_ref, dc_volts
plt.rcParams['axes.grid'] = True
plt.figure()
plt.subplot(2,1,1)
plt.plot(time_vector[:-1],torque_ref[:-1],'k--',linewidth=3, label='Reference Torque')
plt.plot(time_vector[:-1],states[:,0],'r',linewidth=2, label='Actual Torque')
plt.ylabel('tau [Nm]')
plt.legend(loc='upper right')
plt.title('System Response')

plt.subplot(2,1,2)
plt.plot(time_vector[:-1],torquep_ref[:-1],'k--',linewidth=3, label='Ref. Derivative')

if USE_PID:
    # If PID, the second state is the Integral of Error
    plt.plot(time_vector[:-1], states[:,1], 'b', linewidth=2, label='Integral Error')
    plt.ylabel('Int. Error')
else:
    # If original controller, the second state is Voltage
    plt.plot(time_vector[:-1], states[:,1], 'g', linewidth=2, label='Control Voltage')
    plt.ylabel('Voltage [V]')

plt.legend(loc='upper right')

if save_fig:
    figname = "pictures/" + title + ".png"
    plt.savefig(figname)
if show_fig:
    plt.show()

import numpy as np

# Parâmetros do Motor (Default)
A_DEFAULT = 1
K_DEFAULT = 1
A_ERROR = 0.1
K_ERROR = 0.3


def dc_motor_model(x1_m, u, a=A_DEFAULT, k=K_DEFAULT):
    dx1_m = -a * k * x1_m + k * u
    return dx1_m


def original_control_law(
    tau, tau_ref, taup_ref, k_c1, a=A_DEFAULT, k=K_DEFAULT, a_err=A_ERROR, k_err=K_ERROR
):
    v = taup_ref - k_c1 * (tau - tau_ref)
    return (a + a_err) * tau + v / (k + k_err)


def system_model_original(
    states,
    t,
    tau_ref,
    taup_ref,
    k_c1,
    a=A_DEFAULT,
    k=K_DEFAULT,
    a_err=A_ERROR,
    k_err=K_ERROR,
):
    x1_m, _ = states
    u = original_control_law(x1_m, tau_ref, taup_ref, k_c1, a, k, a_err, k_err)
    taup = dc_motor_model(x1_m, u, a, k)
    return [taup, u]


def system_model_pid(states, t, u_control, a=A_DEFAULT, k=K_DEFAULT):
    x1_m, _ = states
    taup = dc_motor_model(x1_m, u_control, a, k)
    return [taup, u_control]

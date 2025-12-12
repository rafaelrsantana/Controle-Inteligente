import numpy as np


def calculate_indices(ref, output, time_vec, dt):
    error = ref - output
    MAE = np.mean(np.abs(error))
    MSE = np.mean(error**2)
    ITAE = np.sum(time_vec * np.abs(error) * dt)
    c1, c2, c3 = 0.33, 0.33, 0.34
    Goodhart = c1 * MAE + c2 * MSE + c3 * ITAE
    return MAE, MSE, ITAE, Goodhart

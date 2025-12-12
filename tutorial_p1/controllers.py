import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, limit=100.0, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def update(self, error):
        if self.first_run:
            derr = 0.0
            self.first_run = False
        else:
            derr = (error - self.prev_error) / self.dt

        # Proportional
        up = self.kp * error

        # Derivative
        ud = self.kd * derr

        # Integral (Tentative)
        # We calculate what the integral term WOULD be if we integrated normally
        ui_tentative = self.integral + error * self.dt * self.ki

        # Total unsatured output (assuming Ki is multiplied by integral sum, or integral term accumulates Ki*error*dt)
        # Standard PID: u = Kp*e + Ki*int(e) + Kd*de/dt
        # My implementation: self.integral stores sum(error * dt)
        # So term is self.ki * self.integral

        # Let's adjust to match standard implementation where we integrate the error
        # self.integral += error * dt
        # u = ... + self.ki * self.integral

        # Tentative integral sum
        integral_tentative = self.integral + error * self.dt

        u_unsat = up + self.ki * integral_tentative + ud

        # Anti-Windup Logic (Clamping / Conditional Integration)
        # If saturated, we only integrate if the error helps to desaturate

        u = u_unsat
        is_saturated = False

        if u_unsat > self.limit:
            u = self.limit
            is_saturated = True
        elif u_unsat < -self.limit:
            u = -self.limit
            is_saturated = True

        # Conditional Integration
        if is_saturated:
            # If saturated positive and error is negative (trying to reduce), allow integration
            if (u > 0 and error < 0) or (u < 0 and error > 0):
                self.integral = integral_tentative
            # Else: Do not update integral (Freeze)
        else:
            self.integral = integral_tentative

        self.prev_error = error
        return u

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

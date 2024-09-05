import model
import control
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

r_safe = 1
v_ref = 1
omega_ref = 1
T = 10 # simulation time
dt = 0.01 # minimum timestep
U_a = []
T_a = []
def closed_loop(t, x):
    u = control.ECBF_controller(x,r_safe, v_ref, omega_ref)
    U_a.append(u[0])
    T_a.append(t)
    x_dot = model.bicycle(t, x, u)
    return x_dot
x_0 = np.array([0.1, 0.1, 0.1, 0, 0])
res = solve_ivp(closed_loop,
          [0, T], 
          x_0, 
          t_eval=np.linspace(0, T, int(T/dt)),
          dense_output=True)

plt.figure()
plt.suptitle('Simulation')
plt.subplot(3,2,1)
plt.title('Velocity')
plt.plot(res.t, res.y[2], label='v')
plt.plot(res.t, [v_ref]*len(res.t), label='v_ref')
plt.grid()
plt.legend()
plt.subplot(3,2,2)
plt.title('angular velocity')
plt.plot(res.t, res.y[4], label='omega')
plt.plot(res.t, [omega_ref]*len(res.t), label='omega_ref')
plt.grid()
plt.subplot(3,2,3)
plt.title('acceleration')
plt.plot(T_a, U_a, label='a')
plt.grid()
plt.subplot(3,2,4)
plt.title('State-space')
plt.plot(res.y[0], res.y[1], label='trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.subplot(3,2,5)
plt.title('Theta')
plt.plot(res.t, res.y[3], label='theta')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.subplot(3,2,6)
plt.title('h')
h = r_safe**2 - (res.y[0]**2 + res.y[1]**2)
plt.plot(res.t, h, label='h')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.show()

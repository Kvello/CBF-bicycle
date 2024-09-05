import model
import control
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

r_safe = 1.4
v_ref = 1.0
T = 100 # simulation time
dt = 0.001 
U_a = []
U_omega = []
T_c = []
deltas = []
def closed_loop(t, x):
    u,delta = control.ECBF_controller(x,r_safe, v_ref)
    U_a.append(u[0])
    U_omega.append(u[1])
    T_c.append(t)
    deltas.append(delta)
    x_dot = model.bicycle(t, x, u)
    return x_dot
x_0 = np.array([-0.1, 0.1, 0.1, 0])
res = solve_ivp(closed_loop,
          [0, T], 
          x_0, 
          t_eval=np.linspace(0, T, int(T/dt)),
          method='RK45',
          rtol=1e-6,
          atol=1e-5)

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
plt.plot(T_c,U_omega, label='omega')
plt.grid()
plt.subplot(3,2,3)
plt.title('acceleration')
plt.plot(T_c, U_a, label='a')
plt.grid()
plt.subplot(3,2,4)
plt.title('State-space')
plt.plot(res.y[0], res.y[1], label='trajectory')
plt.plot(x_0[0], x_0[1], 'ro', label='initial position')
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

plt.figure()
plt.subplot(2,2,1)
plt.title('x')
plt.plot(res.t, res.y[0], label='x')
plt.grid()
plt.subplot(2,2,2)
plt.title('y')
plt.plot(res.t, res.y[1], label='y')
plt.grid()
plt.subplot(2,2,3)
plt.title('delta')
plt.plot(T_c, deltas, label='delta')
plt.grid()
plt.show()
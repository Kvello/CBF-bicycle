import model
import control
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

def closed_loop(t, x):
    u = control.ECBF_controller(x)
    x_dot = model.bicycle(t, x, u)
    return x_dot
x_0 = np.array([0, 0, 0, 0, 0])

res = solve_ivp(closed_loop,
          [0, 10], 
          x_0, 
          t_eval=np.linspace(0, 10, 100))
plt.figure()
plt.suptitle('Simulation')
plt.subplot(2,1,1)
plt.title('Trajectory')
plt.plot(res.t, res.y[0], label='x')
plt.plot(res.t, res.y[1], label='y')
plt.legend()
plt.subplot(2,1,2)
plt.title('State-space')
plt.plot(res.y[0], res.y[1], label='trajectory')
plt.legend()
plt.show()

import numpy as np
def bicycle(t,x:np.ndarray, u:np.ndarray) -> np.ndarray:
    """
    Model of a simple bicycle.
    Args:
    t: time(unused)
    x: state vector [x position, y position, abs velocity, heading, angular velocity]
    u: input vector [abs acceleration, steering torque]
    dt: timestep
    Returns:
    x_dot: derivative of the state vector
    """
    x_dot = np.zeros(5)
    x_dot[0] = x[2]*np.cos(x[2])
    x_dot[1] = x[2]*np.sin(x[2])
    x_dot[2] = u[0]
    x_dot[3] = x[4]
    x_dot[4] = u[1]
    return x_dot
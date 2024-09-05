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
    pos_x = x[0]
    pos_y = x[1]
    v = x[2] # velocity
    theta = x[3] # heading
    x_dot = x[2]*np.cos(x[3])
    y_dot = x[2]*np.sin(x[3])
    v_dot = u[0]
    theta_dot = u[1]
    return np.array([x_dot, y_dot, v_dot, theta_dot])
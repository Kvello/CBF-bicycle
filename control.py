import numpy as np

def ECBF_controller(x:np.ndarray)->np.ndarray:
    """
    Controller that uses a Exponential Control Barrier Function.
    Args:
    x: state vector [x position, y position, abs velocity, heading, angular velocity]
    Returns:
    u: input vector [abs acceleration, steering torque]
    """
    u = np.zeros(2)
    return u
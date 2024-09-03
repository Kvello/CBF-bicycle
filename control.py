import numpy as np
import numpy as np
import qpsolvers as qp

def CLF_controller(x:np.ndarray,
                   v_ref:float,
                     omega_ref:float)->np.ndarray:
    """
    Controller that uses a Control Lyapunov Function.
    Args:
    x: state vector [x position, y position, abs velocity, heading, angular velocity]
    Returns:
    u: input vector [abs acceleration, steering torque]
    """
    # States
    pos_x = x[0]
    pos_y = x[1]
    v = x[2] # velocity
    theta = x[3] # heading
    omega = x[4] # angular velocity

    gamma = 2 # CLF constraint parameter
    V = 0.5*(v-v_ref)**2 + 0.5*(omega-omega_ref)**2 # Lyapunov function
    G_clf = np.array([[v-v_ref, omega-omega_ref]]) # Gradient of Lyapunov function
    h_clf = np.array([-gamma*V])
    # QP problem
    P = np.eye(2) # Input cost matrix
    lb = np.array([0,-np.inf])
    sol = qp.solve_qp(P, np.zeros(2), G_clf, h_clf,lb=lb,solver="clarabel")
    if sol is None:
        print("QP solver failed at x: ",x)
        print("Using default control")
        return [-0.01*(v_ref-v), -0.01*(omega_ref-omega)]
    u = sol
    return u

    
    
def ECBF_controller(x:np.ndarray,
                    r_safe:float,
                    v_ref:float,
                    omega_ref:float)->np.ndarray:
    """
    Controller that uses a Exponential Control Barrier Function.
    Args:
    x: state vector [x position, y position, abs velocity, heading, angular velocity]
    K_alpha: gain constraint for the controller
    Returns:
    u: input vector [abs acceleration, steering torque]
    """
    # States
    pos_x = x[0]
    pos_y = x[1]
    v = x[2] # velocity
    theta = x[3] # heading
    omega = x[4] # angular velocity

    # Canonical controlable form of the ECBF system    
    F = np.matrix([[0,1],[0,0]])
    G = np.matrix([[0],[1]])
    C = np.matrix([1,0])
    eta_b = np.array([
        [r_safe - (pos_x**2 + pos_y**2)], # h(x) = r_safe -(x² + y²)
       [-2*v*(np.cos(theta)*pos_x + np.sin(theta)*pos_y)] # h_dot(x)
       ])

    # CLF constraint
    gamma = 10 # CLF constraint parameter
    V = 0.5*(v-v_ref)**2 # Lyapunov function
    G_clf = np.array([[v-v_ref, 0, 0, -1]]) # Gradient of Lyapunov function
    h_clf = -gamma*V

    # ECBF constraints
    poles = ECBF_get_poles(x,r_safe) # Desired poles
    # Pole placement
    k1 = poles[0]*poles[1]
    k2 = -(poles[0] + poles[1])
    K_alpha = np.array([[k1, k2]])
    # print("Closed loop system: ",F-G@K_alpha)
    # print("Closed-loop system eigenvalues: ",np.linalg.eigvals(F-G@K_alpha))
    A_ecbf = np.array([[2*np.cos(theta)*pos_x + 2*np.sin(theta)*pos_y,
                        2*v*(np.cos(theta)*pos_y-np.sin(theta)*pos_x + v),
                        -1, 
                        0]])
    b_ecbf = 0
    G_ecbf = np.array([[0, 0, -1, 0]])
    h_ecbf = (K_alpha@eta_b).item()

    # QP problem
    G = np.block([[G_clf],[G_ecbf]])
    h = np.array([h_clf,h_ecbf])
    A = A_ecbf
    b = np.array([b_ecbf])
    p = 1 # relaxation parameter(safety vs convergence)
    P = np.array([[1, 0,0, 0], [0, 1, 0,0], [0,0,0,0],[0,0,0,p]]) # Input cost matrix
    lb = np.array([0,-np.inf,-np.inf,-np.inf])
    # print("P:",P,"\nG:",G,"\nh:",h,"\nA:",A,"\nb:",b)
    sol = qp.solve_qp(P, np.zeros(4), G, h, A, b,lb,solver="clarabel")
    # print(sol)
    if sol is None:
        print("QP solver failed at x: ",x)
        print("Using default control")
        return [-0.01*(v_ref-v), -0.01*(omega_ref-omega)]
    u = sol[0:2]
    return u

def ECBF_get_poles(x:np.ndarray, r_safe:float)->np.ndarray:
    """
    Calcultes the desired poles of the ECBF constraint system
    """
    # h(x) = r_safe -(x² + y²)
    pos_x = x[0]
    pos_y = x[1]
    v = x[2]
    theta = x[3]
    omega = x[4]
    delta = 0.01 # minimum pole value
    p1 = delta + np.max([0,
                 (2*v*(np.cos(theta)*pos_x + np.sin(theta)*pos_y))/(
                     r_safe**2 - pos_x**2 - pos_y**2
                 )])
    p2 = delta + np.max([0,
                 (delta + 2*v*omega*(np.cos(theta)*pos_y - np.sin(theta)*pos_x))/
                 (p1*(r_safe**2 - pos_x**2 - pos_y**2) - \
                     2*v*(np.cos(theta)*pos_x + np.sin(theta)*pos_y))
                 ])
    # print("Desired poles: ",[-p1, -p2])
    return np.array([-p1, -p2])

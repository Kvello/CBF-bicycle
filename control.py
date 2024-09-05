import numpy as np
import numpy as np
import qpsolvers as qp

def CLF_controller(x:np.ndarray,
                   v_ref:float)->np.ndarray:
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
    gamma = 5 # CLF constraint parameter
    V = 0.5*(v-v_ref)**2 # Lyapunov function
    G_clf = np.array([[v-v_ref, 0]]) # Gradient of Lyapunov function
    h_clf = np.array([-gamma*V])
    # QP problem
    P = np.eye(2) # Input cost matrix
    sol = qp.solve_qp(P, np.zeros(2), G_clf, h_clf,solver="clarabel")
    if sol is None:
        print("QP solver failed at x: ",x)
        print("Using default control")
        return [-0.01*(v_ref-v), 0]
    u = sol
    return u

    
    
def ECBF_controller(x:np.ndarray,
                    r_safe:float,
                    v_ref:float)->np.ndarray:
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
    # Canonical controlable form of the ECBF system    
    F = np.matrix([[0,1],[0,0]])
    G = np.matrix([[0],[1]])
    C = np.matrix([1,0])
    eta_b = np.array([
        [r_safe - (pos_x**2 + pos_y**2)], # h(x) = r_safe -(x² + y²)
       [-2*v*(np.cos(theta)*pos_x + np.sin(theta)*pos_y)] # h_dot(x)
       ])
    # CLF constraint
    gamma = 5 # CLF constraint parameter
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
    A_ecbf = np.array([[-2*np.cos(theta)*pos_x + 2*np.sin(theta)*pos_y,
                        2*v*(np.sin(theta)*pos_x-np.cos(theta)*pos_y),
                        -1, 
                        0]])
    b_ecbf = 2*v**2
    G_ecbf = np.array([[0, 0, -1, 0]])
    h_ecbf = (K_alpha@eta_b).item()

    # QP problem
    G = np.block([[G_clf],[G_ecbf]])
    h = np.array([h_clf,h_ecbf])
    A = A_ecbf
    b = np.array([b_ecbf])
    p = 20 # relaxation parameter(safety vs convergence)
    P = np.array([[1, 0,0, 0], [0, 1, 0,0], [0,0,0,0],[0,0,0,p]]) # Input cost matrix
    omega_ss = v_ref*r_safe
    q = np.array([0,-omega_ss,0,0])
    # print("P:",P,"\nG:",G,"\nh:",h,"\nA:",A,"\nb:",b)
    # lb = np.array([-10,-np.pi/6,-np.inf,-np.inf])
    # ub = np.array([10,np.pi/6,np.inf,np.inf])
    lb = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
    ub = np.array([np.inf, np.inf, np.inf, np.inf])
    sol = qp.solve_qp(P, q, G, h, A, b,lb=lb,ub=ub,solver="clarabel")
    # print(sol)
    if sol is None:
        print("QP solver failed at x: ",x)
        print("Using default control")
        return [-0.01*(v_ref-v), 0], 0
    u = sol[0:2]
    delta = sol[3]
    return u, delta

def ECBF_get_poles(x:np.ndarray, r_safe:float)->np.ndarray:
    """
    Calcultes the desired poles of the ECBF constraint system
    """
    # h(x) = r_safe -(x² + y²)
    pos_x = x[0]
    pos_y = x[1]
    v = x[2]
    theta = x[3]
    omega = 0
    delta = 1 # minimum pole value
    h = r_safe - (pos_x**2 + pos_y**2)
    h_dot = -2*v*(np.cos(theta)*pos_x + np.sin(theta)*pos_y)
    h_ddot = -2*v**2

    p1 = np.maximum(0,-h_dot/(h)) + delta
    p2 = np.maximum(0,-(h_ddot + p1*h_dot)/(h_dot + p1*h)) + delta
    return np.array([-p1, -p2])

# ssm_existence.py
# In this script, we demonstrate that the state-space model 
# can fail to exist for certain parameters of a power system
# In turn, using Lemma 3 from the paper, we can construct an alternative model that is guaranteed to exist and has the same eigenvalues as the original system.

# Circuit diagram:
#
#          +------GEN series RL branch------Bus 1
#          |                                  |                                
#          GEN                                RC1                             
#          |                                  |                                 
#          ground                            ground                                

# -----------------------------------------
# Import necessary libraries
# -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, block_diag, inv

# -----------------------------------------
# Parameters and model of series RC circuit
# -----------------------------------------

# Parameters for the series RC circuit
w_base = 2 * np.pi * 60 # rad/s
r_rc = 1 # pu
c_rc = 0.1 # pu

# State-space representation of the series RC circuit
E_rc = c_rc * np.eye(2)
A_rc = c_rc * w_base * np.array([  [0, 1],
                                    [-1, 0]])
B_rc = np.eye(2)
C_rc = np.eye(2)
D_rc = r_rc * np.eye(2)

# ----------------------------------------------------------------------
# Parameters and model of Thevenin equivalent plus constant-power source
# ----------------------------------------------------------------------

# Parameters
r_gen = 0.1
x_gen = 1
p_s = 1
q_s = 0
v_mag = 1.0

# State-space representation
E_gen = np.eye(2)
A_gen =  w_base * np.array( [[-r_gen/x_gen,    1], 
                             [ -1, -r_gen/x_gen]])
B_gen = w_base * np.array(
            [[  1/x_gen,    0, -1/x_gen,    0], 
             [    0,  1/x_gen,    0, -1/x_gen]]) 
B_gen_d = w_base * np.array([[  1/x_gen,    0],
                             [    0,  1/x_gen]])
B_gen_g = w_base * np.array([[-1/x_gen,    0],
                             [    0, -1/x_gen]])
C_gen = np.eye(2)
D_gen_d = np.zeros((2, 2))
D_gen_g = 1/v_mag**2 * np.array([[p_s, q_s],
                                 [-q_s, p_s]])

# --------------------------------------------------
# Construct system-level model for the system 
# --------------------------------------------------

# Construct composite model (stacked model)
E_stack = block_diag(E_gen, E_rc)
A_stack = block_diag(A_gen, A_rc)
B_stack = block_diag(np.block([B_gen_d, B_gen_g]), B_rc)
C_stack = block_diag(C_gen, C_rc)
D_stack = block_diag(np.block([D_gen_d, D_gen_g]), D_rc)

# Construct graph matrices 
Gamma = np.array([1])

F22 = np.kron(Gamma, np.eye(2))
F31 = np.kron(Gamma.T, np.eye(2))
F = np.block([[np.zeros((2,2)), np.zeros((2,2))],
              [np.zeros((2,2)), F22],
              [F31, np.zeros((2,2))]])
G = np.block([[np.eye(2)], [np.zeros((2,2))], [np.zeros((2,2))]])

# Construct system-level model
phi = np.eye(D_stack.shape[0]) - D_stack @ F

if np.linalg.matrix_rank(phi) < phi.shape[0]:
    print("Matrix (I - D_stack @ F) is not invertible. The state-space model does not exist for this system. We will construct an alternative model instead.")
    E_sys = block_diag(E_gen, E_rc, np.zeros((4, 4)))

    A_sys = np.block([  [A_gen, np.zeros((2, 2)), B_gen_g, np.zeros((2, 2))],
                        [np.zeros((2, 2)), A_rc, np.zeros((2, 2)), B_rc],
                        [np.zeros((2, 2)), F22 @ C_rc, -np.eye(2), F22 @ D_rc],
                        [F31 @ C_gen, np.zeros((2, 2)), F31 @ D_gen_g, -np.eye(2)]])

    eigenvalues, eigenvectors = eig(A_sys, E_sys)

    print("Eigenvalues:\n", eigenvalues)

    # This model serves to validate the eigenvalues of the descriptor system
    # for when the state-space model fail to exist.

    # index reduction
    # redefine E_rc, A_rc, B_rc, taking E_rc inverse
    A_rc = np.linalg.inv(E_rc) @ A_rc
    B_rc = np.linalg.inv(E_rc) @ B_rc

    omega_red = np.eye(2) + B_gen_g @ np.linalg.inv(B_rc @ D_gen_g) @ np.linalg.inv(C_rc) @ D_rc @ C_gen
    A_red = np.linalg.inv(omega_red) @ (A_gen + B_gen_g @ np.linalg.inv(B_rc @ D_gen_g) @ A_rc @ np.linalg.inv(C_rc) @ D_rc @ C_gen - B_gen_g @ np.linalg.inv(B_rc @ D_gen_g) @ B_rc @ C_gen)

    print("A_red eigenvalues:")
    print(np.linalg.eigvals(A_red))

else:
    print("Matrix (I - D_stack @ F) is invertible. The state-space model exists for this system.")
    E_sys = E_stack
    A_sys = (A_stack + B_stack @ F @ inv(phi) @ C_stack)
    B_sys = B_stack @ F @ phi @ D_stack @ G + B_stack @ G






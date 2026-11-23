# -----------------------------------------
# Import necessary libraries
# -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, eig, block_diag, inv

# -----------------------------------------
# Parameters and model of series RC circuit
# -----------------------------------------

# Parameters for the series RC circuit
w_base = 2 * np.pi * 60 # rad/s
r_rc1 = 1 # pu
c_rc1 = 0 # pu

# Make a second RC circuit with different parameters
r_rc2 = 0.5 # pu
c_rc2 = 0.2 # pu

E_rc2 = c_rc2 * np.eye(2)
A_rc2 = c_rc2 * np.array([[0, w_base], 
                 [-w_base, 0]])
B_rc2 = np.eye(2)
C_rc2 = np.eye(2)
D_rc2 = r_rc2 * np.eye(2)

# ----------------------------------------------------------------------
# Parameters and model of Thevenin equivalent plus constant-power source
# ----------------------------------------------------------------------

# Parameters
r_gen = 0.1 + 0.1
x_gen = 1 + 1
p_s = 0
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

C_gen = np.array([[1, 0],
                  [0, 1]])

D_gen_d = np.zeros((2, 2))

D_gen_g = 1/v_mag**2 * np.array([[p_s, q_s],
                                 [-q_s, p_s]])


# -----------------------------------------
# Construct composite model (stacked model)
# -----------------------------------------
E_stack = block_diag(E_gen, E_rc2)
A_stack = block_diag(A_gen, A_rc2)
B_stack = block_diag(np.block([B_gen_d, B_gen_g]), B_rc2)
C_stack = block_diag(C_gen, C_rc2)
D_stack = block_diag(np.block([D_gen_d, D_gen_g]), D_rc2)

# -----------------------------------------
# Construct Graph matrices
# -----------------------------------------

Gamma = np.array([[1]])

F22 = np.kron(Gamma , np.eye(2))
F31 = np.kron(Gamma.T, np.eye(2))

dg = 2 
gg = 2
gs = 2
yg = 2
ys = 2

F = np.block([  [np.zeros((dg, yg)), np.zeros((dg, ys))],
                [np.zeros((gg, yg)), F22],
                [F31, np.zeros((gs, ys))],
                ])

G = np.block([[np.eye(dg)],
              [np.zeros((gg, dg))],
              [np.zeros((gs, dg))],
            ])


# -----------------------------------------
# Construct system-level model
# -----------------------------------------

phi = np.eye(D_stack.shape[0]) - D_stack @ F
E_sys = E_stack

A_sys = (A_stack + B_stack @ F @ inv(phi) @ C_stack)
B_sys = B_stack @ F @ phi @ D_stack @ G + B_stack @ G


# -----------------------------------------
# Eigenvalue analysis of the system-level model
# -----------------------------------------
eigvals, eigvecs = eig(A_sys, E_sys)

print("Eigenvalues of the system-level model:")
print(eigvals)
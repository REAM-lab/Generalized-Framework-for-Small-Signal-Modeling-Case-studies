# -----------------------------------------
# Import necessary libraries
# -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, eig, block_diag

# -----------------------------------------
# Parameters and model of series RC circuit
# -----------------------------------------

# Parameters for the series RC circuit
w_base = 2 * np.pi * 60 # rad/s
r_rc1 = 1 # pu
c_rc1 = 0.1 # pu

# State-space representation of the series RC circuit
A_rc1 = np.array([[0, w_base], 
                 [-w_base, 0]])

B_rc1 = np.array([[1/c_rc1, 0],
                 [0, 1/c_rc1]])

C_rc1 = np.array([[1, 0],
                 [0, 1]])

D_rc1 = np.array([[r_rc1, 0],
                 [0, r_rc1]])

# Make a second RC circuit with different parameters
r_rc2 = 0.5 # pu
c_rc2 = 0.2 # pu

A_rc2 = np.array([[0, w_base],
                 [-w_base, 0]])

B_rc2 = np.array([[1/c_rc2, 0],
                 [0, 1/c_rc2]])

C_rc2 = np.array([[1, 0],
                 [0, 1]])

D_rc2 = np.array([[r_rc2, 0],
                 [0, r_rc2]])

# ----------------------------------------------------------------------
# Parameters and model of Thevenin equivalent plus constant-power source
# ----------------------------------------------------------------------

# Parameters
r_gen = 0.1
x_gen = 1
p_s = 0
q_s = 0
v_mag = 1.0

# State-space representation
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
# Parameters and model of series RL circuit
# -----------------------------------------

r_br = 0.1
x_br = 1

A_br = w_base * np.array(
                        [[-r_br/x_br,        1], 
                         [      -1, -r_br/x_br]])

B_br = w_base * np.array(
                        [[  1/x_br,    0, -1/x_br,    0], 
                         [    0,  1/x_br,    0, -1/x_br]])

C_br = np.array([[1, 0],
                 [0, 1]])

D_br = np.zeros((2, 4))

# -----------------------------------------
# Construct composite model (stacked model)
# -----------------------------------------

A_stack = block_diag(A_gen, A_rc1, A_rc2, A_br)
B_stack = block_diag(np.block([B_gen_d, B_gen_g]), B_rc1, B_rc2, B_br)
C_stack = block_diag(C_gen, C_rc1, C_rc2, C_br)
D_stack = block_diag(np.block([D_gen_d, D_gen_g]), D_rc1, D_rc2, D_br)

# -----------------------------------------
# Construct Graph matrices
# -----------------------------------------

Gamma = np.array([[1, 0]])
Inc = np.array([[1],
                [1]])
Inc_or = np.array([[1],
                   [-1]])

F22 = np.kron(Gamma , np.eye(2))
F31 = np.kron(Gamma.T, np.eye(2))
F33 = np.kron(Inc_or, np.eye(2))
F42 = np.kron( 0.5 * ( np.kron(Inc.T, np.array([[1],[1]])) + np.kron(Inc_or.T, np.array([[-1], [1]])) ) , np.eye(2))

dg = 2 
gg = 2
gb = 4
gs = 4
yg = 2
ys = 4
yb = 2

F = np.block([  [np.zeros((dg, yg)), np.zeros((dg, ys)), np.zeros((dg, yb))],
                [np.zeros((gg, yg)), F22, np.zeros((gg, yb))],
                [F31, np.zeros((gs, ys)), F33],
                [np.zeros((gb, yg)), F42, np.zeros((gb, yb))]])

G = np.block([[np.eye(dg)],
              [np.zeros((gg, dg))],
              [np.zeros((gs, dg))],
              [np.zeros((gb, dg))]])


# -----------------------------------------
# Construct system-level model
# -----------------------------------------

phi = np.eye(D_stack.shape[0]) - D_stack @ F
A_sys = A_stack + B_stack @ F @ np.linalg.inv(phi) @ C_stack
B_sys = B_stack @ F @ phi @ D_stack @ G + B_stack @ G


# -----------------------------------------
# Eigenvalue analysis of the system-level model
# -----------------------------------------
eigvals, eigvecs = eig(A_sys)

print("Eigenvalues of the system-level model:")
print(eigvals)
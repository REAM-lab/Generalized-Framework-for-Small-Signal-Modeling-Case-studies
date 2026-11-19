# ssm_existence.py
# In this script, we demonstrate that the standard space-state model 
# can fail to exist for certain parameters of a power system
# In turn, using Lemma 3 from the paper, we can construct a descriptor model that 
# is well-defined. 


# -----------------------------------------
# Import necessary libraries
# -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, eig

# -----------------------------------------
# Parameters and model of series RC circuit
# -----------------------------------------

# Parameters for the series RC circuit
w_base = 2 * np.pi * 60 # rad/s
r_rc = 1 # pu
c_rc = 0.1 # pu

# State-space representation of the series RC circuit
A_rc = np.array([[0, w_base], 
                 [-w_base, 0]])

B_rc = np.array([[1/c_rc, 0],
                 [0, 1/c_rc]])

C_rc = np.array([[1, 0],
                 [0, 1]])

D_rc = np.array([[r_rc, 0],
                 [0, r_rc]])

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

D_gen_g = 1/v_mag**2 * np.array([[p_s, q_s],
                                 [-q_s, p_s]])

# -----------------------------------------
# Construct composite model (stacked model)
# -----------------------------------------
phi = np.block([[np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))],
                [np.zeros((2, 2)), np.eye(2), -D_rc],
                [np.zeros((2, 2)), -D_gen_g, np.eye(2)]])

A_stack = np.block([[A_gen, np.zeros((2, 2))],
                    [np.zeros((2, 2)), A_rc]])
B_stack = np.block([[B_gen, np.zeros((2, 2))],
                    [np.zeros((2, 4)), B_rc]])
C_stack = np.block([[C_gen, np.zeros((2, 2))],
                    [np.zeros((2, 2)), C_rc]])

# ------------------------------------------------
# Construct the interconnection matrices (Lemma 3)
# ------------------------------------------------
F22 = np.eye(2)
F31 = np.eye(2)

F = np.block([[np.zeros((2,2)), np.zeros((2,2))],
              [np.zeros((2,2)), F22],
              [F31, np.zeros((2,2))]])


G = np.block([[np.eye(2)], [np.zeros((2,2))], [np.zeros((2,2))]])

# -----------------------------------------
# Construct the descriptor model matrices
# -----------------------------------------
E = np.diag([1, 1, 1, 1, 0, 0, 0, 0])

A_new = np.block([[A_gen, np.zeros((2, 2)), B_gen_g, np.zeros((2, 2))],
                  [np.zeros((2, 2)), A_rc, np.zeros((2, 2)), B_rc],
                  [np.zeros((2, 2)), F22 @ C_rc, -np.eye(2), F22 @ D_rc],
                  [F31 @ C_gen, np.zeros((2, 2)), F31 @ D_gen_g, -np.eye(2)]])

print("A_new:")
print(A_new)

eigenvalues, eigenvectors = eig(A_new, E)

print("Eigenvalues:\n", eigenvalues)

# -----------------------------------------
# Construct the index-reduced model
# -----------------------------------------

# This model serves to validate the eigenvalues of the descriptor system
# for when the state-space model fail to exist.

# index reduction
omega_red = np.eye(2) + B_gen_g @ np.linalg.inv(B_rc @ D_gen_g) @ np.linalg.inv(C_rc) @ D_rc @ C_gen
A_red = np.linalg.inv(omega_red) @ (A_gen + B_gen_g @ np.linalg.inv(B_rc @ D_gen_g) @ A_rc @ np.linalg.inv(C_rc) @ D_rc @ C_gen - B_gen_g @ np.linalg.inv(B_rc @ D_gen_g) @ B_rc @ C_gen)

print("omega_red:")
print(omega_red)
print("A_red:")
print(A_red)

print("A_red eigenvalues:")
print(np.linalg.eigvals(A_red))
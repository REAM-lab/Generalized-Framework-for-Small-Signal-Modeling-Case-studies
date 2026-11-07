import numpy as np
from solve_dae.integrate import solve_dae
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import null_space, eig

w_base = 2 * np.pi * 60
r_rc = 1
c_rc = 0.1
A_rc = np.array([[0, w_base], 
                 [-w_base, 0]])

B_rc = np.array([[1/c_rc, 0],
                 [0, 1/c_rc]])

C_rc = np.array([[1, 0],
                 [0, 1]])

D_rc = np.array([[r_rc, 0],
                 [0, r_rc]])

r_gen = 0.1
x_gen = 1
p_s = 1
q_s = 0
v_mag = 1.0

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

phi = np.block([[np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))],
                [np.zeros((2, 2)), np.eye(2), -D_rc],
                [np.zeros((2, 2)), -D_gen_g, np.eye(2)]])

A_stack = np.block([[A_gen, np.zeros((2, 2))],
                    [np.zeros((2, 2)), A_rc]])
B_stack = np.block([[B_gen, np.zeros((2, 2))],
                    [np.zeros((2, 4)), B_rc]])
C_stack = np.block([[C_gen, np.zeros((2, 2))],
                    [np.zeros((2, 2)), C_rc]])

Z1 = np.eye(2) @ D_rc
Z2 = np.zeros((2, 2))
Z3 = np.eye(2) @ D_gen_g

F = np.block([[np.zeros((2,2)), np.zeros((2,2))],
              [np.zeros((2,2)), np.eye(2)],
              [np.eye(2), np.zeros((2,2))]])
F22 = np.eye(2)
F31 = np.eye(2)

G = np.block([[np.eye(2)], [np.zeros((2,2))], [np.zeros((2,2))]])

E = np.diag([1, 1, 1, 1, 0, 0, 0, 0])

A_new = np.block([[A_gen, np.zeros((2, 2)), B_gen_g, np.zeros((2, 2))],
                  [np.zeros((2, 2)), A_rc, np.zeros((2, 2)), B_rc],
                  [np.zeros((2, 2)), F22 @ C_rc, -np.eye(2), F22 @ D_rc],
                  [F31 @ C_gen, np.zeros((2, 2)), F31 @ D_gen_g, -np.eye(2)]])

print("A_new:")
print(A_new)

eigenvalues, eigenvectors = eig(A_new, E)

print("Eigenvalues:\n", eigenvalues)

# index reduction
omega_red = np.eye(2) + B_gen_g @ np.linalg.inv(B_rc @ D_gen_g) @ np.linalg.inv(C_rc) @ D_rc @ C_gen
A_red = np.linalg.inv(omega_red) @ (A_gen + B_gen_g @ np.linalg.inv(B_rc @ D_gen_g) @ A_rc @ np.linalg.inv(C_rc) @ D_rc @ C_gen - B_gen_g @ np.linalg.inv(B_rc @ D_gen_g) @ B_rc @ C_gen)

print("omega_red:")
print(omega_red)
print("A_red:")
print(A_red)

print("A_red eigenvalues:")
print(np.linalg.eigvals(A_red))
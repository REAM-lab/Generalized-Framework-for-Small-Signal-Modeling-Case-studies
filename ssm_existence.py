import numpy as np

w_base = 2 * np.pi * 60
r_rc = 1
c_rc = 1
A_rc = np.array([[0, w_base], 
                 [-w_base, 0]])

B_rc = np.array([[1/c_rc, 0],
                 [0, 1/c_rc]])

C_rc = np.array([[1, 0],
                 [0, 1]])

D_rc = np.array([[r_rc, 0],
                 [0, r_rc]])

r_gen = 0.0001
x_gen = 0.001
p_s = 0.99999
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


# Small-signal model of the system according to [16]

Omega = (np.eye(2) - D_rc @ D_gen_g)

A11 = A_rc + B_rc @ D_gen_g @ np.linalg.inv(Omega) @ C_rc 
A12 = B_rc @ C_gen + B_rc @ D_gen_g @ np.linalg.inv(Omega) @ D_rc @ C_gen
A21 = B_gen_g @ np.linalg.inv(Omega) @ C_rc
A22 = A_gen + B_gen_g @ np.linalg.inv(Omega) @ D_rc @ C_gen

A = np.block([[A11, A12],
              [A21, A22]])

# Compute the eigenvalues of the system
eigenvalues = np.linalg.eigvals(A)

# Order the eigenvalues by their real parts
eigenvalues = eigenvalues[np.argsort(eigenvalues.real)]

# Print the eigenvalues
for i, eig in enumerate(eigenvalues):
    print(f"Eigenvalue {i+1}: {eig.real:.4f} + {eig.imag:.4f}j")
import numpy as np
from solve_dae.integrate import solve_dae
import matplotlib.pyplot as plt
import cvxpy as cp

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

#r_gen = 0.0001
#x_gen = 0.001
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

print("phi:")
print(phi)
print("A_stack:")
print(A_stack)
print("B_stack:")
print(B_stack)
print("C_stack:")
print(C_stack)

# Small-signal model of the system according to [16]
if np.linalg.det(phi) != 0:
    
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
    #eigenvalues = eigenvalues[np.argsort(eigenvalues.real)]

    # Print the eigenvalues
    for i, eig in enumerate(eigenvalues):
        print(f"Eigenvalue {i+1}: {eig.real:.4f} + {eig.imag:.4f}j")


# Differential equations for the system
A_gen_rc = np.block([[A_gen, np.zeros((2, 2))],
                         [B_rc @ C_gen, A_rc]])

B_gen_rc = np.block([[B_gen_d], [np.zeros((2, 2))]])
B_vbus = np.block([[B_gen_g], [np.zeros((2, 2))]])
B_vbus2 = np.block([[np.zeros((2, 2))], [B_rc]])

def dae_system(t, x, dx, u):

    # Unpack the state variables
    i_the_d, i_the_q, v_rc_d, v_rc_q, v_bus_d, v_bus_q = x

    # Unpack derivative variables
    di_the_d_dt, di_the_q_dt, dv_rc_d_dt, dv_rc_q_dt, dv_bus_d_dt, dv_bus_q_dt = dx

    # Unpack the input variables
    v_ref_d = u['v_d_ref'](t)
    v_ref_q = u['v_q_ref'](t)

    # Stack i_the and v_rc
    dx_gen_rc = np.array([[di_the_d_dt], [di_the_q_dt], [dv_rc_d_dt], [dv_rc_q_dt]])
    x_gen_rc = np.array([[i_the_d], [i_the_q], [v_rc_d], [v_rc_q]])

    eq1 = -dx_gen_rc + A_gen_rc @ x_gen_rc + B_gen_rc @ np.array([[v_ref_d], [v_ref_q]]) + B_vbus @ np.array([[v_bus_d], [v_bus_q]]) + B_vbus2 @ np.array([[v_bus_d], [v_bus_q]])
    eq2 = (-np.eye(2) + D_rc @ D_gen_g) @ np.array([[v_bus_d], [v_bus_q]]) + C_rc @ np.array([[v_rc_d], [v_rc_q]]) + D_rc @ C_gen @ np.array([[i_the_d], [i_the_q]])

    return [eq1.flatten()[0], eq1.flatten()[1], eq1.flatten()[2], eq1.flatten()[3], eq2.flatten()[0], eq2.flatten()[1]]

# Compute initial conditions for the system
v_bus_d0 = 0
v_bus_q0 = 0.0
v_ref_d0 = 0
v_ref_q0 = 0.0

x0 = np.zeros(6)  # Initial state vector
xp0 = np.zeros_like(x0)  # Initial derivatives are zero

def smooth_step(t, t0, width, initial_value, final_value):
    return initial_value + (final_value - initial_value) * 0.5 * (1 + np.tanh((t - t0)/width))

def v_d_ref(t):
      return smooth_step(t, t0=1, width=1e-1, initial_value=0, final_value=1)

def v_q_ref(t):
      return 0

inputs = {'v_d_ref': v_d_ref, 'v_q_ref': v_q_ref}
# Solve DAEs
t_max = 2
solution = solve_dae(dae_system, 
                        [0, t_max], 
                        x0, 
                        xp0,
                        dense_output=True, 
                        args=(inputs,),
                        max_step=1e-3, 
                        method='Radau')
print(f"success: {solution.success}")
print(f"status: {solution.status}")
print(f"message: {solution.message}")

tps = np.linspace(0, t_max, 500)
solution = solution.sol(tps)[0]

i_the_d = solution[0]
i_the_q = solution[1]
v_rc_d = solution[2]
v_rc_q = solution[3]


m=6
n=4
T = cp.Variable((n, m))
E = cp.Variable((n, n))

constraints = [T @ phi == E * B_stack, cp.trace(E) <= 10]
objective = cp.Maximize(cp.trace(E))
prob = cp.Problem(objective, constraints)
prob.solve()

print("E:")
print(E.value)
print("T:")
print(T.value)

E = E.value
T = T.value

# rank of E
rank_E = np.linalg.matrix_rank(E)
print(f"Rank of E: {rank_E}")

F = np.block([[np.zeros((2,2)), np.zeros((2,2))],
              [np.zeros((2,2)), np.eye(2)],
              [np.eye(2), np.zeros((2,2))]])

G = np.block([[np.eye(2)], [np.zeros((2,2))], [np.zeros((2,2))]])

# round E to 10 decimal places
#E = np.round(E, 4)
print("E after rounding:")
print(E)

rank_E = np.linalg.matrix_rank(E)
print(f"Rank of E: {rank_E}")



#T = np.round(T, 50)
print("T after rounding:")
print(T)

A_sys = E @ A_stack + T @ F @ C_stack
B_sys = T @ G

# simulate DAE system with the new A_sys and B_sys matrices
# E dotx = A_sys x + B_sys u

from scipy.linalg import eig
lam = eig(A_sys, E, right=False)

print("Generalized eigenvalues:")
print(lam)

print("A_sys:")
print(A_sys)
print("B_sys:")
print(B_sys)
print("E:")
print(E)

def dae_system_new(t, x, dx, u):
    # Unpack the state variables
    i_the_d, i_the_q, v_rc_d, v_rc_q = x

    # Unpack derivative variables
    di_the_d_dt, di_the_q_dt, dv_rc_d_dt, dv_rc_q_dt = dx

    # Unpack the input variables
    v_ref_d = u['v_d_ref'](t)
    v_ref_q = u['v_q_ref'](t)

    # Stack i_the and v_rc
    dx_stack = np.array([[di_the_d_dt], [di_the_q_dt], [dv_rc_d_dt], [dv_rc_q_dt]])
    x_stack = np.array([[i_the_d], [i_the_q], [v_rc_d], [v_rc_q]])

    eq1 = -E @ dx_stack + A_sys @ x_stack + B_sys @ np.array([[v_ref_d], [v_ref_q]])

    return eq1.flatten()

x0_new = x0[:4]  # Only take the first 4 states for the new system
xp0_new = xp0[:4]  # Only take the first 4 derivatives for
solution = solve_dae(dae_system_new, 
                        [0, t_max], 
                        x0_new, 
                        xp0_new,
                        dense_output=True, 
                        args=(inputs,),
                        max_step=1e-2, 
                        method='Radau')
print(f"success: {solution.success}")
print(f"status: {solution.status}")
print(f"message: {solution.message}")

tps = np.linspace(0, t_max, 1000)
solution = solution.sol(tps)[0]

i_the_d_des = solution[0]


# final values of i_the_d_des
from scipy.optimize import fsolve

def initial_conditions(x0):
    # Unpack the state variables
    i_the_d, i_the_q, v_rc_d, v_rc_q, v_bus_d, v_bus_q = x0

    # Compute the derivatives (set to zero for steady-state)
    dx0 = np.zeros_like(x0)

    # Compute the equations at steady-state
    eqs = dae_system(0, x0, dx0, {'v_d_ref': lambda t: 1, 'v_q_ref': lambda t: 0})

    return eqs

# Initial guess for the state variables
x0_guess = np.array([0.0, 0.0, v_mag, 0.0, v_bus_d0, v_bus_q0])

# Solve for the initial conditions
x0 = fsolve(initial_conditions, x0_guess)


# Print the initial conditions
print("Initial conditions:")
print(f"i_the_d: {x0[0]:.4f}, i_the_q: {x0[1]:.4f}, v_rc_d: {x0[2]:.4f}, v_rc_q: {x0[3]:.4f}, v_bus_d: {x0[4]:.4f}, v_bus_q: {x0[5]:.4f}")





# plot i_the_d and i_the_d_des
plt.figure()
plt.plot(tps, i_the_d, label='i_the_d')
plt.plot(tps, i_the_d_des, label='i_the_d_des', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Current (p.u.)')
plt.title('Comparison of i_the_d and i_the_d_des')
plt.legend()
#plt.grid()
#plt.show()
#plt.close()

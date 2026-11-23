# -----------------------------------------
# Import necessary libraries
# -----------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import null_space, eig, block_diag, inv
from juliacall import Main as jl
import scienceplots
import os
from pathlib import Path

# -----------------------------------------
# Parameters and model of series RC circuit
# -----------------------------------------

# Parameters for the series RC circuit
w_base = 2 * np.pi * 60 # rad/s
r_rc1 = 1 # pu
c_rc1 = 0 # pu

# State-space representation of the series RC circuit
E_rc1 = c_rc1 * np.eye(2)
A_rc1 = c_rc1 * w_base * np.array([ [0, 1],
                                    [-1, 0]])
B_rc1 = np.eye(2)
C_rc1 = np.eye(2)
D_rc1 = r_rc1 * np.eye(2)

# Make a second RC circuit with different parameters
r_rc2 = 0.5 # pu
c_rc2 = 0.2 # pu

E_rc2 = c_rc2 * np.eye(2)
A_rc2 = c_rc2 * w_base * np.array([[0, 1],
                                   [-1, 0]])
B_rc2 = np.eye(2)
C_rc2 = np.eye(2)
D_rc2 = r_rc2 * np.eye(2)

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


# -----------------------------------------
# Parameters and model of series RL circuit
# -----------------------------------------

r_br = 0.1
x_br = 1
E_br = np.eye(2)
A_br = w_base * np.array(
                        [[-r_br/x_br,        1], 
                         [      -1, -r_br/x_br]])

B_br = w_base * np.array(
                        [[  1/x_br,    0, -1/x_br,    0], 
                         [    0,  1/x_br,    0, -1/x_br]])

C_br = np.array([[1, 0],
                 [0, 1]])

D_br = np.zeros((2, 4))

# --------------------------------------------------
# Construct system-level model for the system (as is)
# --------------------------------------------------

# Construct composite model (stacked model)
E_stack = block_diag(E_gen, E_rc1, E_rc2, E_br)
A_stack = block_diag(A_gen, A_rc1, A_rc2, A_br)
B_stack = block_diag(np.block([B_gen_d, B_gen_g]), B_rc1, B_rc2, B_br)
C_stack = block_diag(C_gen, C_rc1, C_rc2, C_br)
D_stack = block_diag(np.block([D_gen_d, D_gen_g]), D_rc1, D_rc2, D_br)

# Construct graph matrices 
Gamma = np.array([[1, 0]])
Inc = np.array([[1],
                [1]])
Inc_or = np.array([[1],
                   [-1]])

# Construct interconnection matrices
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

# Construct system-level model
phi = np.eye(D_stack.shape[0]) - D_stack @ F
E_sys = E_stack
A_sys = (A_stack + B_stack @ F @ inv(phi) @ C_stack)
B_sys = B_stack @ F @ phi @ D_stack @ G + B_stack @ G

# Compute eigenvalues
eigvals, eigvecs = eig(A_sys, E_sys)

print("Eigenvalues of the system-level model (1):")
print(eigvals)

# -------------------------------------------------------
# Construct system-level model for the equivalent system 
# -------------------------------------------------------

# Agregate the generator and the series RL circuit into a single equivalent generator
r_gen_br = r_gen + r_br
x_gen_br = x_gen + x_br

# Redefine the generator model with the series RL circuit model
E_gen_br = np.eye(2)
A_gen_br =  w_base * np.array( [[-r_gen_br/x_gen_br,    1], 
                             [ -1, -r_gen_br/x_gen_br]])
B_gen_br = w_base * np.array(
            [[  1/x_gen_br,    0, -1/x_gen_br,    0], 
             [    0,  1/x_gen_br,    0, -1/x_gen_br]]) 
B_gen_br_d = w_base * np.array([[  1/x_gen_br,    0],
                             [    0,  1/x_gen_br]])
B_gen_br_g = w_base * np.array([[-1/x_gen_br,    0],
                             [    0, -1/x_gen_br]])
C_gen_br = np.eye(2)
D_gen_br_d = np.zeros((2, 2))
D_gen_br_g = 1/v_mag**2 * np.array([[p_s, q_s],
                                 [-q_s, p_s]])

# Construct composite model (stacked model) for the equivalent system
E_stack = block_diag(E_gen_br, E_rc2)
A_stack = block_diag(A_gen_br, A_rc2)
B_stack = block_diag(np.block([B_gen_br_d, B_gen_br_g]), B_rc2)
C_stack = block_diag(C_gen_br, C_rc2)
D_stack = block_diag(np.block([D_gen_br_d, D_gen_br_g]), D_rc2)

# Construct graph matrices for the equivalent system
# No incidence matrix because the branch is now part of the generator model
Gamma = np.array([[1]])

# Construct interconnection matrices for the equivalent system
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

# Construct system-level model for the equivalent system
phi = np.eye(D_stack.shape[0]) - D_stack @ F
E_sys_red = E_stack
A_sys_red = (A_stack + B_stack @ F @ inv(phi) @ C_stack)
B_sys_red = B_stack @ F @ phi @ D_stack @ G + B_stack @ G

# Compute eigenvalues
eigvals, eigvecs = eig(A_sys_red, E_sys_red)

print("Eigenvalues of the reduced system-level model (2):")
print(eigvals)

# --------------------------------------------
# Simulate the system-level models using Julia
# --------------------------------------------

# We use Julia OrdinaryDiffEq.jl because it has the solver Rodas5P() which is suitable for stiff systems and mass systems.

jl.seval("""
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
""")

# Transfer the system matrices to Julia
jl.n_sys = A_sys.shape[0]
jl.E_sys = E_sys
jl.A_sys = A_sys
jl.B_sys = B_sys

# Transfer the reduced system matrices to Julia
jl.n_red = A_sys_red.shape[0]
jl.E_sys_red = E_sys_red
jl.A_sys_red = A_sys_red
jl.B_sys_red = B_sys_red

jl.seval("""
function step(t)
    if t > 0.1  
        return 1.0
    else
        return 0.0
    end
end

function model_1!(dx, x, p, t)
    dx .= (A_sys * x + B_sys * [step(t);0])
end

function model_2!(dx, x, p, t)
    dx .= (A_sys_red * x + B_sys_red * [step(t);0])
end

f1 = ODEFunction(model_1!, mass_matrix=E_sys)
f2 = ODEFunction(model_2!, mass_matrix=E_sys_red)

tspan = (0.0, 1.0)
x0 = zeros(n_sys)
problem1 = ODEProblem(f1, x0, tspan)
sol1 = solve(problem1, Rodas5P())

x0_red = zeros(n_red)
problem2 = ODEProblem(f2, x0_red, tspan)
sol2 = solve(problem2, Rodas5P())

if !SciMLBase.successful_retcode(sol1)
    error("Julia integration failed: $(sol1.retcode)")
end
if !SciMLBase.successful_retcode(sol2)
    error("Julia integration failed: $(sol2.retcode)")
end
""")

# Retrieve the solution from Julia
t1 = jl.seval("sol1.t")
x1 = jl.seval("[sol1.u[i][1] for i in eachindex(sol1.t)]")

t2 = jl.seval("sol2.t")
x2 = jl.seval("[sol2.u[i][1] for i in eachindex(sol2.t)]")

# Plotting settings
plt.style.use(['science','ieee'])
plt.rcParams['text.usetex'] = False
plt.rc('font',**{'family':'serif','serif':['Times'], 'size': 7})
plt.rcParams['axes.formatter.useoffset'] = False # this prevent scientific notation in y-axis

# Create a plot
fig, ax = plt.subplots(figsize=(3, 3), dpi=1000)

# Plot the results
ax.plot(t1, x1, linestyle='solid', color='blue', label='System')
ax.plot(t2, x2, linestyle='--', color='red', label='Reduced System')
ax.set_xlim([0, 0.8])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Generator current of d-axis (pu)')
ax.legend()

# Make inset axes between 0.1 and 0.2 seconds
axins = ax.inset_axes([0.55, 0.5, 0.35, 0.35])
axins.plot(t1, x1, linestyle='solid', color='blue', label='System')
axins.plot(t2, x2, linestyle='--', color='red', label='Reduced System')
axins.set_xlim([0.1, 0.2])
axins.set_ylim([0.1, 0.3])
ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5, linewidth=0.7)


directory = os.path.join(Path(__file__).resolve().parent)

# Save figure
plt.savefig(os.path.join(directory, "comparison_no_shunt.pdf"), dpi=1000)



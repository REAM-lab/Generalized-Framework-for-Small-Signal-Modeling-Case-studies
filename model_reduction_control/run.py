import os
from pathlib import Path

import control as ct
import cvxpy as cp
import numpy as np
import polars as pl
from sting.utils.transformations import abc2dq0

# Local packages
from cmaspy.partial_state_feedback import (
    mas_output_feedback,
    single_agent_output_feedback,
)
from scipy.linalg import eigvals, solve_continuous_are
from sting import main
from sting.modules.model_order_reduction.balanced_truncation import BalancedTruncation
from sting.utils.dynamical_systems import smooth_step
from wscc_9 import wscc_9
from wscc_9_controller import wscc_9_with_controller

# ------------------------------------------------------------
# WSCC 9 bus and simulation setup
# ------------------------------------------------------------

# Location of all outputs
case_directory = os.path.join(Path(__file__).resolve().parent)

# Load the WSCC 9 bus system from the default datasets in STING
system = wscc_9(case_directory=case_directory)
system.apply("post_system_init", system)

# Create input signal to GFMI (proposed project)
inputs = {
    'gfmi_18a_0': {
        'v_ref': lambda t: smooth_step(t, step_time=0.1, initial_value=0.0, final_value=0.10, transient_width=5e-3),
        }
}
# Simulation length in seconds
t_max = 1.5

# ------------------------------------------------------------
# Model-order reduction
# ------------------------------------------------------------

# Construct a small-signal model
system, ssm = main.run_ssm(system=system, case_directory=case_directory)

# Create a reduced order model of all components in the zone labeled as "external".
# We will then connect this reduced order model to the zone labeled "study", which
# consists of a grid forming inverter (GFMI 18A) at bus 2.

# Vanilla balanced truncation removing the states that are hardest to control and observe.
balanced_truncation = {
    "external": BalancedTruncation(r=33, method="truncate")
    }
# Construct a reduced-order model (ROM).
rom = main.run_model_reduction(ssm=ssm, reductions=balanced_truncation)

# Compute stats of the ROM and FOM
ss_fom = ct.ss(*rom.system.linear_subsystems[0].full_order_model.data)
ss_rom = ct.ss(*rom.system.linear_subsystems[0].reduced_order_model.data)

print("Eigenvalue of ROM", np.max(np.linalg.eigvals(ss_rom.A).real))

print("Full-order model has", ss_fom.nstates, "states")
print("Reduced-order model (without proposed project) ", ss_rom.nstates, "states")
print("H_2 Error", round(100 * ct.norm(ss_fom - ss_rom,p=2) / ct.norm(ss_fom, p=2),3), "%")
print("H_inf Error", round(100 * ct.linfnorm(ss_fom - ss_rom)[0] / ct.linfnorm(ss_fom)[0], 3), "%")

print("Max eigenvalue of the ROM + study area: ", np.max(np.linalg.eigvals(rom.model.A).real))

# Simulate the full-order model
ssm.output_directory = os.path.join(case_directory, "outputs", "small_signal_model")
os.makedirs(ssm.output_directory , exist_ok=True)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

# Simulate the reduced-order model
rom.output_directory = os.path.join(case_directory, "outputs", "model_order_reduction")
os.makedirs(rom.output_directory, exist_ok=True)
rom.simulate_ssm(t_max=t_max, inputs=inputs)

# ------------------------------------------------------------
# Output feedback control design
# ------------------------------------------------------------

# Design of the output feedback control
# Matrix of A of ROM
A_c = rom.model.A
# inputs = [p_ref, q_ref, v_ref, ...]
B_c = rom.model.B[:, 0:1] # take only p_ref, q_ref, v_ref of the GFM in the proposed project

# outputs = []
C_c = np.zeros((5, A_c.shape[0]))
C_c[0, 1] = 1 # w_pc
C_c[1, 7] = 1 # i_vsc_d
C_c[2, 8] = 1 # i_vsc_q
C_c[3, 9] = 1 # i_bus_d
C_c[4, 10] = 1 # i_bus_q 

D_c = np.zeros((C_c.shape[0], B_c.shape[1]))

Q = 10**4*np.eye(A_c.shape[0])

R = 10**6*np.eye(B_c.shape[1])

solve_settings = {'solver': cp.MOSEK,
                  'verbose': False}

# Solve CARE to obtain P
P = solve_continuous_are(A_c, B_c, Q, R)

# Use MAS output feedback
alpha_coef = 100
beta_coef = 0
gamma_coef = 0
mas_out = mas_output_feedback(A_c, [B_c], [C_c], [D_c], [Q], [R], [P], alpha_coef, beta_coef, gamma_coef, **solve_settings)

# Print dominant eigenvalues of the closed-loop system
eigenvalues = eigvals(mas_out.Acl_F)
dominant_eigenvalue = eigenvalues[np.argmax(eigenvalues.real)]
print("Dominant eigenvalues of the closed-loop system: ", dominant_eigenvalue)

# Save closed-loop a matrix as csv file
Acl_F = mas_out.Acl_F
pl.DataFrame(Acl_F).write_csv(os.path.join(case_directory, "outputs", "closed_loop_A.csv"))

breakpoint() 
# ------------------------------------------------------------
# Simulate the EMT (before and after controller placement)
# ------------------------------------------------------------

# Run EMT simulation

#path_no_ctrl = os.path.join(case_directory, "outputs", "emt_no_control")
path_with_ctrl = os.path.join(case_directory, "outputs", "emt_with_control")

#os.makedirs(path_no_ctrl, exist_ok=True)
os.makedirs(path_with_ctrl, exist_ok=True)

#system.case_directory = path_no_ctrl
#main.run_emt(inputs=inputs, t_max=t_max, system=system)

# Run EMT simulation
system2 = wscc_9(case_directory=case_directory)
system2.case_directory = path_with_ctrl
# Apply any post initialization "updates" of system components
system2.apply("post_system_init", system2)


def output_feedback_control(t: float, x: np.ndarray, id: dict):

    F = mas_out.F[0]
    w0 = 1
    i_vsc_d0 = system2.gfmi_18a[0].lcl_filter.emt_init.i_vsc_d
    i_vsc_q0 = system2.gfmi_18a[0].lcl_filter.emt_init.i_vsc_q
    i_bus_d0 = system2.gfmi_18a[0].lcl_filter.emt_init.i_bus_d
    i_bus_q0 = system2.gfmi_18a[0].lcl_filter.emt_init.i_bus_q

    i_vsc_d, i_vsc_q, _ = abc2dq0(x[id['gfmi_18a_0']['i_vsc_a']], x[id['gfmi_18a_0']['i_vsc_b']], x[id['gfmi_18a_0']['i_vsc_c']], x[id['gfmi_18a_0']['angle']])
    i_bus_d, i_bus_q, _ = abc2dq0(x[id['gfmi_18a_0']['i_bus_a']], x[id['gfmi_18a_0']['i_bus_b']], x[id['gfmi_18a_0']['i_bus_c']], x[id['gfmi_18a_0']['angle']])

    delta_y = np.array([x[id['gfmi_18a_0']['w']] - w0, 
                        i_vsc_d - i_vsc_d0, 
                        i_vsc_q - i_vsc_q0, 
                        i_bus_d - i_bus_d0, 
                        i_bus_q - i_bus_q0])
    delta_u = F @ delta_y

    return delta_u[0]

inputs2 = {
    'gfmi_18a_0': {
        'p_ref': output_feedback_control,
        'v_ref': lambda t: smooth_step(t, step_time=0.1, initial_value=0.0, final_value=0.10, transient_width=5e-3),
        }
}

# Run EMT simulation
main.run_emt(inputs=inputs2, t_max=t_max, system=system2)
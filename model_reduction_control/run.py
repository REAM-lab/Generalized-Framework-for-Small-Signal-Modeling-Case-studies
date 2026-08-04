import os
import numpy as np

from sting import main, datasets
from sting.modules.model_order_reduction.reductions import BalancedTruncation, SingularPerturbation
from sting.utils.dynamical_systems import smooth_step
from pathlib import Path
from wscc_9 import wscc_9
from scipy.linalg import solve_continuous_are, eigvals, block_diag
import cvxpy as cp
from cmaspy.partial_state_feedback import single_agent_output_feedback, mas_output_feedback

# Location of all outputs
case_directory = os.path.join(Path(__file__).resolve().parent)

# Load the WSCC 9 bus system from the default datasets in STING
system = wscc_9(case_directory=case_directory)

# Apply any post initialization "updates" of system components
system.apply("post_system_init", system)

# Construct a small-signal model
system, fom = main.run_ssm(system=system, case_directory=case_directory)


# Create a reduced order model of all components in the zone labeled as "external".
# We will then connect this reduced order model to the zone labeled "study", which
# consists of a grid forming inverter (GFMI 18A) at bus 2.


# Vanilla balanced truncation removing the states that are hardest to control and observe.
# We will use the "singular perturbation" to eliminate states in order to enforce zero steady-state error
# at the expense of accuracy in higher-frequency dynamics. 
balanced_truncation = {
    "external": BalancedTruncation(r=30, gramian_c="lyapunov", gramian_o="lyapunov", method="truncate")
    }
# Construct a reduced-order model (ROM).
rom = main.run_model_reduction(ssm=fom, reductions=balanced_truncation)
print(np.max(np.linalg.eigvals(rom.model.A).real))

# COMPARE the dynamics of a step change to the voltage reference set point of the 
# grid forming inverter (GFLI 18A) at bus 2
inputs = {
    'gfmi_18a_0': {
        'v_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=0.10, transient_width=5e-3),
        }
}

t_max = 2.5 # Simulation length in seconds

# Simulate the full-order model
fom.output_directory = os.path.join(case_directory, "outputs", "full_order_model_simulation")
os.makedirs(fom.output_directory , exist_ok=True)
fom.simulate_ssm(t_max=t_max, inputs=inputs)

# Simulate the reduced-order model
rom.output_directory = os.path.join(case_directory, "outputs", "balanced_truncation_simulation")
os.makedirs(rom.output_directory , exist_ok=True)
rom.simulate_ssm(t_max=t_max, inputs=inputs)

# Design of the output feedback control
# Matrix of A of ROM
A_c = rom.model.A

# inputs = [p_ref, q_ref, v_ref, ...]
B_c = rom.model.B[:, 0:3] # take only p_ref, q_ref, v_ref of the GFM in the proposed project

# outputs = []
C_c = np.zeros((10, A_c.shape[0]))
C_c[0, 1] = 1 # w_pc
C_c[1, 7] = 1 # i_vsc_d
C_c[2, 8] = 1 # i_vsc_q
C_c[3, 9] = 1 # i_bus_d
C_c[4, 10] = 1 # i_bus_q 

D_c = np.zeros((C_c.shape[0], B_c.shape[1]))

Q = 10**5*np.eye(A_c.shape[0])

R = 10**3*np.eye(B_c.shape[1])

solve_settings = {'solver': cp.MOSEK,
                  'verbose': False}

# Solve CARE to obtain P
P = solve_continuous_are(A_c, B_c, Q, R)

# Use MAS output feedback
alpha_coef = 1000
beta_coef = 0
gamma_coef = 0
mas_out = mas_output_feedback(A_c, [B_c], [C_c], [D_c], [Q], [R], [P], alpha_coef, beta_coef, gamma_coef, **solve_settings)

print(mas_out.F[0])

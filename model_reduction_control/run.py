import os

from sting import main, datasets
from sting.utils.dynamical_systems import smooth_step
from pathlib import Path
from wscc_9 import wscc_9

# Location of all outputs
case_directory = os.path.join(Path(__file__).resolve().parent)

# Load the WSCC 9 bus system from the default datasets in STING
system = wscc_9(case_directory=case_directory)

# Apply any post initialization "updates" of system components
system.apply("post_system_init", system)

# Construct a small-signal model
system, ssm = main.run_ssm(system=system, case_directory=case_directory)

# Simulate dynamics of a step change to the power reference set points of the 
# grid forming inverter (GFLI 18A) at bus 2 
inputs = {
    'gfmi_18a_0': {
        'v_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=0.10, transient_width=5e-3),
        }
}

t_max = 5.5 # Simulation length in seconds
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
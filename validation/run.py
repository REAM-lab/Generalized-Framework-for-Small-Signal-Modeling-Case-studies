"""
Compare the small-signal model response to the EMT response for small changes in the reference inputs.
"""

# Import libraries
import os
from sting import datasets, main
from sting.utils.dynamical_systems import smooth_step
from sting.modules.simulation_emt.core import SimulationEMT
from pathlib import Path

# Location of all outputs
case_directory = os.path.join(Path(__file__).resolve().parent)

# Load the WSCC 9 bus system from the default datasets in STING
system = datasets.wscc_9(case_directory=case_directory)
# Apply any post initialization "updates" of system components
system.apply("post_system_init", system)

# Construct a small-signal model
system, ssm = main.run_ssm(system=system, case_directory=case_directory)

# Apply a small step to each input
#for generator in ['gfmi_18a_0', 'gfmi_18a_1', 'gfli_16a_0']:
for generator in ['gfmi_18a_0']:
    for reference in ['p_ref', 'q_ref']:
        #for amplitude in [0.01, 0.05, 0.1]:
        for amplitude in [0.1]:
            # Location of all outputs
            output_directory = os.path.join(case_directory, "simulations", f"{generator}-{reference}-{int(100*amplitude)}")
            ssm_dir = os.path.join(output_directory, "ssm")
            emt_dir = os.path.join(output_directory, "emt")
            os.makedirs(ssm_dir, exist_ok=True)
            os.makedirs(emt_dir, exist_ok=True)

            inputs = {
                generator: {
                    reference: lambda t, final_value=amplitude: smooth_step(t, step_time=0.5, initial_value=0.0, final_value=final_value, transient_width=5e-3),
                    }
            }
            t_max = 2.5 # Simulation length in seconds

            ssm.output_directory = ssm_dir
            ssm.simulate_ssm(t_max=t_max, inputs=inputs)

            # Run EMT simulation
            emt_sc = SimulationEMT(system=system, 
                                   power_flow_directory=os.path.join(case_directory, "outputs", "ac_power_flow"),
                                   output_directory=emt_dir)
            emt_sc.sim(t_max, inputs)
            break

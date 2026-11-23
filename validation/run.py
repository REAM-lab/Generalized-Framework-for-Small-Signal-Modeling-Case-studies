# This script runs step-response simulations for the WSCC 9 bus system using both small-signal and EMT models. 
# It applies small step changes to the active and reactive power references of two generators and saves the simulation results in organized directories.
# You will see the results in the "simulations" folder, with subfolders for each generator and reference combination, containing both small-signal and EMT simulation outputs.
# The idea is to use those results to, then plot the comparison between the small-signal and EMT simulations using compare_ssm_vs_emt, as shown in the "comparison_no_shunt.pdf" file.

# Import libraries
import os
from sting import main
from sting.utils.dynamical_systems import smooth_step
from sting.modules.simulation_emt.core import SimulationEMT
from pathlib import Path
from wscc_9 import wscc_9

# Location of all outputs
case_directory = os.path.join(Path(__file__).resolve().parent)

# Load the WSCC 9 bus system from the default datasets in STING
system = wscc_9(case_directory=case_directory)

# Apply any post initialization "updates" of system components
system.apply("post_system_init", system)

# Run AC power flow solution, build Component Connection matrices, and construct a small-signal model
system, ssm = main.run_ssm(system=system, case_directory=case_directory)

# Apply a small step to each input
for generator in ['gfmi_18a_0', 'gfli_16a_0']:
    for reference in ['p_ref', 'q_ref']:
        for amplitude in [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]:
            # Location of all outputs
            if amplitude < 0:
                file_name = f"{generator}-{reference}-minus-{int(100*abs(amplitude))}"
            else:
                file_name = f"{generator}-{reference}-plus-{int(100*abs(amplitude))}"
            output_directory = os.path.join(case_directory, "simulations", file_name)
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


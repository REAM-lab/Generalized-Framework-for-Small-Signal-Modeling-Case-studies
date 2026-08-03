"""
This code script compares the small-signal model response to the EMT response for small changes in the reference inputs.
"""

# Import libraries
import os
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt

# Location of all outputs
case_directory = os.path.join(Path(__file__).resolve().parent)

for generator in ['gfmi_18a_0']:
    for reference in ['q_ref']:
        #for amplitude in [0.01, 0.05, 0.1]:
        for amplitude in [0.1]:
            # Location of all outputs
            output_directory = os.path.join(case_directory, "simulations", f"{generator}-{reference}-{int(100*amplitude)}")

            # Read states of emt
            emt_states = pl.read_csv(os.path.join(output_directory, "emt", generator + ".csv"))
            # Take time, and psh
            time_emt = emt_states.select("time").to_numpy().flatten()
            psh_emt = emt_states.select("p_sh").to_numpy().flatten()
            qsh_emt = emt_states.select("q_sh").to_numpy().flatten()

            # Read states of ssm
            ssm_states = pl.read_csv(os.path.join(output_directory, "ssm", generator + ".csv"))
            # Take time, and psh
            time_ssm = ssm_states.select("time").to_numpy().flatten()
            v_sh_d = ssm_states.select("v_lcl_sh_d").to_numpy().flatten()
            v_sh_q = ssm_states.select("v_lcl_sh_q").to_numpy().flatten()
            i_bus_d = ssm_states.select("i_bus_d").to_numpy().flatten()
            i_bus_q = ssm_states.select("i_bus_q").to_numpy().flatten()
            # Compute power
            p_sh_ssm = v_sh_d*i_bus_d + v_sh_q*i_bus_q
            q_sh_ssm = v_sh_q*i_bus_d - v_sh_d*i_bus_q

# Plot: psh_emt vs psh_ssm

fig, ax = plt.subplots()
ax.plot(time_emt, qsh_emt, label="EMT")
ax.plot(time_ssm, q_sh_ssm, label="SSM")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Power [pu]")
ax.set_title(f"Comparison of EMT and SSM for {generator} with {reference} step of {amplitude}")
ax.legend() 

# Save the figure
plt.savefig(os.path.join(output_directory, f"comparison_{generator}_{reference}_{int(100*amplitude)}.png"))
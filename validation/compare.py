"""
This code script compares the small-signal model response to the EMT response for small changes in the reference inputs.
"""

# Import libraries
import os
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import scienceplots

# Location of all outputs
case_directory = os.path.join(Path(__file__).resolve().parent)

arbitrary_shunt_to_read ={'gfmi_18a_0': 'shunt_parallel_rc_4', 
                          'gfli_16a_0': 'shunt_parallel_rc_4',}

emt_data = {}
ssm_data = {}

for generator in ['gfmi_18a_0', 'gfli_16a_0']:
    for reference in ['p_ref']:
        #for amplitude in [0.01, 0.05, 0.1]:
        for amplitude in [0.1]:
            # Location of all outputs
            output_directory = os.path.join(case_directory, "simulations", f"{generator}-{reference}-{int(100*amplitude)}")

            # Read states of generator from EMT simulation
            emt_states = pl.read_csv(os.path.join(output_directory, "emt", generator + ".csv"))
            time_emt = emt_states.select("time").to_numpy().flatten()
            p_emt = emt_states.select("p_sh").to_numpy().flatten()
            q_emt = emt_states.select("q_sh").to_numpy().flatten()
            i_bus_d = emt_states.select("i_bus_d").to_numpy().flatten()
            i_bus_q = emt_states.select("i_bus_q").to_numpy().flatten()
            # Read states of shunt connected to generator from EMT simulation
            shunt_states = pl.read_csv(os.path.join(output_directory, "emt", arbitrary_shunt_to_read[generator] + ".csv"))
            v_sh_D = shunt_states.select("v_bus_D").to_numpy().flatten()
            v_sh_Q = shunt_states.select("v_bus_Q").to_numpy().flatten()
            # Voltage magnitude of shunt
            v_emt = (v_sh_D**2 + v_sh_Q**2)**0.5
            # Current magnitude of bus current
            i_emt = (i_bus_d**2 + i_bus_q**2)**0.5
            # Store data for plotting
            emt_data[(generator, reference, amplitude)] = {
                "time": time_emt,
                "p": p_emt,
                "q": q_emt,
                "v": v_emt,
                "i": i_emt,
            }

            # Read states of generator from SSM simulation
            ssm_states = pl.read_csv(os.path.join(output_directory, "ssm", generator + ".csv"))
            time_ssm = ssm_states.select("time").to_numpy().flatten()
            v_sh_d = ssm_states.select("v_lcl_sh_d").to_numpy().flatten()
            v_sh_q = ssm_states.select("v_lcl_sh_q").to_numpy().flatten()
            i_bus_d = ssm_states.select("i_bus_d").to_numpy().flatten()
            i_bus_q = ssm_states.select("i_bus_q").to_numpy().flatten()
            # Read states of shunt connected to generator from SSM simulation
            shunt_states = pl.read_csv(os.path.join(output_directory, "ssm", arbitrary_shunt_to_read[generator] + ".csv"))
            v_sh_D = shunt_states.select("v_bus_D").to_numpy().flatten()
            v_sh_Q = shunt_states.select("v_bus_Q").to_numpy().flatten()
            # Compute power
            p_ssm = v_sh_d*i_bus_d + v_sh_q*i_bus_q
            q_ssm = v_sh_q*i_bus_d - v_sh_d*i_bus_q
            # Voltage magnitude of shunt
            v_ssm = (v_sh_D**2 + v_sh_Q**2)**0.5
            # Current magnitude of bus current
            i_ssm = (i_bus_d**2 + i_bus_q**2)**0.5
            # Store data for plotting
            ssm_data[(generator, reference, amplitude)] = {
                "time": time_ssm,
                "p": p_ssm,
                "q": q_ssm,
                "v": v_ssm,
                "i": i_ssm,
            }


# Plotting settings
plt.style.use(['science','ieee'])
plt.rcParams['text.usetex'] = False
plt.rc('font',**{'family':'serif','serif':['Times'], 'size': 7})
plt.rcParams['axes.formatter.useoffset'] = False # this prevent scientific notation in y-axis

# Plot 3 x 2:   (0,0) = GFM active power, (0,1) = GFL active power, 
#               (1,0) = GFM bus voltage,  (1,1) = GFL bus voltage, 
#               (2,0) = GFM current magnitude, (2,1) = GFL current magnitude

# Create a 3x2 grid of subplots
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(4.3, 4), sharex=True)

# Plot active power for GFM and GFL
t_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["p"]
ax[0, 0].plot(t_emt, y_emt, label='EMT', color='blue')

t_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["p"]
ax[0, 0].plot(t_ssm, y_ssm, label='SSM', color='orange', linestyle='--')

# Add an inset plot for the GFM active power
ax_inset = ax[0, 0].inset_axes([0.46, 0.29, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_ssm, y_ssm, label='SSM', color='orange', linestyle='--')
ax_inset.plot(t_emt, y_emt, label='EMT', color='blue')
ax_inset.set_xlim(0.5, 1.5)
ax_inset.set_ylim(1.33, 1.35)
ax[0, 0].indicate_inset_zoom(ax_inset, lw=0.5)

# Plot active power for GFL
t_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["p"]
ax[0, 1].plot(t_emt, y_emt, label='EMT', color='blue')

t_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["p"]
ax[0, 1].plot(t_ssm, y_ssm, label='SSM', color='orange', linestyle='--')


# Add an inset plot for the GFL active power
ax_inset = ax[0, 1].inset_axes([0.46, 0.29, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_emt, y_emt, label='EMT', color='blue')
ax_inset.plot(t_ssm, y_ssm, label='SSM', color='orange', linestyle='--')
ax_inset.set_xlim(0.5, 1.5)
ax_inset.set_ylim(0.58, 0.62)
ax[0, 1].indicate_inset_zoom(ax_inset, lw=0.5)

# Plot voltage magnitude for the case when the GFM is excited
t_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["v"]
ax[1, 0].plot(t_emt, y_emt, label='EMT', color='blue')

t_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["v"]
ax[1, 0].plot(t_ssm, y_ssm, label='SSM', color='orange', linestyle='--')


# Add inset plot for the voltage magnitude when the GFM is excited
ax_inset = ax[1, 0].inset_axes([0.49, 0.42, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_emt, y_emt, label='EMT', color='blue')
ax_inset.plot(t_ssm, y_ssm, label='SSM', color='orange', linestyle='--')
ax_inset.set_xlim(0.5, 1.5)
ax_inset.set_ylim(0.979, 0.986)
ax[1, 0].indicate_inset_zoom(ax_inset, lw=0.5)

# Plot voltage magnitude for the case when the GFL is excited
t_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["v"]
ax[1, 1].plot(t_emt, y_emt, label='EMT', color='blue')
t_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["v"] 
ax[1, 1].plot(t_ssm, y_ssm, label='SSM', color='orange', linestyle='--')

# Add inset plot for the voltage magnitude when the GFL is excited
ax_inset = ax[1, 1].inset_axes([0.49, 0.42, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_emt, y_emt, label='EMT', color='blue')
ax_inset.plot(t_ssm, y_ssm, label='SSM', color='orange', linestyle='--')
ax_inset.set_xlim(0.5, 1.5)
ax_inset.set_ylim(0.989, 0.992)
ax[1, 1].indicate_inset_zoom(ax_inset, lw=0.5)


ax[2, 0].plot(emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"], emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["i"], label='EMT', color='blue')
ax[2, 0].plot(ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"], ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["i"], label='SSM', color='orange', linestyle='--')

ax[2, 1].plot(emt_data[('gfli_16a_0', 'p_ref', 0.1)]["time"], emt_data[('gfli_16a_0', 'p_ref', 0.1)]["i"], label='EMT', color='blue')
ax[2, 1].plot(ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["time"], ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["i"], label='SSM', color='orange', linestyle='--')

# Spacing between subplots
fig.subplots_adjust(wspace=0.20, hspace=0.10)

# Save figure
plt.savefig(os.path.join(case_directory, "simulations", "comparison_ssm_emt.png"), dpi=1000)
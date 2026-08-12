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

arbitrary_shunt_to_read ={'gfmi_18a_0': 'shunt_parallel_rc_2', 
                          'gfli_16a_0': 'shunt_parallel_rc_4',}

emt_data = {}
ssm_data = {}

for generator in ['gfmi_18a_0', 'gfli_16a_0']:
    for reference in ['p_ref']:
        #for amplitude in [0.01, 0.05, 0.1]:
        for amplitude in [0.1]:
            # Location of all outputs
            output_directory = os.path.join(case_directory, "simulations", f"{generator}-{reference}-plus-{int(100*amplitude)}")

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
ora = "#E57A73"
blu = "#4A9FE0"
gre = "#63B784" 
linewidth = 0.80

emt_color = ora
ssm_color = blu


# Plot 3 x 2:   (0,0) = GFM active power, (0,1) = GFL active power, 
#               (1,0) = GFM bus voltage,  (1,1) = GFL bus voltage, 
#               (2,0) = GFM current magnitude, (2,1) = GFL current magnitude

# Create a 3x2 grid of subplots
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(4.3, 4.3), sharex=True)

# Plot active power for GFM and GFL
t_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["p"]
ax[0, 0].plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)

t_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["p"]
ax[0, 0].plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)


# Add header to a), c), and e) subplots
ax[0, 0].text(0.5, 1.15, 'GFMI at bus 3', transform=ax[0, 0].transAxes,  va='top', ha='center')

# Add header to a), c), and e) subplots
ax[0, 1].text(0.5, 1.15, 'GFLI at bus 5', transform=ax[0, 1].transAxes,  va='top', ha='center')


# Add a) text
ax[0, 0].text(0.05, 0.95, 'a)', transform=ax[0, 0].transAxes,  va='top', ha='left')

# Add y label
ax[0, 0].set_ylabel("Active power to bus [pu]")

# Add an inset plot for the GFM active power
ax_inset = ax[0, 0].inset_axes([0.46, 0.29, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)
ax_inset.plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)
ax_inset.set_xlim(0.5, 1.2)
ax_inset.set_ylim(2.15, 2.235)
ax[0, 0].indicate_inset_zoom(ax_inset, lw=0.5)

# Plot active power for GFL
t_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["p"]
ax[0, 1].plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)

t_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["p"]
ax[0, 1].plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)

# Add b) text
ax[0, 1].text(0.05, 0.95, 'b)', transform=ax[0, 1].transAxes,  va='top', ha='left')

# Add an inset plot for the GFL active power
ax_inset = ax[0, 1].inset_axes([0.46, 0.29, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)
ax_inset.plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)
ax_inset.set_xlim(0.5, 1.2)
ax_inset.set_ylim(0.58, 0.615)
ax[0, 1].indicate_inset_zoom(ax_inset, lw=0.5)

# Plot voltage magnitude for the case when the GFM is excited
t_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["v"]
ax[1, 0].plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)

t_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["v"]
ax[1, 0].plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)

# Add y label
ax[1, 0].set_ylabel("Bus voltage magnitude [pu]")

# Add c) text
ax[1, 0].text(0.05, 0.95, 'c)', transform=ax[1, 0].transAxes,  va='top', ha='left')

# Add inset plot for the voltage magnitude when the GFM is excited
ax_inset = ax[1, 0].inset_axes([0.49, 0.42, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)
ax_inset.plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)
ax_inset.set_xlim(0.48, 1.2)
ax_inset.set_ylim(0.991, 0.993)
ax[1, 0].indicate_inset_zoom(ax_inset, lw=0.5)

# Plot voltage magnitude for the case when the GFL is excited
t_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["v"]
ax[1, 1].plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)
t_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["v"] 
ax[1, 1].plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)

# Add d) text
ax[1, 1].text(0.05, 0.95, 'd)', transform=ax[1, 1].transAxes,  va='top', ha='left')

# Add inset plot for the voltage magnitude when the GFL is excited
ax_inset = ax[1, 1].inset_axes([0.49, 0.42, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)
ax_inset.plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)
ax_inset.set_xlim(0.47, 1.2)
ax_inset.set_ylim(1.0, 1.004)
ax[1, 1].indicate_inset_zoom(ax_inset, lw=0.5)

# Plot current magnitude for the case when the GFM is excited
t_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfmi_18a_0', 'p_ref', 0.1)]["i"]
ax[2, 0].plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)
t_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfmi_18a_0', 'p_ref', 0.1)]["i"]
ax[2, 0].plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)

# Add y label
ax[2, 0].set_ylabel("Bus current magnitude [pu]")

# Add x label
ax[2, 0].set_xlabel("Time [s]")

# Add e) text
ax[2, 0].text(0.05, 0.95, 'e)', transform=ax[2, 0].transAxes,  va='top', ha='left')

# Add inset plot for the current magnitude when the GFM is excited
ax_inset = ax[2, 0].inset_axes([0.49, 0.30, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)
ax_inset.plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)
ax_inset.set_xlim(0.48, 1.2)
ax_inset.set_ylim(2.16, 2.19)
ax[2, 0].indicate_inset_zoom(ax_inset, lw=0.5)

# Plot current magnitude for the case when the GFL is excited
t_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_emt = emt_data[('gfli_16a_0', 'p_ref', 0.1)]["i"]
ax[2, 1].plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)
t_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["time"]
y_ssm = ssm_data[('gfli_16a_0', 'p_ref', 0.1)]["i"]
ax[2, 1].plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)

# Add f) text
ax[2, 1].text(0.05, 0.95, 'f)', transform=ax[2, 1].transAxes,  va='top', ha='left')

# Add x label
ax[2, 1].set_xlabel("Time [s]")

# Add inset plot for the current magnitude when the GFL is excited
ax_inset = ax[2, 1].inset_axes([0.49, 0.30, 0.45, 0.47])  # x0 position, y0 position, width, height
ax_inset.plot(t_emt, y_emt, label='EMT', color=emt_color, linewidth=linewidth)
ax_inset.plot(t_ssm, y_ssm, label='SSM', color=ssm_color, linestyle='--', linewidth=linewidth)
ax_inset.set_xlim(0.48, 1.2)
ax_inset.set_ylim(0.62, 0.66)
ax[2, 1].indicate_inset_zoom(ax_inset, lw=0.5)

# Add legend ( - EMT, -- SSM) below the last subplot
ax[2, 1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.25), ncol=2, frameon=False, fontsize=7)

# Spacing between subplots
fig.subplots_adjust(wspace=0.20, hspace=0.17)

# Save figure
plt.savefig(os.path.join(case_directory, "simulations", "comparison_ssm_emt.pdf"), dpi=1000)
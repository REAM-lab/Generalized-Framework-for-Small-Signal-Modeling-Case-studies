# Import libraries
import os
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

# Location of all outputs
case_directory = os.path.join(Path(__file__).resolve().parent)
output_directory = os.path.join(case_directory, "outputs")

# Read states of EMT simulation
emt_no_control = pl.read_csv(os.path.join(output_directory, "simulation_emt_no_control", "gfmi_18a_0.csv"))

# Read states of SSM simulation
ssm = pl.read_csv(os.path.join(output_directory, "small_signal_model", "gfmi_18a_0.csv"))
ssm = ssm.with_columns(
                    (pl.col("v_lcl_sh_d") * pl.col("i_bus_d") + pl.col("v_lcl_sh_q") * pl.col("i_bus_q")).alias("p_sh"),
                    )


# Read states of balanced truncation simulation
mor = pl.read_csv(os.path.join(output_directory, "model_order_reduction", "gfmi_18a_0.csv"))
mor = mor.with_columns(
                    (pl.col("v_lcl_sh_d") * pl.col("i_bus_d") + pl.col("v_lcl_sh_q") * pl.col("i_bus_q")).alias("p_sh"),
                    )

# Read matrix A of the small-signal model
ssm_A = pl.read_csv(os.path.join(output_directory, "small_signal_model", "A.csv"))
ssm_A = ssm_A[0:, 1:].to_numpy()

# Read matrix A of the closed-loop system with output feedback control
rom_Acl = pl.read_csv(os.path.join(output_directory, "output_feedback_control", "closed_loop_A.csv"))
rom_Acl = rom_Acl[0:, 0:].to_numpy()

#control_A = pl.read_csv(os.path.join(output_directory, "model_order_reduction", "A.csv"))

# Read states of the EMT with control implemented
#emt_with_control = pl.read_csv(os.path.join(output_directory, "simulation_emt_with_control", "gfmi_18a_0.csv"))


# Make a plot 1 x 4
# Plot 1: EMT simulation without control. Active power injected by the GFMI (proposed project)
# Plot 2: SSM and ROM simulation without control. Active power injected by the GFMI (proposed project)
# Plot 3: Eigenvalues of the SSM with and without control.
# Plot 4: EMT simulation with control. Active power injected by the GFMI (proposed project)

# Plotting settings
plt.style.use(['science','ieee'])
plt.rcParams['text.usetex'] = False
plt.rc('font',**{'family':'serif','serif':['Times'], 'size': 7})
plt.rcParams['axes.formatter.useoffset'] = False # this prevent scientific notation in y-axis
ora = "#D55E00"
blu = "#0072B2"
gre = "#009E73" 
linewidth = 0.80

# Create a plot
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 2.0), dpi=1000)

# Share y-axis between the first 3
ax[1].sharey(ax[0])
ax[3].sharey(ax[0])

# Plot 1: EMT simulation without control. Active power injected by the GFMI (proposed project)
ax[0].plot(emt_no_control['time'], emt_no_control['p_sh'], color=ora, linewidth=linewidth)
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Active power [MW]")
ax[0].set_xlim([0, 1])
ax[0].set_title("a)")

# Plot 2: SSM and ROM simulation without control. Active power injected by the GFMI (proposed project)
ax[1].plot(ssm['time'], ssm['p_sh'], color=blu, linewidth=linewidth, label='SSM')
ax[1].plot(mor['time'], mor['p_sh'], color=gre, linewidth=linewidth, label='ROM')
ax[1].set_xlabel("Time [s]")
ax[1].set_xlim([0, 1])
ax[1].set_title("b)")

# Plot 3: Eigenvalues of the SSM with and without control.
eigenvalues_ssm = np.linalg.eigvals(ssm_A)
eigenvalues_rom_cl = np.linalg.eigvals(rom_Acl)
ax[2].scatter(eigenvalues_ssm.real, eigenvalues_ssm.imag, color='red', s=10, marker='x', label='Open-loop')
ax[2].scatter(eigenvalues_rom_cl.real, eigenvalues_rom_cl.imag, color='blue', s=10, marker='s', facecolors='none', label='Closed-loop', linewidths=0.8)

ax[2].set_yscale("symlog")
ax[2].set_xscale("symlog")
ax[2].set_title("c)")

# Spacing between subplots
fig.subplots_adjust(wspace=0.20, hspace=0.17)

# Save figure
plt.savefig(os.path.join(case_directory, "pipeline_plot.pdf"), dpi=1000)


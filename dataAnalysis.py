import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data

C60_00_df = pd.read_csv('outputData/C60_00_mean_for_graph.csv')
C60_05_df = pd.read_csv('outputData/C60_05_mean_for_graph.csv')
C60_10_df = pd.read_csv('outputData/C60_10_mean_for_graph.csv')

# Extract columns

C60_00_time = C60_00_df['hours']
C60_00_eqe  = C60_00_df['EQE']

C60_05_time = C60_05_df['hours']
C60_05_eqe  = C60_05_df['EQE']

C60_10_time = C60_10_df['hours']
C60_10_eqe  = C60_10_df['EQE']

# Colors for each dataset

colors = {
    "00": "blue",
    "05": "red",
    "10": "gold"
}

# Fit polynomial curves

mask = np.ones(len(C60_05_time), dtype=bool)
mask[4] = False

APPROX_ORDER = 3
p_00 = np.poly1d(np.polyfit(C60_00_time, C60_00_eqe, APPROX_ORDER))
p_05 = np.poly1d(np.polyfit(C60_05_time[mask], C60_05_eqe[mask], APPROX_ORDER))
p_10 = np.poly1d(np.polyfit(C60_10_time, C60_10_eqe, APPROX_ORDER))

xp = np.linspace(min(C60_00_time), max(C60_00_time), 200)

# Create ONE plot with ALL elements + a usable legend

plt.figure(figsize=(10, 7))

# Best-fit lines (legend labels apply here)
plt.plot(xp, p_00(xp), color=colors["00"], linewidth=2, label="C60 (0 cycles)")
plt.plot(xp, p_05(xp), color=colors["05"], linewidth=2, label="C60 (5 cycles)")
plt.plot(xp, p_10(xp), color=colors["10"], linewidth=2, label="C60 (10 cycles)")

# Scatter points (no labels → avoids duplicate legend entries)
plt.scatter(C60_00_time, C60_00_eqe, color=colors["00"], s=40)
plt.scatter(C60_05_time, C60_05_eqe, color=colors["05"], s=40)
plt.scatter(C60_10_time, C60_10_eqe, color=colors["10"], s=40)

# Titles and labels
plt.title("C60 Device EQE Degradation – Best Fit Curves", fontsize=16)
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Mean EQE (%)", fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)

# Add the key (legend) ON the graph

plt.legend(
    title="Cycle Key",
    title_fontsize=13,
    fontsize=12,
    loc='upper right'  # legend on graph
)

# Save the final plot
plt.savefig('outputData/C60_cycles_best_fit.png', dpi=300)

plt.show()
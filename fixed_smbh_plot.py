import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PUBLICATION-QUALITY SETUP
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",  # STIX fonts look like Times for math
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# ============================================================
# PREPARE DATA
# ============================================================

# Dynamically measured data
x_meas = np.array([float(m[0][0]) for m in dynamicallymeasured], dtype=np.float64)
y_meas = np.array([float(m[0][1]) for m in dynamicallymeasured], dtype=np.float64)
xerr = np.array([float(m[1][0]) for m in dynamicallymeasured], dtype=np.float64)
yerr = np.array([float(m[1][1]) for m in dynamicallymeasured], dtype=np.float64)

# Broad-line AGN data
combined_agn = broadLINEAGNdata 
x_agn = np.array([float(d[0][0]) for d in combined_agn], dtype=np.float64)
y_agn = np.array([float(d[0][1]) for d in combined_agn], dtype=np.float64)

# Model lines
x_line = np.linspace(5, 15, 100)  # More points for smoother lines
y_line = 8.7 + 1.3 * (x_line - 11)

# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(6, 4))

# Professional colorblind-friendly colors
color_meas = '#0173B2'   # Blue
color_agn = '#DE8F05'    # Orange  
color_model = '#029E73'  # Green

# Plot model lines (plot first so they're in background)
ax.plot(x_line, y_line, 
        color=color_model, 
        linewidth=2.0, 
        linestyle='-',
        label='Best fit', 
        zorder=2)

ax.fill_between(x_line, y_line - 0.6, y_line + 0.6,
                alpha=0.2, 
                color=color_model,
                linewidth=0,
                label=r'$\pm 0.6$ dex',
                zorder=1)

# Plot AGN data (smaller points, lower zorder)
ax.scatter(x_agn, y_agn, 
          s=15, 
          color=color_agn, 
          alpha=0.6,
          marker='o',
          linewidths=0.5,
          edgecolors='none',
          label='Broad-line AGN',
          zorder=3)

# Plot dynamically measured data (on top)
ax.errorbar(x_meas, y_meas, 
           xerr=xerr, 
           yerr=yerr, 
           fmt='o', 
           ms=5,
           color=color_meas,
           ecolor=color_meas,
           elinewidth=1.0,
           capsize=2,
           capthick=1.0,
           alpha=0.8,
           label='Dynamical measurements',
           zorder=4)

# ============================================================
# FORMATTING
# ============================================================

# Axis limits
ax.set_xlim(8, 12)
ax.set_ylim(4, 11)

# Labels with proper LaTeX
ax.set_xlabel(r'$\log_{10}(M_{\star}/M_{\odot})$')
ax.set_ylabel(r'$\log_{10}(M_{\rm BH}/M_{\odot})$')

# Grid
ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.8, color='gray', zorder=0)
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5, color='gray', zorder=0)

# Enable minor ticks
ax.minorticks_on()

# Legend with better styling
legend = ax.legend(loc='lower right', 
                  framealpha=0.95, 
                  edgecolor='black',
                  fancybox=False,
                  frameon=True)

# Text box with parameters (upper left)
textstr = r'$a = 9.0$' + '\n' + r'$b = 1.4$' + '\n' + r'$\sigma = 0.5$'
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.0)
ax.text(0.05, 0.95, textstr, 
        transform=ax.transAxes, 
        fontsize=11,
        verticalalignment='top',
        bbox=props)

# Tight layout
plt.tight_layout()

# ============================================================
# SAVE FIGURE
# ============================================================

# Save in multiple formats
plt.savefig('smbh_stellar_mass_relation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('smbh_stellar_mass_relation.png', dpi=300, bbox_inches='tight')

plt.show()

print("✓ Figure saved as:")
print("  - smbh_stellar_mass_relation.pdf")
print("  - smbh_stellar_mass_relation.png")

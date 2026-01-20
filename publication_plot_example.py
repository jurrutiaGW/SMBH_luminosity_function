import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# ============================================================
# PUBLICATION-QUALITY MATPLOTLIB SETUP
# ============================================================

# Set the global font to Times/serif fonts (choose ONE approach)

# OPTION 1: Use LaTeX rendering (best quality, but requires LaTeX installation)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
})

# OPTION 2: If you don't have LaTeX, use this instead:
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "DejaVu Serif"],
#     "mathtext.fontset": "stix",  # STIX fonts look like Times for math
#     "font.size": 11,
#     "axes.labelsize": 12,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "legend.fontsize": 10,
# })

# Additional publication settings
plt.rcParams.update({
    "figure.dpi": 150,           # High DPI for screen
    "savefig.dpi": 300,          # Publication quality when saving
    "savefig.bbox": "tight",     # Tight bounding box
    "savefig.pad_inches": 0.05,  # Small padding
    "lines.linewidth": 1.5,      # Thicker lines
    "axes.linewidth": 1.0,       # Thicker axes
    "grid.linewidth": 0.5,       # Thinner grid
    "xtick.major.width": 1.0,    # Tick widths
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.direction": "in",     # Ticks point inward
    "ytick.direction": "in",
    "xtick.top": True,            # Ticks on all sides
    "ytick.right": True,
})

# ============================================================
# CREATE YOUR PLOT
# ============================================================

# Use single-column width (3.5") or double-column width (7.0")
# For ApJ/A&A: single column ~3.5", double column ~7.0"
fig, ax = plt.subplots(figsize=(6, 4))  # or (3.5, 2.625) for single column

# Professional color scheme (colorblind-friendly)
colors = ['#0173B2', '#DE8F05', '#029E73']  # blue, orange, green

# Plot with better styling
ax.plot(xvalues, yvalues, 
        label=r'$z=0$', 
        color=colors[0], 
        linewidth=2.0,
        linestyle='-')

ax.plot(xvalues, yvalues2, 
        label=r'$z=1$',
        color=colors[1], 
        linewidth=2.0,
        linestyle='--')

ax.plot(xvalues, yvalues3, 
        label=r'$z=3$', 
        color=colors[2], 
        linewidth=2.0,
        linestyle=':')

# Set scales
ax.set_xscale('linear')
ax.set_yscale('log')

# Labels with proper LaTeX formatting
ax.set_xlabel(r'$\log_{10}(\mathcal{M}_{\rm c}/M_{\odot})$')
ax.set_ylabel(r'$\mathrm{d}R_{\rm BH}/\mathrm{d}\ln\mathcal{M}_{\rm c}$ [Mpc$^{-3}$ yr$^{-1}$]')

# Grid styling
ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5, color='gray')

# Legend with better positioning
ax.legend(loc='lower left', 
          framealpha=0.95, 
          edgecolor='black',
          fancybox=False,      # Square corners
          shadow=False)

# Set axis limits if needed
# ax.set_xlim(7, 11)
# ax.set_ylim(1e-10, 1e-2)

# Tight layout
plt.tight_layout()

# ============================================================
# SAVE THE FIGURE
# ============================================================

# Save in multiple formats
# PDF: Best for LaTeX papers
plt.savefig('merger_rate.pdf', format='pdf', dpi=300, bbox_inches='tight')

# PNG: Good for presentations/slides
plt.savefig('merger_rate.png', format='png', dpi=300, bbox_inches='tight')

# EPS: Required by some journals
# plt.savefig('merger_rate.eps', format='eps', dpi=300, bbox_inches='tight')

plt.show()


# ============================================================
# ALTERNATIVE: QUICK SETUP FUNCTION
# ============================================================

def setup_publication_plot():
    """
    Quick function to set up publication-quality plots.
    Call this at the beginning of your notebook.
    """
    plt.rcParams.update({
        "text.usetex": False,  # Set to True if you have LaTeX
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

# Usage:
# setup_publication_plot()
# # Then create your plots as normal

"""
Plotting Utilities for EDA Framework
------------------------------------
Provides standardized plotting configuration and chart builders for visualization.
"""

import seaborn as sns
import matplotlib.pyplot as plt

def setup_plot_style() -> None:
    """Sets up a modern, high-quality, professional plotting aesthetic."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 120,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
    })

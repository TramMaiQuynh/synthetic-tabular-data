"""
Visual Overlay Generator
------------------------
Generates visual evaluation charts for real vs synthetic datasets:
1. Marginal distribution overlay grid (KDE/histogram for continuous, bar chart for categorical).
2. Correlation heatmap comparisons (Real, Synthetic, and Absolute Difference).
3. DCR (Distance to Closest Record) distribution curves.
Saves all plots to artifacts/<dataset_name>/evaluation/plots/.
"""

import os
import logging
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


class VisualOverlayGenerator:
    """Generates and saves visual quality assurance charts."""
    
    def __init__(
        self,
        continuous_cols: List[str],
        categorical_cols: List[str],
        plots_dir: str,
    ) -> None:
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.plots_dir = plots_dir
        
        # Create output directories
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def plot_distributions(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> str:
        """
        Plot marginal distribution comparisons for all features in a grid.
        
        Returns the path to the saved grid image.
        """
        all_cols = self.continuous_cols + self.categorical_cols
        n_cols = len(all_cols)
        if n_cols == 0:
            return ""
            
        # Grid parameters: 3 columns wide
        cols_per_row = 3
        rows = int(np.ceil(n_cols / cols_per_row))
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, 5 * rows))
        axes = np.atleast_1d(axes).flatten()
        
        try:
            for idx, col in enumerate(all_cols):
                ax = axes[idx]
                
                if col not in real_df.columns or col not in synth_df.columns:
                    ax.text(0.5, 0.5, f"Column {col} not found", ha="center", va="center")
                    continue
                    
                r_series = real_df[col].dropna()
                s_series = synth_df[col].dropna()
                
                if col in self.continuous_cols:
                    # Continuous: Overlay Histograms & KDEs
                    sns.histplot(
                        r_series, kde=True, stat="density", color="blue", label="Real",
                        alpha=0.4, ax=ax, linewidth=0,
                    )
                    sns.histplot(
                        s_series, kde=True, stat="density", color="red", label="Synthetic",
                        alpha=0.4, ax=ax, linewidth=0,
                    )
                    ax.set_title(f"Continuous: {col}")
                    ax.legend()
                else:
                    # Categorical: Grouped bar charts of frequencies
                    r_counts = r_series.astype(str).value_counts(normalize=True)
                    s_counts = s_series.astype(str).value_counts(normalize=True)
                    
                    # Combine
                    union_idx = r_counts.index.union(s_counts.index)
                    freq_df = pd.DataFrame({
                        "Real": [r_counts.get(cat, 0.0) for cat in union_idx],
                        "Synthetic": [s_counts.get(cat, 0.0) for cat in union_idx],
                    }, index=union_idx)
                    
                    # Limit categories plotted if too high cardinality
                    if len(freq_df) > 10:
                        freq_df = freq_df.sort_values(by="Real", ascending=False).head(10)
                        ax.set_title(f"Categorical: {col} (Top 10)")
                    else:
                        ax.set_title(f"Categorical: {col}")
                        
                    freq_df.plot(kind="bar", ax=ax, color=["blue", "red"], alpha=0.6, rot=30)
                    ax.set_ylabel("Density")
                    # Wrap labels — set both ticks and labels to avoid deprecation warning
                    current_ticks = ax.get_xticks()
                    ax.set_xticks(current_ticks)
                    labels = [label.get_text()[:10] for label in ax.get_xticklabels()]
                    ax.set_xticklabels(labels, ha="right", rotation=30)
                    ax.legend()
                    
            # Hide any unused subplots
            for idx in range(n_cols, len(axes)):
                fig.delaxes(axes[idx])
                
            plt.tight_layout()
            os.makedirs(self.plots_dir, exist_ok=True)
            grid_path = os.path.join(self.plots_dir, "distributions_grid.png")
            plt.savefig(grid_path, dpi=120)
        finally:
            plt.close(fig)
        
        logger.info("Saved distribution overlay grid to %s", grid_path)
        return grid_path

    def plot_correlation_difference(self, real_corr: pd.DataFrame, synth_corr: pd.DataFrame) -> str:
        """
        Plot real correlation, synthetic correlation, and absolute difference side-by-side.
        
        Returns the path to the saved correlation image.
        """
        if real_corr.empty or synth_corr.empty:
            return ""
            
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        
        try:
            # Absolute difference
            diff_corr = (real_corr - synth_corr).abs()
            
            # Color palettes
            cmap_corr = sns.diverging_palette(220, 20, as_cmap=True)
            cmap_diff = sns.light_palette("purple", as_cmap=True)
            
            # Subplot 1: Real Correlation
            sns.heatmap(
                real_corr, ax=axes[0], cmap=cmap_corr, vmin=-1, vmax=1, center=0,
                square=True, cbar_kws={"shrink": 0.8}, annot=False,
            )
            axes[0].set_title("Real Correlation Matrix")
            
            # Subplot 2: Synthetic Correlation
            sns.heatmap(
                synth_corr, ax=axes[1], cmap=cmap_corr, vmin=-1, vmax=1, center=0,
                square=True, cbar_kws={"shrink": 0.8}, annot=False,
            )
            axes[1].set_title("Synthetic Correlation Matrix")
            
            # Subplot 3: Absolute Difference
            sns.heatmap(
                diff_corr, ax=axes[2], cmap=cmap_diff, vmin=0, vmax=1,
                square=True, cbar_kws={"shrink": 0.8}, annot=False,
            )
            axes[2].set_title("Absolute Correlation Difference")
            
            for ax in axes:
                # Wrap tick labels — set ticks explicitly before labels to avoid deprecation
                ax.set_xticks(ax.get_xticks())
                ax.set_yticks(ax.get_yticks())
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
                
            plt.tight_layout()
            os.makedirs(self.plots_dir, exist_ok=True)
            corr_path = os.path.join(self.plots_dir, "correlation_comparison.png")
            plt.savefig(corr_path, dpi=120)
        finally:
            plt.close(fig)
        
        logger.info("Saved correlation difference matrix to %s", corr_path)
        return corr_path

    def plot_dcr_distribution(self, dcr_vals: np.ndarray, leakage_threshold: float = 0.01) -> str:
        """
        Plot the histogram & KDE curve of Distance to Closest Record (DCR) values.
        
        Returns the path to the saved DCR plot.
        """
        if len(dcr_vals) == 0:
            return ""
            
        fig = plt.figure(figsize=(8, 5))
        try:
            sns.histplot(dcr_vals, kde=True, color="purple", stat="density", alpha=0.5, linewidth=0)
            
            # Draw threshold marker for warning of leakage (DCR < leakage_threshold)
            plt.axvline(x=leakage_threshold, color="red", linestyle="--", alpha=0.8, label=f"Leakage Threshold ({leakage_threshold:.4f})")
            
            plt.title("Distribution of Distance to Closest Record (DCR)")
            plt.xlabel("L2 Distance in Normalized Space")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            os.makedirs(self.plots_dir, exist_ok=True)
            dcr_path = os.path.join(self.plots_dir, "dcr_distribution.png")
            plt.savefig(dcr_path, dpi=120)
        finally:
            plt.close(fig)
        
        logger.info("Saved DCR distribution plot to %s", dcr_path)
        return dcr_path

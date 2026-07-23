"""
Statistical Tests Module
------------------------
Thực hiện các kiểm định thống kê phi tham số để so sánh hiệu suất của 
nhiều mô hình sinh dữ liệu (CTGAN, CTVAE, Diffusion) trên nhiều Folds/Datasets.

- Kiểm định Friedman: Đánh giá xem có sự khác biệt có ý nghĩa thống kê
  giữa các mô hình hay không (H0: Các mô hình có hiệu suất như nhau).
- Kiểm định Nemenyi (Post-hoc): Nếu Friedman bác bỏ H0, Nemenyi sẽ so sánh
  từng cặp mô hình để xem cụ thể mô hình nào vượt trội hơn mô hình nào.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

__all__ = [
    "run_statistical_evaluation",
    "plot_nemenyi_heatmap",
    "compute_critical_difference",
    "plot_cd_diagram",
]


def run_statistical_evaluation(
    df_metrics: pd.DataFrame, 
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Thực hiện chuỗi kiểm định Friedman và Nemenyi.

    Args:
        df_metrics: DataFrame chứa kết quả đo lường. 
                    - Index (Rows): Các khối (Folds hoặc Datasets).
                    - Columns: Tên các mô hình (vd: 'CTGAN', 'CTVAE', 'Diffusion').
                    - Values: Điểm số metric (vd: Fidelity score, F1 score).
        alpha: Mức ý nghĩa thống kê (thường là 0.05).

    Returns:
        Một Dictionary chứa thống kê Friedman, p-value, và ma trận p-value của Nemenyi.
    """
    if df_metrics.shape[1] < 3:
        raise ValueError("Kiểm định Friedman yêu cầu ít nhất 3 mô hình (columns) để so sánh.")

    if df_metrics.isnull().values.any():
        logger.warning("Dữ liệu chứa giá trị NaN. Hệ thống sẽ loại bỏ các dòng bị thiếu (Dropna).")
        df_metrics = df_metrics.dropna()

    if df_metrics.shape[0] < 3:
        raise ValueError(
            f"Kiểm định Friedman yêu cầu ít nhất 3 blocks/folds (rows) hợp lệ, "
            f"nhưng hiện tại chỉ có {df_metrics.shape[0]} dòng."
        )

    # 1. Kiểm định Friedman
    # Chuyển đổi DataFrame thành list các mảng 1D (mỗi mảng là kết quả của 1 mô hình)
    model_scores = [df_metrics[col].values for col in df_metrics.columns]
    
    stat, p_value = friedmanchisquare(*model_scores)
    
    if np.isnan(p_value):
        p_value = 1.0

    is_significant = bool(p_value < alpha)
    
    logger.info("--- Friedman Test Results ---")
    logger.info("Test Statistic: %.4f", stat)
    logger.info("p-value: %.4e", p_value)
    
    result = {
        "friedman_stat": float(stat),
        "friedman_p_value": float(p_value),
        "alpha": alpha,
        "is_significant": is_significant,
        "nemenyi_p_values": None
    }

    if is_significant:
        logger.info(
            "Kết luận: CÓ sự khác biệt ý nghĩa thống kê giữa các mô hình (p < %.2f). "
            "Tiến hành kiểm định Nemenyi Post-hoc...", alpha
        )
        
        # 2. Kiểm định Nemenyi (sử dụng thư viện scikit-posthocs)
        # Nemenyi yêu cầu input data ở dạng ma trận hoặc block.
        # Ở đây ta truyền nguyên DataFrame, hàm sẽ tự động xếp hạng từng dòng (Fold) 
        # và so sánh các cột (Mô hình).
        nemenyi_p_matrix = sp.posthoc_nemenyi_friedman(df_metrics)
        result["nemenyi_p_values"] = nemenyi_p_matrix
        
        logger.info("Ma trận p-value Nemenyi:\n%s", nemenyi_p_matrix.round(4))
    else:
        logger.info(
            "Kết luận: KHÔNG có sự khác biệt ý nghĩa thống kê giữa các mô hình (p >= %.2f). "
            "Dừng lại, không chạy Nemenyi.", alpha
        )

    return result


def plot_nemenyi_heatmap(
    nemenyi_p_matrix: Optional[pd.DataFrame], 
    alpha: float = 0.05, 
    save_path: Optional[str] = None,
    metric_name: str = "Metric"
) -> Optional[plt.Figure]:
    """
    Vẽ Heatmap cho ma trận p-value của kiểm định Nemenyi. 
    Làm nổi bật các cặp có sự khác biệt có ý nghĩa thống kê (p < alpha).

    Args:
        nemenyi_p_matrix: DataFrame ma trận p-value trả về từ run_statistical_evaluation.
        alpha: Mức ý nghĩa thống kê để kẻ ranh giới màu.
        save_path: Đường dẫn để lưu file ảnh (nếu cần).
        metric_name: Tên độ đo để hiển thị trên Title.

    Returns:
        Đối tượng Figure của matplotlib, hoặc None nếu input is None.
    """
    if nemenyi_p_matrix is None:
        logger.error("Không có ma trận Nemenyi (Friedman test có thể chưa qua ngưỡng alpha).")
        return None

    # Tạo mask để hiển thị ma trận tam giác dưới (bỏ qua các cặp trùng lặp và đường chéo)
    mask = np.triu(np.ones_like(nemenyi_p_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # Custom color map: Xanh (có ý nghĩa, p < alpha: h_neg=130), Đỏ/Xám (không có ý nghĩa, p >= alpha: h_pos=10)
        cmap = sns.diverging_palette(130, 10, as_cmap=True)

        sns.heatmap(
            nemenyi_p_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".3f", 
            cmap=cmap, 
            vmax=1.0, # Dải p-value đầy đủ trong [0, 1] không bị nén màu tại 0.1
            vmin=0.0,
            center=alpha, # Điểm giao màu tại chính mức alpha (p < alpha => Green, p > alpha => Red/Gray)
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .7, "label": "p-value"},
            ax=ax
        )
        
        ax.set_title(
            f"Nemenyi Post-hoc p-values for {metric_name}\n"
            f"(Green shades indicate statistical significance, p < {alpha})", 
            pad=20
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Đã lưu biểu đồ Nemenyi tại: %s", save_path)
    except Exception as exc:
        logger.error("Lỗi khi vẽ Heatmap Nemenyi: %s", exc)
        plt.close(fig)
        return None
        
    return fig


def compute_critical_difference(n_models: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Compute Nemenyi Critical Difference (CD) according to Demšar (2006).

    Formula:
        CD = q_alpha * sqrt( (k * (k + 1)) / (6 * n) )
    where q_alpha is the Studentized range statistic divided by sqrt(2).
    """
    if n_models < 2 or n_blocks < 1:
        return 0.0

    try:
        from scipy.stats import studentized_range
        q_alpha = studentized_range.ppf(1.0 - alpha, n_models, np.inf) / np.sqrt(2.0)
    except (ImportError, AttributeError):
        # Fallback table for alpha=0.05, k=2..10 (Demšar 2006, Table 5)
        q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        q_alpha = q_table.get(n_models, 3.0)

    cd = q_alpha * np.sqrt((n_models * (n_models + 1)) / (6.0 * n_blocks))
    return float(cd)


def plot_cd_diagram(
    df_metrics: pd.DataFrame,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
    metric_name: str = "Metric",
    higher_is_better: bool = True,
) -> Optional[plt.Figure]:
    """
    Vẽ Critical Difference (CD) Diagram theo chuẩn Demšar (2006).

    Hiển thị Rank trung bình của các mô hình trên trục ngang [1, K].
    Các mô hình không có sự khác biệt có ý nghĩa thống kê (|Rank_i - Rank_j| <= CD)
    được kết nối bằng một thanh ngang màu đỏ.

    Args:
        df_metrics: DataFrame với Rows = Datasets/Folds, Columns = Models.
        alpha: Mức ý nghĩa thống kê (mặc định 0.05).
        save_path: Đường dẫn lưu file ảnh (nếu có).
        metric_name: Tên độ đo hiển thị trên tiêu đề.
        higher_is_better: True nếu điểm số cao hơn là tốt hơn (mặc định True).

    Returns:
        Đối tượng Figure của Matplotlib, hoặc None nếu dữ liệu không hợp lệ.
    """
    if df_metrics is None or df_metrics.shape[1] < 2 or df_metrics.shape[0] < 1:
        logger.error("Dữ liệu không đủ để vẽ CD Diagram.")
        return None

    df_clean = df_metrics.dropna()
    n_blocks, n_models = df_clean.shape
    if n_models < 2 or n_blocks < 1:
        return None

    # Calculate average ranks (1 = best rank)
    ranks = df_clean.rank(axis=1, ascending=not higher_is_better)
    mean_ranks = ranks.mean(axis=0).sort_values()

    cd = compute_critical_difference(n_models, n_blocks, alpha=alpha)

    fig, ax = plt.subplots(figsize=(10, 4))
    try:
        # Plot main axis line
        ax.plot([1, n_models], [0, 0], color="black", lw=2)

        # Plot rank ticks
        for r in range(1, n_models + 1):
            ax.plot([r, r], [-0.05, 0.05], color="black", lw=1.5)
            ax.text(r, 0.12, str(r), ha="center", va="bottom", fontsize=11, fontweight="bold")

        # Layout models with alternating labels
        y_step = 0.2
        for idx, (m_name, m_rank) in enumerate(mean_ranks.items()):
            y_pos = -0.3 - (idx * y_step)

            ax.plot([m_rank, m_rank], [0, y_pos], color="gray", linestyle="--", lw=1)
            ax.plot(m_rank, y_pos, marker="o", color="#1f77b4", markersize=8)

            text_str = f"{m_name} ({m_rank:.2f})"
            ax.text(
                m_rank + (0.05 if idx % 2 == 0 else -0.05),
                y_pos,
                text_str,
                ha="left" if idx % 2 == 0 else "right",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        # Draw CD bars for non-significant pairs
        sorted_ranks = list(mean_ranks.values)
        cd_bar_y = -0.3 - (n_models * y_step) - 0.2
        for i in range(len(sorted_ranks)):
            for j in range(i + 1, len(sorted_ranks)):
                if abs(sorted_ranks[i] - sorted_ranks[j]) <= cd:
                    ax.plot(
                        [sorted_ranks[i], sorted_ranks[j]],
                        [cd_bar_y, cd_bar_y],
                        color="#d62728",
                        lw=4,
                        solid_capstyle="round",
                    )
                    cd_bar_y -= 0.15

        # Draw CD reference indicator at top
        ax.plot([1, 1 + cd], [0.45, 0.45], color="#d62728", lw=3)
        ax.text(
            1 + (cd / 2.0),
            0.52,
            f"CD = {cd:.3f} (α={alpha})",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#d62728",
            fontweight="bold",
        )

        ax.set_xlim(0.5, n_models + 0.5)
        min_y = min(-1.0, cd_bar_y - 0.2)
        ax.set_ylim(min_y, 0.7)
        ax.axis("off")

        ax.set_title(
            f"Critical Difference (CD) Diagram for {metric_name}\n"
            f"(Models connected by red bars are NOT significantly different)",
            pad=15,
            fontsize=12,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Đã lưu biểu đồ CD Diagram tại: %s", save_path)
    except Exception as exc:
        logger.error("Lỗi khi vẽ CD Diagram: %s", exc)
        plt.close(fig)
        return None

    return fig


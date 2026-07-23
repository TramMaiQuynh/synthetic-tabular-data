"""
Friedman + Nemenyi
Các kiểm định thống kê để so sánh nhiều models trên nhiều datasets.

Friedman test: 
  - Non-parametric alternative to repeated-measures ANOVA
  - H0: tất cả models có performance tương đương
  - H1: ít nhất 1 model khác biệt
  
Nemenyi post-hoc:
  - So sánh pairwise sau khi Friedman reject H0
  - Critical Difference (CD) cho biết chênh lệch bao nhiêu là đáng kể

Quy tắc sử dụng (theo Demšar, 2006):
  - Cần ≥ 2 datasets + ≥ 2 models
  - Friedman test trước, Nemenyi sau nếu p < 0.05
  - Vẽ CD diagram để trực quan hóa
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def friedman_test(
    data: pd.DataFrame,
    value_col: str,
    model_col: str = "model_type",
    dataset_col: str = "dataset",
) -> dict:
    """
    Friedman test: so sánh nhiều models trên nhiều datasets.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame chứa kết quả experiments.
    value_col : str
        Tên cột chứa giá trị metric (vd: 'avg_wasserstein', 'mia_auc').
    model_col : str
        Tên cột chứa tên model (vd: 'model_type').
    dataset_col : str
        Tên cột chứa tên dataset (vd: 'dataset').
    
    Returns
    -------
    dict với keys:
        - 'statistic': Friedman test statistic (Q)
        - 'p_value': p-value
        - 'is_significant': True nếu p < 0.05
        - 'n_models': số models
        - 'n_datasets': số datasets
        - 'mean_ranks': dict {model: avg_rank}
        - 'rank_matrix': DataFrame (dataset × model) với rank values
    """
    from scipy.stats import friedmanchisquare
    
    # Pivot: mỗi dataset là 1 row, mỗi model là 1 column
    pivot = data.pivot_table(
        index=dataset_col,
        columns=model_col,
        values=value_col,
        aggfunc="mean",
    )
    
    models = list(pivot.columns)
    datasets = list(pivot.index)
    
    n_models = len(models)
    n_datasets = len(datasets)
    
    if n_models < 2:
        return {
            "error": f"Cần ≥ 2 models, có {n_models}",
            "is_significant": None,
        }
    if n_datasets < 2:
        return {
            "error": f"Cần ≥ 2 datasets, có {n_datasets}",
            "is_significant": None,
        }
    
    # Rank within each dataset (1 = best)
    # Với metric: lower is better (Wasserstein, JSD, MIA AUC, leakage%)
    # Với metric: higher is better (DCR, F1, ROC-AUC)
    # Mặc định: lower is better → ascending=True
    rank_matrix = pivot.rank(axis=1, ascending=True, method="average")
    
    # Friedman test
    groups = [pivot[model].values for model in models]
    statistic, p_value = friedmanchisquare(*groups)
    
    # Mean ranks
    mean_ranks = rank_matrix.mean(axis=0).to_dict()
    mean_ranks = {k: float(v) for k, v in mean_ranks.items()}
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "is_significant": bool(p_value < 0.05),
        "n_models": n_models,
        "n_datasets": n_datasets,
        "mean_ranks": mean_ranks,
        "rank_matrix": rank_matrix,
        "models": models,
        "datasets": datasets,
    }


def nemenyi_posthoc(rank_matrix: pd.DataFrame) -> dict:
    """
    Nemenyi post-hoc test: pairwise comparison sau Friedman.
    
    Parameters
    ----------
    rank_matrix : pd.DataFrame
        DataFrame (dataset × model) với rank values (1 = best).
        Có thể lấy từ kết quả friedman_test['rank_matrix'].
    
    Returns
    -------
    dict với keys:
        - 'p_values': DataFrame pairwise p-values
        - 'is_significant': DataFrame boolean (p < 0.05)
        - 'cd': Critical Difference (chênh lệch rank tối thiểu để significant)
    """
    import scikit_posthocs as sp
    
    # scikit_posthocs.nemenyi_friedman expects wide format (datasets × models)
    # hoặc long format. Dùng wide.
    p_values = sp.posthoc_nemenyi_friedman(rank_matrix.values)
    p_values = pd.DataFrame(
        p_values,
        index=rank_matrix.columns,
        columns=rank_matrix.columns,
    )
    
    is_sig = p_values < 0.05
    
    # Critical Difference (Demšar, 2006 eq. 7)
    # CD = q_alpha * sqrt(k * (k + 1) / (6 * N))
    # q_alpha = studentized range statistic / sqrt(2)
    from scipy.stats import distributions
    k = rank_matrix.shape[1]  # number of models
    N = rank_matrix.shape[0]  # number of datasets
    
    # For infinite df, q_inf = 3.314 (k=3, alpha=0.05)
    # Tra cứu từ studentized range distribution
    try:
        q_alpha = distributions.studentized_range.ppf(0.95, k, 100000) / np.sqrt(2)
    except AttributeError:
        # Fallback values (Demšar 2006, Table 1)
        q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
        q_alpha = q_table.get(k, 2.343) / np.sqrt(2)
    
    cd = float(q_alpha * np.sqrt(k * (k + 1) / (6 * N)))
    
    return {
        "p_values": p_values,
        "is_significant": is_sig,
        "cd": cd,
        "k": k,
        "N": N,
    }


def plot_cd_diagram(
    mean_ranks: Dict[str, float],
    cd: float,
    models: List[str],
    title: str = "Critical Difference Diagram",
    higher_is_better: bool = False,
) -> str:
    """
    Vẽ Critical Difference (CD) diagram — biểu đồ chuẩn cho Nemenyi test.
    
    Parameters
    ----------
    mean_ranks : dict {model: avg_rank}
    cd : float (Critical Difference từ nemenyi_posthoc)
    models : list (thứ tự models theo rank)
    title : str
    higher_is_better : bool (nếu True, rank cao = tốt → đảo chiều)
    
    Returns
    -------
    str: đường dẫn đến file PNG đã lưu
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Sắp xếp models theo mean rank
    sorted_models = sorted(mean_ranks.items(), key=lambda x: x[1], reverse=higher_is_better)
    
    fig, ax = plt.subplots(figsize=(10, 3 + 0.4 * len(models)))
    
    # Vẽ axis
    ax.set_xlim(1, len(models))
    ax.set_ylim(-0.5, 1)
    ax.axhline(y=0, color="gray", linewidth=1)
    
    # Vẽ từng model
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for i, (model, rank) in enumerate(sorted_models):
        ax.plot(rank, 0, "o", color=colors[i], markersize=12, zorder=5)
        ax.text(rank, -0.15, model, ha="center", va="top", fontsize=10, rotation=30)
        ax.text(rank, 0.1, f"{rank:.3f}", ha="center", va="bottom", fontsize=8, color="gray")
    
    # Vẽ CD bar
    # CD bar nằm ở trên cùng, bên phải
    cd_start = len(models) - cd
    cd_end = len(models)
    cd_mid = (cd_start + cd_end) / 2
    
    ax.plot([cd_start, cd_end], [0.7, 0.7], "k-", linewidth=2)
    ax.plot([cd_start, cd_start], [0.65, 0.75], "k-", linewidth=1)
    ax.plot([cd_end, cd_end], [0.65, 0.75], "k-", linewidth=1)
    ax.text(cd_mid, 0.78, f"CD = {cd:.3f}", ha="center", fontsize=9)
    
    # Vẽ groups (models không khác biệt nhau)
    # Nếu chênh lệch rank < CD, chúng thuộc cùng 1 group
    sorted_list = sorted(mean_ranks.items(), key=lambda x: x[1])
    groups = []
    used = set()
    for i in range(len(sorted_list)):
        if i in used:
            continue
        group = [sorted_list[i][0]]
        for j in range(i + 1, len(sorted_list)):
            if abs(sorted_list[j][1] - sorted_list[i][1]) < cd:
                group.append(sorted_list[j][0])
                used.add(j)
        groups.append(group)
        used.add(i)
    
    # Vẽ group bars
    for g_idx, group in enumerate(groups):
        if len(group) <= 1:
            continue
        ranks_g = [mean_ranks[m] for m in group]
        min_r, max_r = min(ranks_g), max(ranks_g)
        y_pos = 0.35 + g_idx * 0.08
        ax.plot([min_r, max_r], [y_pos, y_pos], "k-", linewidth=3, alpha=0.6)
    
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Mean Rank (lower = better)" if not higher_is_better else "Mean Rank (higher = better)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    import os
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    os.makedirs(output_dir, exist_ok=True)
    
    sanitized_title = title.replace(" ", "_").replace(":", "").lower()
    output_path = os.path.join(output_dir, f"cd_diagram_{sanitized_title}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return output_path


def compare_models(
    data: pd.DataFrame,
    value_col: str,
    model_col: str = "model_type",
    dataset_col: str = "dataset",
    title: str = "",
    higher_is_better: bool = False,
    save_plot: bool = True,
) -> dict:
    """
    So sánh nhiều models trên nhiều datasets: Friedman → Nemenyi → CD Diagram.
    
    Đây là API cấp cao, gộp cả 3 bước.
    
    Parameters
    ----------
    data : pd.DataFrame
    value_col : str (vd: 'avg_wasserstein', 'mia_auc', 'tstr_rf_f1')
    model_col : str = 'model_type'
    dataset_col : str = 'dataset'
    title : str (cho CD diagram)
    higher_is_better : bool (nếu True: DCR, F1, ROC-AUC)
    save_plot : bool
    
    Returns
    -------
    dict với keys:
        - 'friedman': kết quả Friedman test
        - 'nemenyi': kết quả Nemenyi post-hoc
        - 'cd_diagram_path': đường dẫn đến CD diagram PNG (nếu save_plot)
        - 'interpretation': string giải thích kết quả
    """
    # 1. Friedman
    friedman = friedman_test(data, value_col, model_col, dataset_col)
    
    if friedman.get("error") or not friedman.get("is_significant"):
        interpretation = (
            f"Friedman test: p = {friedman['p_value']:.4f} "
            f"({'significant' if friedman.get('is_significant') else 'NOT significant'}). "
            f"Không đủ bằng chứng để kết luận các models khác nhau."
        )
        return {
            "friedman": friedman,
            "nemenyi": None,
            "cd_diagram_path": None,
            "interpretation": interpretation,
        }
    
    # 2. Nemenyi post-hoc
    nemenyi = nemenyi_posthoc(friedman["rank_matrix"])
    
    # 3. CD Diagram
    cd_path = None
    if save_plot:
        cd_path = plot_cd_diagram(
            friedman["mean_ranks"],
            nemenyi["cd"],
            friedman["models"],
            title=title or f"CD Diagram: {value_col}",
            higher_is_better=higher_is_better,
        )
    
    # 4. Interpretation
    mean_ranks = friedman["mean_ranks"]
    sorted_models = sorted(mean_ranks.items(), key=lambda x: x[1])
    best_model = sorted_models[0]
    worst_model = sorted_models[-1]
    
    # Check which pairs are significantly different
    sig_pairs = []
    for i, m1 in enumerate(friedman["models"]):
        for m2 in friedman["models"][i+1:]:
            p_val = nemenyi["p_values"].loc[m1, m2]
            if p_val < 0.05:
                sig_pairs.append((m1, m2, p_val))
    
    interpretation = (
        f"Friedman test rejects H0 (p = {friedman['p_value']:.4f} < 0.05). "
        f"Mean ranks: {', '.join(f'{m}={r:.3f}' for m, r in sorted_models)}. "
        f"Best: {best_model[0]} (rank {best_model[1]:.3f}). "
        f"CD (Nemenyi) = {nemenyi['cd']:.3f}. "
    )
    if sig_pairs:
        interpretation += f"Significant pairs: {'; '.join(f'{a} vs {b} (p={p:.4f})' for a, b, p in sig_pairs)}."
    else:
        interpretation += "No pairwise differences detected by Nemenyi post-hoc."
    
    return {
        "friedman": friedman,
        "nemenyi": nemenyi,
        "cd_diagram_path": cd_path,
        "interpretation": interpretation,
    }
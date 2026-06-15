"""
Compliance Report Generator
---------------------------
Formats the evaluated fidelity, privacy, and utility metrics into:
1. `compliance_report.md` (Markdown format)
2. `compliance_report.html` (Beautifully styled, self-contained HTML report with CSS)
"""

import os
import html
import datetime
import logging
import numpy as np
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class ComplianceReporter:
    """Consolidates metrics and exports Markdown and HTML reports."""
    
    def __init__(self, dataset_name: str, output_dir: str) -> None:
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_report(
        self,
        fidelity_results: Dict[str, Any],
        privacy_results: Dict[str, Any],
        utility_results: Dict[str, Any],
        relative_plot_paths: Dict[str, str], # paths to grid, corr, dcr relative to output_dir
        target_col: str = "",
        sensitive_col: str = "",
    ) -> Tuple[str, str]:
        """
        Generate Markdown and HTML reports.
        
        Returns the absolute paths to the saved Markdown and HTML files.
        """
        md_content = self._build_markdown(fidelity_results, privacy_results, utility_results, relative_plot_paths, target_col, sensitive_col)
        html_content = self._build_html(fidelity_results, privacy_results, utility_results, relative_plot_paths, target_col, sensitive_col)
        
        md_path = os.path.join(self.output_dir, "compliance_report.md")
        html_path = os.path.join(self.output_dir, "compliance_report.html")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info("Compliance reports generated successfully at %s and %s", md_path, html_path)
        return md_path, html_path
 
    def _build_markdown(
        self,
        fidelity: Dict[str, Any],
        privacy: Dict[str, Any],
        utility: Dict[str, Any],
        plots: Dict[str, str],
        target_col: str,
        sensitive_col: str,
    ) -> str:
        """Construct the Markdown report content."""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Title and Header
        lines = [
            f"# Compliance Audit Report — Dataset: `{self.dataset_name}`",
            f"**Generated at:** {date_str}  ",
            "**Module 3 Audit Engine — Enterprise Production Quality Assurance**",
            "",
            "## 1. Executive Summary",
            "",
        ]
        
        # Calculate overall compliance statuses
        dcr_leakage = privacy.get("dcr_leakage_pct", 0.0)
        avg_js = np.mean(list(fidelity.get("js_divergence", {}).values())) if fidelity.get("js_divergence") else 0.0
        avg_wasserstein = np.mean(list(fidelity.get("wasserstein", {}).values())) if fidelity.get("wasserstein") else 0.0
        avg_corr_diff = fidelity.get("correlation_difference", 0.0)
        mia_auc = privacy.get("mia_auc", 0.5)
        
        # Determine status badges
        privacy_status = "🟢 SECURE" if dcr_leakage < 1.0 and mia_auc <= 0.65 else "🟡 WARNING"
        if dcr_leakage >= 5.0 or mia_auc >= 0.8:
            privacy_status = "🔴 LEAKAGE DETECTED"
            
        fidelity_status = "🟢 HIGH FIDELITY" if avg_js < 0.05 and avg_corr_diff < 0.1 else "🟡 MEDIUM FIDELITY"
        if avg_js >= 0.15:
            fidelity_status = "🔴 LOW FIDELITY"
            
        lines.extend([
            f"| Metric Domain | Audit Score | Status | Description |",
            f"|---|---|---|---|",
            f"| **Privacy & Security** | DCR Leakage: {dcr_leakage:.2f}% <br> MIA AUC: {mia_auc:.2f} | {privacy_status} | Checks for record memorization & membership leakage. |",
            f"| **Statistical Fidelity** | Avg JSD: {avg_js:.4f} <br> Correlation Diff: {avg_corr_diff:.4f} | {fidelity_status} | Measures similarity of marginal and joint distributions. |",
            f"| **Machine Learning Utility** | Task: {utility.get('task', 'unknown')} | 🟢 EVALUATED | Measures prediction capacity on synthetic training data. |",
            "",
            "---",
            "",
            "## 2. Statistical Fidelity Audit",
            "",
            "### Marginal Distributions Similarity",
            "| Column Name | Metric | Distance/Divergence | Status |",
            "|---|---|---|---|",
        ])
        
        # continuous features (Wasserstein)
        for col, val in fidelity.get("wasserstein", {}).items():
            status = "🟢 Good" if val < 0.05 else "🟡 Moderate"
            if val >= 0.15:
                status = "🔴 High Divergence"
            lines.append(f"| `{col}` | Wasserstein Distance | {val:.4f} | {status} |")
            
        # categorical features (JS Divergence)
        for col, val in fidelity.get("js_divergence", {}).items():
            status = "🟢 Good" if val < 0.03 else "🟡 Moderate"
            if val >= 0.08:
                status = "🔴 High Divergence"
            lines.append(f"| `{col}` | Jensen-Shannon Divergence | {val:.4f} | {status} |")
            
        lines.extend([
            "",
            f"**Average Correlation Difference (Joint Distribution):** {avg_corr_diff:.4f}",
            "",
            "---",
            "",
            "## 3. Privacy & Memorization Audit",
            "",
            f"- **Distance to Closest Record (DCR) Mean:** {privacy.get('dcr_mean', 0.0):.4f} (Normalized L2)",
            f"- **Distance to Closest Record (DCR) Min:** {privacy.get('dcr_min', 0.0):.4f}",
            f"- **DCR Leakage Percentage (<0.01):** {dcr_leakage:.2f}% (Synthetic rows matching real training samples)",
            f"- **Nearest Neighbor Distance Ratio (NNDR) Mean:** {privacy.get('nndr_mean', 0.0):.4f}",
            f"- **Membership Inference Attack (MIA) AUC:** {mia_auc:.4f} (Attacker prediction capability)",
            "",
        ])
        
        # AIA results
        if "aia" in privacy and privacy["aia"]:
            aia = privacy["aia"]
            lines.append("### Attribute Inference Attack (AIA) Simulation")
            if "error" in aia:
                lines.append(f"AIA failed: {aia['error']}")
            else:
                lines.extend([
                    f"- **Target Attribute:** `{sensitive_col}`",
                    f"- **Task Type:** {aia.get('task')}",
                ])
                if aia.get("task") == "classification":
                    lines.extend([
                        f"- **Prediction Accuracy:** {aia.get('accuracy', 0.0) * 100:.2f}%",
                        f"- **F1-Score (Macro):** {aia.get('f1_score', 0.0):.4f}",
                    ])
                else:
                    lines.extend([
                        f"- **MSE:** {aia.get('mse', 0.0):.4f}",
                        f"- **R2 Score:** {aia.get('r2_score', 0.0):.4f}",
                    ])
            lines.append("")
            
        lines.extend([
            "---",
            "",
            "## 4. Machine Learning Utility (TSTR Framework)",
            "",
            f"**Task Type:** {utility.get('task', 'unknown').upper()}",
            f"**Target Column:** `{target_col}`",
            "",
            "| Model Name | TRTR Score (Train Real) | TSTR Score (Train Synthetic) | Difference | Status |",
            "|---|---|---|---|---|",
        ])
        
        # Add TSTR model comparisons
        task_type = utility.get("task")
        for model_name, score_dict in utility.get("metrics", {}).items():
            trtr = score_dict.get("TRTR", {})
            tstr = score_dict.get("TSTR", {})
            
            if task_type == "classification":
                trtr_metric = trtr.get("f1_macro", 0.0)
                tstr_metric = tstr.get("f1_macro", 0.0)
                metric_name = "F1-Macro"
            else:
                trtr_metric = trtr.get("r2", 0.0)
                tstr_metric = tstr.get("r2", 0.0)
                metric_name = "R2-Score"
                
            diff = trtr_metric - tstr_metric
            status = "🟢 Excellent" if diff < 0.05 else "🟡 Acceptable"
            if diff >= 0.15:
                status = "🔴 Low Utility"
                
            lines.append(
                f"| `{model_name}` | {metric_name}: {trtr_metric:.4f} | {metric_name}: {tstr_metric:.4f} | {diff:.4f} | {status} |"
            )
            
        lines.extend([
            "",
            "---",
            "",
            "## 5. Visual Distribution Overlays",
            "",
            "### Feature Distributions Overlay",
            f"![Distributions Grid]({plots.get('distributions')})",
            "",
            "### Correlation Matrix Comparison",
            f"![Correlation Heatmaps]({plots.get('correlation')})",
            "",
            "### Geometric Privacy Curve (DCR)",
            f"![DCR Distribution]({plots.get('dcr')})",
            "",
        ])
        
        return "\n".join(lines)

    def _build_html(
        self,
        fidelity: Dict[str, Any],
        privacy: Dict[str, Any],
        utility: Dict[str, Any],
        plots: Dict[str, str],
        target_col: str,
        sensitive_col: str,
    ) -> str:
        """Construct the self-contained HTML report with CSS styling."""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate summary metrics
        dcr_leakage = privacy.get("dcr_leakage_pct", 0.0)
        avg_js = np.mean(list(fidelity.get("js_divergence", {}).values())) if fidelity.get("js_divergence") else 0.0
        avg_corr_diff = fidelity.get("correlation_difference", 0.0)
        mia_auc = privacy.get("mia_auc", 0.5)
        
        # Compute status strings & colors
        if dcr_leakage < 1.0 and mia_auc <= 0.65:
            priv_badge = "status-green"
            priv_label = "SECURE"
        elif dcr_leakage >= 5.0 or mia_auc >= 0.8:
            priv_badge = "status-red"
            priv_label = "LEAKAGE DETECTED"
        else:
            priv_badge = "status-yellow"
            priv_label = "WARNING"
            
        if avg_js < 0.05 and avg_corr_diff < 0.1:
            fid_badge = "status-green"
            fid_label = "HIGH FIDELITY"
        elif avg_js >= 0.15:
            fid_badge = "status-red"
            fid_label = "LOW FIDELITY"
        else:
            fid_badge = "status-yellow"
            fid_label = "MEDIUM FIDELITY"

        html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compliance Audit Report - {html.escape(self.dataset_name)}</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #333333;
            background-color: #f7f9fa;
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 1100px;
            margin: 40px auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }}
        header {{
            border-bottom: 2px solid #eaeaea;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            font-size: 28px;
            margin: 0 0 10px 0;
            color: #1a1a1a;
            font-weight: 700;
        }}
        .meta {{
            font-size: 14px;
            color: #666666;
        }}
        .meta strong {{
            color: #333333;
        }}
        h2 {{
            font-size: 20px;
            color: #2c3e50;
            margin-top: 40px;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }}
        h3 {{
            font-size: 16px;
            color: #34495e;
            margin-top: 25px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0 25px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e1e8ed;
            font-size: 14px;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #475569;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .status-green {{
            background-color: #d1fae5;
            color: #065f46;
        }}
        .status-yellow {{
            background-color: #fef3c7;
            color: #92400e;
        }}
        .status-red {{
            background-color: #fee2e2;
            color: #991b1b;
        }}
        .img-container {{
            text-align: center;
            margin: 30px 0;
            background: #fafafa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #eaeaea;
        }}
        .img-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 20px;
            border-radius: 8px;
        }}
        .card h4 {{
            margin: 0 0 10px 0;
            color: #475569;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .card .value {{
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Compliance Audit Report</h1>
            <div class="meta">
                Dataset: <strong>{html.escape(self.dataset_name)}</strong> &nbsp;|&nbsp; 
                Audited at: <strong>{date_str}</strong> &nbsp;|&nbsp; 
                Audit Engine: <strong>Module 3 Compliance Validator</strong>
            </div>
        </header>

        <section>
            <h2>1. Executive Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric Domain</th>
                        <th>Audit Scores</th>
                        <th>Status</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Privacy & Security</strong></td>
                        <td>
                            DCR Leakage: {dcr_leakage:.2f}% <br>
                            MIA AUC: {mia_auc:.2f}
                        </td>
                        <td><span class="badge {priv_badge}">{priv_label}</span></td>
                        <td>Ensures synthetic records do not copy real training records and cannot be traced to member sources.</td>
                    </tr>
                    <tr>
                        <td><strong>Statistical Fidelity</strong></td>
                        <td>
                            Avg JSD: {avg_js:.4f} <br>
                            Corr Difference: {avg_corr_diff:.4f}
                        </td>
                        <td><span class="badge {fid_badge}">{fid_label}</span></td>
                        <td>Validates correctness of both marginal feature distributions and multi-column correlations.</td>
                    </tr>
                    <tr>
                        <td><strong>Machine Learning Utility</strong></td>
                        <td>
                            Task: {html.escape(utility.get("task", "unknown").upper())} <br>
                            Target: {html.escape(target_col)}
                        </td>
                        <td><span class="badge status-green">EVALUATED</span></td>
                        <td>Ensures predicting on synthetic data produces models usable in real-world contexts.</td>
                    </tr>
                </tbody>
            </table>
        </section>

        <section>
            <h2>2. Statistical Fidelity Audit</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Metric Type</th>
                        <th>Distance / Divergence</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>"""
        
        for col, val in fidelity.get("wasserstein", {}).items():
            status_cls = "status-green" if val < 0.05 else "status-yellow"
            status_text = "GOOD" if val < 0.05 else "MODERATE"
            if val >= 0.15:
                status_cls = "status-red"
                status_text = "HIGH DIVERGENCE"
            html_report += f"""
                    <tr>
                        <td><code>{html.escape(col)}</code></td>
                        <td>Wasserstein Distance</td>
                        <td>{val:.4f}</td>
                        <td><span class="badge {status_cls}">{status_text}</span></td>
                    </tr>"""
                    
        for col, val in fidelity.get("js_divergence", {}).items():
            status_cls = "status-green" if val < 0.03 else "status-yellow"
            status_text = "GOOD" if val < 0.03 else "MODERATE"
            if val >= 0.08:
                status_cls = "status-red"
                status_text = "HIGH DIVERGENCE"
            html_report += f"""
                    <tr>
                        <td><code>{html.escape(col)}</code></td>
                        <td>Jensen-Shannon Divergence</td>
                        <td>{val:.4f}</td>
                        <td><span class="badge {status_cls}">{status_text}</span></td>
                    </tr>"""
                    
        html_report += f"""
                </tbody>
            </table>
            <p><strong>Average Correlation Difference (Pearson/Cramer/Ratio):</strong> <code>{avg_corr_diff:.4f}</code></p>
        </section>

        <section>
            <h2>3. Privacy & Memorization Audit</h2>
            <div class="card-grid">
                <div class="card">
                    <h4>MIA Attacker AUC</h4>
                    <div class="value">{mia_auc:.4f}</div>
                    <p style="font-size: 12px; margin: 5px 0 0 0; color:#666;">Target: ~0.50 (random guessing)</p>
                </div>
                <div class="card">
                    <h4>DCR Leakage Percentage</h4>
                    <div class="value">{dcr_leakage:.2f}%</div>
                    <p style="font-size: 12px; margin: 5px 0 0 0; color:#666;">Share of rows with L2 distance < 0.01</p>
                </div>
                <div class="card">
                    <h4>DCR Mean Distance</h4>
                    <div class="value">{privacy.get("dcr_mean", 0.0):.4f}</div>
                    <p style="font-size: 12px; margin: 5px 0 0 0; color:#666;">Average distance to real training set</p>
                </div>
                <div class="card">
                    <h4>NNDR Mean Ratio</h4>
                    <div class="value">{privacy.get("nndr_mean", 0.0):.4f}</div>
                    <p style="font-size: 12px; margin: 5px 0 0 0; color:#666;">Ratio of 1st closest vs 2nd closest</p>
                </div>
            </div>"""

        if "aia" in privacy and privacy["aia"]:
            aia = privacy["aia"]
            html_report += f"""
            <h3>Attribute Inference Attack (AIA) Simulation</h3>"""
            if "error" in aia:
                html_report += f"""<p style="color:red;">AIA simulation error: {aia['error']}</p>"""
            else:
                html_report += f"""
                <p>Sensitive Column Audited: <strong><code>{html.escape(sensitive_col)}</code></strong> (Task: <strong>{html.escape(str(aia.get("task", "")))}</strong>)</p>
                <table>
                    <thead>
                        <tr>"""
                if aia.get("task") == "classification":
                    html_report += f"""
                            <th>Accuracy</th>
                            <th>F1-Score (Macro)</th>"""
                else:
                    html_report += f"""
                            <th>Mean Squared Error (MSE)</th>
                            <th>R2 Score</th>"""
                html_report += f"""
                        </tr>
                    </thead>
                    <tbody>
                        <tr>"""
                if aia.get("task") == "classification":
                    html_report += f"""
                            <td>{aia.get("accuracy", 0.0)*100:.2f}%</td>
                            <td>{aia.get("f1_score", 0.0):.4f}</td>"""
                else:
                    html_report += f"""
                            <td>{aia.get("mse", 0.0):.4f}</td>
                            <td>{aia.get("r2_score", 0.0):.4f}</td>"""
                html_report += f"""
                        </tr>
                    </tbody>
                </table>"""

        html_report += f"""
        </section>

        <section>
            <h2>4. Machine Learning Utility (TSTR Framework)</h2>
            <p>Predictive ML models trained on synthetic data vs real data, evaluated on the real holdout test set.</p>
            <table>
                <thead>
                    <tr>
                        <th>Model Name</th>
                        <th>Target Column</th>
                        <th>TRTR Score (Train Real)</th>
                        <th>TSTR Score (Train Synthetic)</th>
                        <th>Difference</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>"""
                
        task_type = utility.get("task")
        for model_name, score_dict in utility.get("metrics", {}).items():
            trtr = score_dict.get("TRTR", {})
            tstr = score_dict.get("TSTR", {})
            
            if task_type == "classification":
                trtr_val = trtr.get("f1_macro", 0.0)
                tstr_val = tstr.get("f1_macro", 0.0)
                m_name = "F1-Macro"
            else:
                trtr_val = trtr.get("r2", 0.0)
                tstr_val = tstr.get("r2", 0.0)
                m_name = "R2-Score"
                
            diff = trtr_val - tstr_val
            status_cls = "status-green" if diff < 0.05 else "status-yellow"
            status_lbl = "EXCELLENT" if diff < 0.05 else "ACCEPTABLE"
            if diff >= 0.15:
                status_cls = "status-red"
                status_lbl = "LOW UTILITY"
                
            html_report += f"""
                    <tr>
                        <td><strong>{model_name}</strong></td>
                        <td><code>{html.escape(target_col)}</code></td>
                        <td>{m_name}: {trtr_val:.4f}</td>
                        <td>{m_name}: {tstr_val:.4f}</td>
                        <td>{diff:.4f}</td>
                        <td><span class="badge {status_cls}">{status_lbl}</span></td>
                    </tr>"""
                    
        html_report += f"""
                </tbody>
            </table>
        </section>

        <section>
            <h2>5. Visual Distribution Overlays</h2>
            
            <h3>Feature Distributions Overlay</h3>
            <div class="img-container">
                <img src="{plots.get("distributions")}" alt="Feature Distribution Comparison Grid">
            </div>
            
            <h3>Correlation Matrix Comparison</h3>
            <div class="img-container">
                <img src="{plots.get("correlation")}" alt="Correlation Comparison Heatmaps">
            </div>
            
            <h3>Geometric Privacy Curve (DCR)</h3>
            <div class="img-container">
                <img src="{plots.get("dcr")}" alt="DCR Distribution Curve">
            </div>
        </section>
    </div>
</body>
</html>"""
        return html_report

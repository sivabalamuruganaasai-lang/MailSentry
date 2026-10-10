"""
Plotting module for Level 6 Threshold & False-Positive Optimization.

Generates 7 publication-quality plots under reports/figures/:
  1. Precision vs Recall vs Threshold
  2. FPR vs Threshold
  3. F1 vs Threshold
  4. F0.5 vs Threshold
  5. Precision vs FPR
  6. ROC Curve (Validation)
  7. Precision-Recall Curve (Validation)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

FIGURES_DIR = Path("d:/email-spam-classifier/reports/figures").resolve()


def style_ax(ax, title: str, xlabel: str, ylabel: str):
    """Apply consistent elegant styling to matplotlib axis."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_all_level6_plots(
    sweeps_dict: Dict[str, pd.DataFrame],
    scores_dict: Dict[str, np.ndarray],
    y_val: np.ndarray,
    selected_points: Dict[str, Dict[str, Any]],
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    Generate the 7 required Level 6 threshold analysis plots.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    colors = {"E03": "#1f77b4", "E11": "#2ca02c", "E02": "#ff7f0e"}
    saved_paths = []

    # ---------------------------------------------------------
    # Plot 1: Precision & Recall vs Threshold for Top Candidates
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_id, df in sweeps_dict.items():
        c = colors.get(exp_id, "#333333")
        ax.plot(df["threshold"], df["precision"], label=f"{exp_id} Precision", color=c, linewidth=2, linestyle="-")
        ax.plot(df["threshold"], df["recall"], label=f"{exp_id} Recall", color=c, linewidth=2, linestyle="--")

        # Mark selected threshold point
        if exp_id in selected_points:
            pt = selected_points[exp_id]
            ax.scatter([pt["threshold"]], [pt["precision"]], color=c, marker="o", s=80, zorder=5)
            ax.scatter([pt["threshold"]], [pt["recall"]], color=c, marker="s", s=80, zorder=5)

    style_ax(ax, "Precision & Recall vs Decision Threshold (Validation)", "Decision Threshold / Score", "Metric Value")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", frameon=True)
    p1 = output_dir / "plot1_precision_recall_vs_threshold.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()
    saved_paths.append(p1)

    # ---------------------------------------------------------
    # Plot 2: FPR vs Threshold
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_id, df in sweeps_dict.items():
        c = colors.get(exp_id, "#333333")
        ax.plot(df["threshold"], df["fpr"] * 100, label=f"{exp_id} FPR (%)", color=c, linewidth=2)
        if exp_id in selected_points:
            pt = selected_points[exp_id]
            ax.scatter([pt["threshold"]], [pt["fpr"] * 100], color=c, marker="*", s=150, zorder=5,
                       label=f"{exp_id} Selected ({pt['fpr']*100:.2f}%)")

    ax.axhline(2.0, color="gray", linestyle=":", label="FPR Target 2.0%")
    ax.axhline(1.0, color="darkgray", linestyle=":", label="FPR Target 1.0%")
    style_ax(ax, "False Positive Rate (FPR) vs Decision Threshold", "Decision Threshold / Score", "False Positive Rate (%)")
    ax.legend(loc="upper right", frameon=True)
    p2 = output_dir / "plot2_fpr_vs_threshold.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()
    saved_paths.append(p2)

    # ---------------------------------------------------------
    # Plot 3: F1 vs Threshold
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_id, df in sweeps_dict.items():
        c = colors.get(exp_id, "#333333")
        ax.plot(df["threshold"], df["f1"], label=f"{exp_id} F1", color=c, linewidth=2)
        if exp_id in selected_points:
            pt = selected_points[exp_id]
            ax.scatter([pt["threshold"]], [pt["f1"]], color=c, marker="*", s=150, zorder=5)

    style_ax(ax, "F1 Score vs Decision Threshold", "Decision Threshold / Score", "F1 Score")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower center", frameon=True)
    p3 = output_dir / "plot3_f1_vs_threshold.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close()
    saved_paths.append(p3)

    # ---------------------------------------------------------
    # Plot 4: F0.5 vs Threshold
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_id, df in sweeps_dict.items():
        c = colors.get(exp_id, "#333333")
        ax.plot(df["threshold"], df["f0_5"], label=f"{exp_id} F0.5 (Precision-weighted)", color=c, linewidth=2)
        if exp_id in selected_points:
            pt = selected_points[exp_id]
            ax.scatter([pt["threshold"]], [pt["f0_5"]], color=c, marker="*", s=150, zorder=5)

    style_ax(ax, "F0.5 Score vs Decision Threshold", "Decision Threshold / Score", "F0.5 Score")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower center", frameon=True)
    p4 = output_dir / "plot4_f0_5_vs_threshold.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=300)
    plt.close()
    saved_paths.append(p4)

    # ---------------------------------------------------------
    # Plot 5: Precision vs FPR
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_id, df in sweeps_dict.items():
        c = colors.get(exp_id, "#333333")
        df_sorted = df.sort_values(by="fpr")
        ax.plot(df_sorted["fpr"] * 100, df_sorted["precision"], label=exp_id, color=c, linewidth=2)
        if exp_id in selected_points:
            pt = selected_points[exp_id]
            ax.scatter([pt["fpr"] * 100], [pt["precision"]], color=c, marker="*", s=180, zorder=5,
                       label=f"{exp_id} Opt Threshold ({pt['threshold']:.3f})")

    ax.set_xlim(-0.1, 5.0)  # Focus on low FPR region 0-5%
    ax.set_ylim(0.85, 1.01)
    style_ax(ax, "Precision vs False Positive Rate (Low-FPR Region 0-5%)", "False Positive Rate (%)", "Precision")
    ax.legend(loc="lower right", frameon=True)
    p5 = output_dir / "plot5_precision_vs_fpr.png"
    plt.tight_layout()
    plt.savefig(p5, dpi=300)
    plt.close()
    saved_paths.append(p5)

    # ---------------------------------------------------------
    # Plot 6: ROC Curve
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    for exp_id, scores in scores_dict.items():
        c = colors.get(exp_id, "#333333")
        fpr_arr, tpr_arr, _ = roc_curve(y_val, scores)
        auc_val = roc_auc_score(y_val, scores)
        ax.plot(fpr_arr, tpr_arr, label=f"{exp_id} (AUC = {auc_val:.4f})", color=c, linewidth=2)

        if exp_id in selected_points:
            pt = selected_points[exp_id]
            ax.scatter([pt["fpr"]], [pt["tpr"]], color=c, marker="*", s=180, zorder=5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Baseline")
    style_ax(ax, "Receiver Operating Characteristic (ROC) Curve (Validation)", "False Positive Rate (FPR)", "True Positive Rate (TPR / Recall)")
    ax.legend(loc="lower right", frameon=True)
    p6 = output_dir / "plot6_roc_curve.png"
    plt.tight_layout()
    plt.savefig(p6, dpi=300)
    plt.close()
    saved_paths.append(p6)

    # ---------------------------------------------------------
    # Plot 7: Precision-Recall Curve
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    for exp_id, scores in scores_dict.items():
        c = colors.get(exp_id, "#333333")
        prec_arr, rec_arr, _ = precision_recall_curve(y_val, scores)
        pr_auc_val = average_precision_score(y_val, scores)
        ax.plot(rec_arr, prec_arr, label=f"{exp_id} (PR-AUC = {pr_auc_val:.4f})", color=c, linewidth=2)

        if exp_id in selected_points:
            pt = selected_points[exp_id]
            ax.scatter([pt["recall"]], [pt["precision"]], color=c, marker="*", s=180, zorder=5)

    style_ax(ax, "Precision-Recall Curve (Validation)", "Recall", "Precision")
    ax.set_ylim(0.85, 1.01)
    ax.set_xlim(0.80, 1.01)
    ax.legend(loc="lower left", frameon=True)
    p7 = output_dir / "plot7_pr_curve.png"
    plt.tight_layout()
    plt.savefig(p7, dpi=300)
    plt.close()
    saved_paths.append(p7)

    return saved_paths

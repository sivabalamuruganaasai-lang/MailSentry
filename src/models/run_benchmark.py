"""
Main Execution Script for Candidate Model Benchmarking (Level 5).

Runs E01-E11 baseline experiments, computes validation metrics, group-aware CV,
generates CSV & Markdown reports, confusion matrix figures, and validation error records.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.data.loader import load_all_splits
from src.models.benchmark import BenchmarkRunner, plot_confusion_matrices, run_group_aware_cv

REPORTS_DIR = Path("d:/email-spam-classifier/reports").resolve()


def generate_benchmark_markdown_report(
    df_results: pd.DataFrame,
    cv_results: dict,
    top_candidates: list
) -> str:
    """Generate comprehensive Markdown report ranking candidate models."""

    # Rank dataframe by 1. F1, 2. Precision, 3. FPR (asc), 4. Recall, 5. PR-AUC
    df_ranked = df_results.sort_values(
        by=["f1", "precision", "fpr", "recall", "pr_auc"],
        ascending=[False, False, True, False, False]
    ).reset_index(drop=True)

    top_3_ids = df_ranked["experiment_id"].head(3).tolist()

    report_lines = [
        "# Candidate Model Benchmarking & Evaluation Report (Level 5)\n",
        "---\n",
        "## Executive Summary",
        "This report details the empirical benchmarking of baseline classification models for email spam detection on the **Validation Set**. **The test set remained 100% untouched and protected** throughout all experiments.\n",
        "A total of **11 controlled experiments (E01 - E11)** were conducted across three candidate classifiers (`MultinomialNB`, `LogisticRegression`, `LinearSVC`) combined with different feature representations (Word TF-IDF, Char TF-IDF, Word+Char TF-IDF, Structural, and URL features).\n",
        "---\n",
        "## 1. Class & Problem Definitions\n",
        "- **Positive Class (1)**: **SPAM**",
        "- **Negative Class (0)**: **HAM**",
        "- **False Positive (FP)**: HAM incorrectly classified as SPAM (*Primary operational risk: user losing legitimate email*).",
        "- **False Positive Rate (FPR)**: $FP / (FP + TN)$",
        "- **True Positive Rate (TPR / Recall)**: $TP / (TP + FN)$\n",
        "---\n",
        "## 2. Ranked Validation Benchmark Results\n",
        "Experiments are ranked by **1. F1 Score**, **2. Precision**, **3. False Positive Rate (lower is better)**, **4. Recall**, and **5. PR-AUC**:\n",
        "| Rank | Exp ID | Model | Features | Accuracy | Precision | Recall | F1 | F0.5 | FPR | ROC-AUC | PR-AUC | Latency (ms/sample) | Size (KB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    ]

    for rank, row in enumerate(df_ranked.to_dict(orient="records"), start=1):
        report_lines.append(
            f"| **{rank}** | **{row['experiment_id']}** | {row['model_name']} | `{row['feature_families']}` | "
            f"{row['accuracy']:.4f} | **{row['precision']:.4f}** | {row['recall']:.4f} | **{row['f1']:.4f}** | "
            f"{row['f0_5']:.4f} | **{row['fpr']:.4f}** | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | "
            f"{row['inference_time_ms_per_sample']:.4f} ms | {row['model_size_kb']:.1f} KB |"
        )

    report_lines.extend([
        "\n---\n",
        "## 3. Stratified Group-Aware 5-Fold Cross-Validation on Training Data\n",
        "To verify stability and rule out overfitting to the validation split, 5-fold cross-validation was performed **strictly on Training Data (4,233 emails)**, enforcing zero group-crossing between folds. All TF-IDF vectorizers, structural extractors, and scalers were **fit inside each fold independently** on fold-training data only.\n",
        "### Summary 5-Fold CV Metrics",
        "| Exp ID | Model | Mean F1 ± Std [Min, Max] | Mean Precision ± Std | Mean Recall ± Std | Mean FPR ± Std |",
        "|---|---|---|---|---|---|"
    ])

    for exp_id, cv_stat in cv_results.items():
        report_lines.append(
            f"| **{exp_id}** | {cv_stat['model_name']} | "
            f"**{cv_stat['mean_f1']:.4f} ± {cv_stat['std_f1']:.4f}** [{cv_stat['min_f1']:.4f}, {cv_stat['max_f1']:.4f}] | "
            f"{cv_stat['mean_precision']:.4f} ± {cv_stat['std_precision']:.4f} | "
            f"{cv_stat['mean_recall']:.4f} ± {cv_stat['std_recall']:.4f} | "
            f"**{cv_stat['mean_fpr']:.4f} ± {cv_stat['std_fpr']:.4f}** |"
        )

    report_lines.extend([
        "\n### Per-Fold Performance Breakdown\n"
    ])

    for exp_id, cv_stat in cv_results.items():
        if "folds" in cv_stat and cv_stat["folds"]:
            report_lines.extend([
                f"#### Experiment {exp_id} ({cv_stat['model_name']}) — Per-Fold Results",
                "| Fold | Train Samples | Val Samples | F1 Score | Precision | Recall | FPR | False Positives | False Negatives |",
                "|---|---|---|---|---|---|---|---|---|"
            ])
            for f_row in cv_stat["folds"]:
                report_lines.append(
                    f"| Fold {f_row['fold']} | {f_row['train_size']} | {f_row['val_size']} | "
                    f"**{f_row['f1']:.4f}** | {f_row['precision']:.4f} | {f_row['recall']:.4f} | "
                    f"**{f_row['fpr']:.4f}** | {f_row['fp']} | {f_row['fn']} |"
                )
            report_lines.extend([
                f"| **Mean ± Std** | - | - | **{cv_stat['mean_f1']:.4f} ± {cv_stat['std_f1']:.4f}** | "
                f"{cv_stat['mean_precision']:.4f} ± {cv_stat['std_precision']:.4f} | "
                f"{cv_stat['mean_recall']:.4f} ± {cv_stat['std_recall']:.4f} | "
                f"**{cv_stat['mean_fpr']:.4f} ± {cv_stat['std_fpr']:.4f}** | - | - |\n"
            ])

    report_lines.extend([
        "---\n",
        "## 4. Top 3 Candidate Models Analysis\n",
        f"Based on multi-objective validation criteria, the top 3 performing candidates are:\n"
    ])

    for i, exp_id in enumerate(top_3_ids, start=1):
        row = df_ranked[df_ranked["experiment_id"] == exp_id].iloc[0]
        report_lines.extend([
            f"### Top {i}: Experiment {exp_id} ({row['model_name']} + `{row['feature_families']}`)",
            f"- **F1 Score**: **{row['f1']:.4f}** (F0.5 = {row['f0_5']:.4f})",
            f"- **Precision**: **{row['precision']:.4f}** ({row['fp']} False Positives out of {row['fp'] + row['tn']} HAM emails)",
            f"- **False Positive Rate**: **{row['fpr']:.4f}** ({round(row['fpr']*100, 2)}%)",
            f"- **Recall**: **{row['recall']:.4f}** ({row['tp']} SPAM detected out of {row['tp'] + row['fn']})",
            f"- **PR-AUC**: **{row['pr_auc']:.4f}** | **ROC-AUC**: **{row['roc_auc']:.4f}**",
            f"- **Inference Speed**: **{row['inference_time_ms_per_sample']:.4f} ms/email** | **Model Size**: **{row['model_size_kb']:.1f} KB**\n"
        ])

    report_lines.extend([
        "---\n",
        "## 5. Confusion Matrices Summary\n",
        "Confusion matrix plots for top candidates have been saved under `reports/figures/`:",
        "- `reports/figures/confusion_matrix_E03.png`",
        "- `reports/figures/confusion_matrix_E08.png`",
        "- `reports/figures/confusion_matrix_E11.png`\n",
        "---\n",
        "## 6. Error Analysis Preparation & Validation Isolation\n",
        "- Validation predictions, decision scores, message IDs, ground truth, and predicted labels are persisted under `reports/val_predictions_<EXP_ID>.csv` for post-hoc error analysis.",
        "- **Test Set Protection Verification**: Zero test samples were used in model fitting, feature extraction fitting, hyperparameter selection, or ranking."
    ])

    report_md = "\n".join(report_lines)
    report_path = REPORTS_DIR / "model_benchmark.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[+] Saved Markdown benchmark report: {report_path}")

    return report_md


def run_benchmark():
    print("[>] Loading train, val, and test splits...")
    splits = load_all_splits()

    train_records, y_train, train_df = splits["train"]
    val_records, y_val, val_df = splits["val"]
    test_records, y_test, test_df = splits["test"]

    print(f"    Train size: {len(y_train)} samples")
    print(f"    Val size:   {len(y_val)} samples")
    print(f"    Test size:  {len(y_test)} samples (HOLDOUT - TOUCH NOTHING)")

    groups_train = np.array([r["group_id"] for r in train_records])

    runner = BenchmarkRunner(random_state=42)
    df_results, pipeline_cache, predictions_cache = runner.run_all(
        train_records=train_records,
        y_train=y_train,
        val_records=val_records,
        y_val=y_val
    )

    # Rank top models
    df_ranked = df_results.sort_values(
        by=["f1", "precision", "fpr", "recall", "pr_auc"],
        ascending=[False, False, True, False, False]
    ).reset_index(drop=True)

    top_3_ids = df_ranked["experiment_id"].head(3).tolist()
    print(f"[+] Top 3 Candidate Experiments: {top_3_ids}")

    # Generate Confusion Matrix figures for top 3 candidates + E10 + E11
    serious_candidates = list(dict.fromkeys(top_3_ids + ["E10", "E11"]))
    plot_confusion_matrices(serious_candidates, predictions_cache, y_val)

    # Save validation predictions for top candidate models (for error analysis preparation)
    for exp_id in serious_candidates:
        if exp_id in predictions_cache:
            y_pred, y_score = predictions_cache[exp_id]
            val_pred_df = pd.DataFrame({
                "message_id": [r["message_id"] for r in val_records],
                "file_path": [r["file_path"] for r in val_records],
                "group_id": [r["group_id"] for r in val_records],
                "true_label": y_val,
                "predicted_label": y_pred,
                "decision_score": np.round(y_score, 5)
            })
            val_pred_path = REPORTS_DIR / f"val_predictions_{exp_id}.csv"
            val_pred_df.to_csv(val_pred_path, index=False)
            print(f"[+] Saved validation predictions for {exp_id}: {val_pred_path}")

    # Run Group-Aware 5-Fold CV on TRAINING DATA ONLY for top 3 candidates + E10 + E11
    exp_config_map = {
        "E01": (["word"], "MultinomialNB", False),
        "E02": (["word"], "LogisticRegression", False),
        "E03": (["word"], "LinearSVC", False),
        "E04": (["char"], "MultinomialNB", False),
        "E05": (["char"], "LogisticRegression", False),
        "E06": (["char"], "LinearSVC", False),
        "E07": (["word", "char"], "MultinomialNB", False),
        "E08": (["word", "char"], "LogisticRegression", False),
        "E09": (["word", "char"], "LinearSVC", False),
        "E10": (["word", "char", "structural"], "LogisticRegression", True), # will adjust dynamically below
        "E11": (["word", "char", "structural", "url"], "LogisticRegression", True),
    }

    # Dynamically build configs for CV based on actual experiment results
    cv_configs = []
    for exp_id in serious_candidates:
        row = df_results[df_results["experiment_id"] == exp_id].iloc[0]
        fams = row["feature_families"].split("+")
        m_name = row["model_name"]
        scaling = row["scaling"] == "MaxAbsScaler"
        cv_configs.append((exp_id, fams, m_name, scaling))

    print("[>] Performing Group-Aware 5-Fold Stratified Cross-Validation on TRAINING data only...")
    cv_raw_results = run_group_aware_cv(
        train_records=train_records,
        y_train=y_train,
        groups_train=groups_train,
        exp_configs=cv_configs,
        n_splits=5,
        seed=42
    )

    # Attach model_name to cv_raw_results
    cv_results = {}
    for exp_id, stats in cv_raw_results.items():
        m_name = df_results[df_results["experiment_id"] == exp_id]["model_name"].values[0]
        cv_results[exp_id] = {"model_name": m_name, **stats}

    # Generate Markdown Report
    generate_benchmark_markdown_report(df_results, cv_results, top_3_ids)
    print("[+] Level 5 Candidate Model Benchmarking Complete!")


if __name__ == "__main__":
    run_benchmark()

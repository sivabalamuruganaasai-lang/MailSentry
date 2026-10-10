"""
Model Benchmarking and Experiment Matrix Orchestrator (Level 5).

Executes controlled experiments E01 through E11 on validation set.
Calculates comprehensive metrics (Accuracy, Precision, Recall, F1, F0.5, FPR, TPR,
ROC-AUC, PR-AUC, Confusion Matrix, Training Time, Latency, Serialized Size).

Protects TEST set strictly (never fitted or evaluated during model selection).
Includes Group-Aware Stratified Cross-Validation on TRAINING data only.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import StratifiedGroupKFold

from src.features.pipeline import CombinedFeaturePipeline
from src.models.naive_bayes import SpamNaiveBayes
from src.models.logistic_regression import SpamLogisticRegression
from src.models.linear_svm import SpamLinearSVM

REPORTS_DIR = Path("d:/email-spam-classifier/reports").resolve()
FIGURES_DIR = REPORTS_DIR / "figures"


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    train_time_sec: float,
    val_time_sec: float,
    model_size_kb: float
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for binary spam classification."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f0_5 = fbeta_score(y_true, y_pred, beta=0.5, pos_label=1, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Rates
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Ranking curves AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except Exception:
        pr_auc = 0.0

    num_samples = len(y_true)
    latency_ms = (val_time_sec / num_samples) * 1000.0 if num_samples > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "f0_5": round(float(f0_5), 4),
        "fpr": round(float(fpr), 4),
        "tpr": round(float(tpr), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_time_sec": round(float(train_time_sec), 4),
        "inference_time_ms_per_sample": round(float(latency_ms), 4),
        "model_size_kb": round(float(model_size_kb), 2)
    }


class SpamModelPipeline:
    """Integrated Feature Pipeline + Scaler + Estimator Wrapper."""

    def __init__(
        self,
        feature_families: List[str],
        model_name: str,
        word_mode: str = "light",
        word_max_features: int = 10000,
        char_max_features: int = 15000,
        apply_scaling: bool = False,
        random_state: int = 42
    ):
        self.feature_families = feature_families
        self.model_name = model_name
        self.word_mode = word_mode
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.apply_scaling = apply_scaling
        self.random_state = random_state

        self.feature_pipeline = CombinedFeaturePipeline(
            feature_families=feature_families,
            word_mode=word_mode,
            word_max_features=word_max_features,
            char_max_features=char_max_features
        )

        self.scaler = MaxAbsScaler() if apply_scaling else None

        if model_name == "MultinomialNB":
            self.model = SpamNaiveBayes(alpha=1.0)
        elif model_name == "LogisticRegression":
            self.model = SpamLogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        elif model_name == "LinearSVC":
            self.model = SpamLinearSVM(C=1.0, max_iter=2000, random_state=random_state)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        self.is_fitted_ = False

    def fit(self, records: List[Dict[str, Any]], y: np.ndarray):
        """Fit feature extractors, scaler, and estimator strictly on training data."""
        t0 = time.perf_counter()
        X_feat = self.feature_pipeline.fit_transform(records)
        if self.scaler is not None:
            X_feat = self.scaler.fit_transform(X_feat)
        self.model.fit(X_feat, y)
        self.is_fitted_ = True
        train_time = time.perf_counter() - t0
        return train_time

    def transform_features(self, records: List[Dict[str, Any]]):
        """Transform records using fitted pipeline and scaler."""
        if not self.is_fitted_:
            raise ValueError("Pipeline not fitted yet.")
        X_feat = self.feature_pipeline.transform(records)
        if self.scaler is not None:
            X_feat = self.scaler.transform(X_feat)
        return X_feat

    def predict(self, records: List[Dict[str, Any]]) -> np.ndarray:
        X_feat = self.transform_features(records)
        return self.model.predict(X_feat)

    def predict_score(self, records: List[Dict[str, Any]]) -> np.ndarray:
        X_feat = self.transform_features(records)
        return self.model.predict_score(X_feat)

    def evaluate(
        self,
        records_val: List[Dict[str, Any]],
        y_val: np.ndarray,
        train_time_sec: float
    ) -> Dict[str, Any]:
        """Evaluate model on validation records and measure latency & size."""
        t0 = time.perf_counter()
        y_pred = self.predict(records_val)
        val_time_sec = time.perf_counter() - t0

        y_score = self.predict_score(records_val)

        # Measure serialized size
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            joblib.dump(self, tmp.name)
            size_bytes = os.path.getsize(tmp.name)
            try:
                os.remove(tmp.name)
            except Exception:
                pass

        model_size_kb = size_bytes / 1024.0

        metrics = compute_evaluation_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_score=y_score,
            train_time_sec=train_time_sec,
            val_time_sec=val_time_sec,
            model_size_kb=model_size_kb
        )
        return metrics, y_pred, y_score


class BenchmarkRunner:
    """Executes experiment matrix and manages experiment artifacts."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def run_all(
        self,
        train_records: List[Dict[str, Any]],
        y_train: np.ndarray,
        val_records: List[Dict[str, Any]],
        y_val: np.ndarray
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run experiments E01 through E11."""

        experiments = [
            ("E01", ["word"], "MultinomialNB", False),
            ("E02", ["word"], "LogisticRegression", False),
            ("E03", ["word"], "LinearSVC", False),
            ("E04", ["char"], "MultinomialNB", False),
            ("E05", ["char"], "LogisticRegression", False),
            ("E06", ["char"], "LinearSVC", False),
            ("E07", ["word", "char"], "MultinomialNB", False),
            ("E08", ["word", "char"], "LogisticRegression", False),
            ("E09", ["word", "char"], "LinearSVC", False),
        ]

        results = []
        pipeline_cache = {}
        predictions_cache = {}

        print("[>] Executing Baseline Experiments E01 - E09...")
        for exp_id, feat_fams, model_name, scaling in experiments:
            print(f"    Running {exp_id}: {model_name} + {feat_fams}...")
            pipeline = SpamModelPipeline(
                feature_families=feat_fams,
                model_name=model_name,
                apply_scaling=scaling,
                random_state=self.random_state
            )
            train_time = pipeline.fit(train_records, y_train)
            metrics, y_pred, y_score = pipeline.evaluate(val_records, y_val, train_time)

            res = {
                "experiment_id": exp_id,
                "model_name": model_name,
                "feature_families": "+".join(feat_fams),
                "word_max_features": 10000 if "word" in feat_fams else 0,
                "char_max_features": 15000 if "char" in feat_fams else 0,
                "scaling": "MaxAbsScaler" if scaling else "None",
                "random_state": self.random_state,
                "train_size": len(y_train),
                "val_size": len(y_val),
                **metrics
            }
            results.append(res)
            pipeline_cache[exp_id] = pipeline
            predictions_cache[exp_id] = (y_pred, y_score)

        # Identify best candidate model/features from E01-E09 based on F1 and Precision
        df_base = pd.DataFrame(results)
        df_base = df_base.sort_values(
            by=["f1", "precision", "fpr", "recall", "pr_auc"],
            ascending=[False, False, True, False, False]
        )
        best_exp_row = df_base.iloc[0]
        best_exp_id = best_exp_row["experiment_id"]
        best_model_name = best_exp_row["model_name"]
        best_base_fams = best_exp_row["feature_families"].split("+")

        print(f"[+] Top candidate from E01-E09: {best_exp_id} ({best_model_name} on {best_base_fams})")

        # Run E10: Best candidate + Structural features
        print(f"    Running E10: {best_model_name} + {best_base_fams} + structural...")
        e10_fams = best_base_fams + ["structural"]
        # Use MaxAbsScaler when adding numerical structural/URL features
        pipeline_e10 = SpamModelPipeline(
            feature_families=e10_fams,
            model_name=best_model_name,
            apply_scaling=True,
            random_state=self.random_state
        )
        t_time10 = pipeline_e10.fit(train_records, y_train)
        metrics10, y_pred10, y_score10 = pipeline_e10.evaluate(val_records, y_val, t_time10)
        res10 = {
            "experiment_id": "E10",
            "model_name": best_model_name,
            "feature_families": "+".join(e10_fams),
            "word_max_features": 10000 if "word" in e10_fams else 0,
            "char_max_features": 15000 if "char" in e10_fams else 0,
            "scaling": "MaxAbsScaler",
            "random_state": self.random_state,
            "train_size": len(y_train),
            "val_size": len(y_val),
            **metrics10
        }
        results.append(res10)
        pipeline_cache["E10"] = pipeline_e10
        predictions_cache["E10"] = (y_pred10, y_score10)

        # Run E11: Best candidate + Structural + URL features
        print(f"    Running E11: {best_model_name} + {best_base_fams} + structural + url...")
        e11_fams = best_base_fams + ["structural", "url"]
        pipeline_e11 = SpamModelPipeline(
            feature_families=e11_fams,
            model_name=best_model_name,
            apply_scaling=True,
            random_state=self.random_state
        )
        t_time11 = pipeline_e11.fit(train_records, y_train)
        metrics11, y_pred11, y_score11 = pipeline_e11.evaluate(val_records, y_val, t_time11)
        res11 = {
            "experiment_id": "E11",
            "model_name": best_model_name,
            "feature_families": "+".join(e11_fams),
            "word_max_features": 10000 if "word" in e11_fams else 0,
            "char_max_features": 15000 if "char" in e11_fams else 0,
            "scaling": "MaxAbsScaler",
            "random_state": self.random_state,
            "train_size": len(y_train),
            "val_size": len(y_val),
            **metrics11
        }
        results.append(res11)
        pipeline_cache["E11"] = pipeline_e11
        predictions_cache["E11"] = (y_pred11, y_score11)

        df_results = pd.DataFrame(results)

        # Save to CSV
        csv_path = REPORTS_DIR / "model_benchmark.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"[+] Saved experiment results CSV: {csv_path}")

        return df_results, pipeline_cache, predictions_cache


def plot_confusion_matrices(
    top_experiments: List[str],
    predictions_cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_val: np.ndarray
):
    """Generate confusion matrix figures for serious candidates."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for exp_id in top_experiments:
        if exp_id not in predictions_cache:
            continue
        y_pred, _ = predictions_cache[exp_id]
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["HAM (0)", "SPAM (1)"],
            yticklabels=["HAM (0)", "SPAM (1)"]
        )
        plt.title(f"Confusion Matrix: {exp_id}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        fig_path = FIGURES_DIR / f"confusion_matrix_{exp_id}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[+] Confusion matrix figure saved: {fig_path}")


def run_group_aware_cv(
    train_records: List[Dict[str, Any]],
    y_train: np.ndarray,
    groups_train: np.ndarray,
    exp_configs: List[Tuple[str, List[str], str, bool]],
    n_splits: int = 5,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Perform Stratified Group-Aware 5-Fold Cross Validation on TRAINING DATA ONLY.
    Guarantees no connected component group crosses CV fold boundaries.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    cv_results = {}

    for exp_id, feat_fams, model_name, scaling in exp_configs:
        f1_scores = []
        prec_scores = []
        rec_scores = []
        fpr_scores = []

        fold_details = []
        dummy_X = np.zeros(len(y_train))
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(dummy_X, y_train, groups_train), start=1):
            fold_train_records = [train_records[i] for i in train_idx]
            fold_val_records = [train_records[i] for i in val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]

            pipeline = SpamModelPipeline(
                feature_families=feat_fams,
                model_name=model_name,
                apply_scaling=scaling,
                random_state=seed
            )
            pipeline.fit(fold_train_records, y_fold_train)
            y_fold_pred = pipeline.predict(fold_val_records)

            cm = confusion_matrix(y_fold_val, y_fold_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            f1 = float(f1_score(y_fold_val, y_fold_pred, pos_label=1, zero_division=0))
            prec = float(precision_score(y_fold_val, y_fold_pred, pos_label=1, zero_division=0))
            rec = float(recall_score(y_fold_val, y_fold_pred, pos_label=1, zero_division=0))

            f1_scores.append(f1)
            prec_scores.append(prec)
            rec_scores.append(rec)
            fpr_scores.append(fpr)

            fold_details.append({
                "fold": fold,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "fpr": round(fpr, 4),
                "fp": int(fp),
                "fn": int(fn)
            })

        cv_results[exp_id] = {
            "folds": fold_details,
            "mean_f1": round(float(np.mean(f1_scores)), 4),
            "std_f1": round(float(np.std(f1_scores)), 4),
            "min_f1": round(float(np.min(f1_scores)), 4),
            "max_f1": round(float(np.max(f1_scores)), 4),
            "mean_precision": round(float(np.mean(prec_scores)), 4),
            "std_precision": round(float(np.std(prec_scores)), 4),
            "min_precision": round(float(np.min(prec_scores)), 4),
            "max_precision": round(float(np.max(prec_scores)), 4),
            "mean_recall": round(float(np.mean(rec_scores)), 4),
            "std_recall": round(float(np.std(rec_scores)), 4),
            "min_recall": round(float(np.min(rec_scores)), 4),
            "max_recall": round(float(np.max(rec_scores)), 4),
            "mean_fpr": round(float(np.mean(fpr_scores)), 4),
            "std_fpr": round(float(np.std(fpr_scores)), 4),
            "min_fpr": round(float(np.min(fpr_scores)), 4),
            "max_fpr": round(float(np.max(fpr_scores)), 4)
        }

    return cv_results

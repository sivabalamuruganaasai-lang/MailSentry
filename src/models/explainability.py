"""
Explainability Module for Frozen Email Spam Classifier (Level 8).

Provides global feature coefficient analysis and individual prediction explanation tools
for the frozen Candidate B (E11) LinearSVC model.

Maintains strict adherence to exact score reconstruction:
    decision_score(x) = intercept + sum_i (x_i * w_i)
Where x_i are MaxAbsScaler-scaled feature values and w_i are LinearSVC model weights.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from src.models.benchmark import SpamModelPipeline


def get_feature_family(feature_name: str) -> str:
    """Categorize a feature name into its feature family based on naming prefixes."""
    if feature_name.startswith("structural__"):
        return "structural"
    elif feature_name.startswith("url_feature__"):
        return "url"
    elif feature_name.startswith("char_tfidf__"):
        return "char"
    else:
        return "word"


def get_global_feature_importance(
    pipeline: SpamModelPipeline,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Extract global LinearSVC model coefficients mapped to feature names and families.

    Args:
        pipeline: Fitted SpamModelPipeline with LinearSVC.
        top_k: Number of top positive and negative features to return.

    Returns:
        Dict containing global model weights, intercept, top SPAM/HAM features,
        and family-wise breakdowns.
    """
    if not pipeline.is_fitted_:
        raise ValueError("Pipeline must be fitted before extracting global feature importance.")
    if pipeline.model_name != "LinearSVC":
        raise ValueError(f"Explainability module expects LinearSVC, got {pipeline.model_name}")

    feature_names = pipeline.feature_pipeline.get_feature_names_out()
    coefs = pipeline.model.model_.coef_[0]
    intercept = float(pipeline.model.model_.intercept_[0])

    if len(feature_names) != len(coefs):
        raise ValueError(f"Feature count mismatch: {len(feature_names)} names vs {len(coefs)} weights.")

    df = pd.DataFrame({
        "feature_name": feature_names,
        "feature_family": [get_feature_family(fn) for fn in feature_names],
        "coefficient": coefs
    })

    # Sort deterministically (primary: coefficient magnitude/direction, secondary: feature_name)
    df_positive = df[df["coefficient"] > 0].sort_values(
        by=["coefficient", "feature_name"], ascending=[False, True]
    )
    df_negative = df[df["coefficient"] < 0].sort_values(
        by=["coefficient", "feature_name"], ascending=[True, True]
    )

    top_positive = df_positive.head(top_k).to_dict(orient="records")
    top_negative = df_negative.head(top_k).to_dict(orient="records")

    family_summaries = {}
    for fam in df["feature_family"].unique():
        fam_df = df[df["feature_family"] == fam]
        fam_pos = fam_df[fam_df["coefficient"] > 0].sort_values(
            by=["coefficient", "feature_name"], ascending=[False, True]
        ).head(top_k).to_dict(orient="records")
        fam_neg = fam_df[fam_df["coefficient"] < 0].sort_values(
            by=["coefficient", "feature_name"], ascending=[True, True]
        ).head(top_k).to_dict(orient="records")

        family_summaries[fam] = {
            "feature_count": int(len(fam_df)),
            "min_coef": float(fam_df["coefficient"].min()) if len(fam_df) > 0 else 0.0,
            "max_coef": float(fam_df["coefficient"].max()) if len(fam_df) > 0 else 0.0,
            "mean_coef": float(fam_df["coefficient"].mean()) if len(fam_df) > 0 else 0.0,
            "top_positive": fam_pos,
            "top_negative": fam_neg
        }

    return {
        "intercept": intercept,
        "total_features": len(feature_names),
        "top_positive": top_positive,
        "top_negative": top_negative,
        "family_summaries": family_summaries,
        "causation_disclaimer": (
            "LinearSVC coefficients indicate linear associations learned from training data "
            "and do not imply causal mechanisms between features and spam labels."
        )
    }


def explain_individual_prediction(
    pipeline: SpamModelPipeline,
    email_record: Dict[str, Any],
    top_k: int = 10,
    threshold: float = 0.4013,
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """
    Calculate feature-level contributions (x_i * w_i) for a single email and verify
    exact score reconstruction against the raw LinearSVC decision function score.

    Args:
        pipeline: Fitted SpamModelPipeline with LinearSVC.
        email_record: Dict containing email fields (subject, plain_body, html_body, urls, etc.)
        top_k: Max number of top positive/negative contributions to list.
        threshold: Frozen decision threshold (default 0.4013).
        tolerance: Numerical tolerance for score reconstruction check (default 1e-5).

    Returns:
        Dict containing decision score, reconstructed score, intercept, verification status,
        prediction, top SPAM/HAM feature contributions, and family contribution breakdown.
    """
    if not pipeline.is_fitted_:
        raise ValueError("Pipeline must be fitted before generating explanations.")

    # 1. Transform email feature vector (includes scaling if enabled)
    X_feat = pipeline.transform_features([email_record])
    if issparse(X_feat):
        x_dense = X_feat.toarray()[0]
    else:
        x_dense = np.asarray(X_feat)[0]

    feature_names = pipeline.feature_pipeline.get_feature_names_out()
    coefs = pipeline.model.model_.coef_[0]
    intercept = float(pipeline.model.model_.intercept_[0])

    # 2. Compute exact contributions: c_i = x_i * w_i
    contributions = x_dense * coefs
    sum_contributions = float(np.sum(contributions))
    reconstructed_score = intercept + sum_contributions

    # 3. Obtain actual pipeline score
    raw_decision_score = float(pipeline.predict_score([email_record])[0])
    diff = abs(reconstructed_score - raw_decision_score)
    is_reconstructed_exact = bool(diff < tolerance)

    # 4. Decision rule: score >= threshold -> SPAM, else HAM
    is_spam = bool(raw_decision_score >= threshold)
    predicted_label = "SPAM" if is_spam else "HAM"

    # 5. Extract active features (non-zero values or non-zero contributions)
    active_indices = np.where(x_dense != 0)[0]

    active_records = []
    family_sums: Dict[str, float] = {}

    for idx in active_indices:
        fname = feature_names[idx]
        fam = get_feature_family(fname)
        val = float(x_dense[idx])
        w = float(coefs[idx])
        c = float(contributions[idx])

        active_records.append({
            "feature_name": fname,
            "feature_family": fam,
            "feature_value": val,
            "coefficient": w,
            "contribution": c
        })

        family_sums[fam] = family_sums.get(fam, 0.0) + c

    df_active = pd.DataFrame(active_records) if active_records else pd.DataFrame(
        columns=["feature_name", "feature_family", "feature_value", "coefficient", "contribution"]
    )

    if not df_active.empty:
        df_pos = df_active[df_active["contribution"] > 0].sort_values(
            by=["contribution", "feature_name"], ascending=[False, True]
        )
        df_neg = df_active[df_active["contribution"] < 0].sort_values(
            by=["contribution", "feature_name"], ascending=[True, True]
        )

        top_positive_contributions = df_pos.head(top_k).to_dict(orient="records")
        top_negative_contributions = df_neg.head(top_k).to_dict(orient="records")
    else:
        top_positive_contributions = []
        top_negative_contributions = []

    return {
        "raw_decision_score": raw_decision_score,
        "reconstructed_score": reconstructed_score,
        "intercept": intercept,
        "sum_contributions": sum_contributions,
        "diff": diff,
        "is_reconstructed_exact": is_reconstructed_exact,
        "tolerance": tolerance,
        "threshold": threshold,
        "is_spam": is_spam,
        "predicted_label": predicted_label,
        "total_active_features": len(active_records),
        "top_positive_contributions": top_positive_contributions,
        "top_negative_contributions": top_negative_contributions,
        "family_contributions_summary": family_sums
    }

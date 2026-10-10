"""
Streamlit Live Dashboard for Email Spam Classifier (Level 13).

Provides a professional, interactive demonstration interface for email spam classification,
three-tier operational routing, model explainability, and offline URL security analysis.

Strict Safeguards:
- 100% Frozen E11 model, 10,036 features, threshold 0.4013, three-tier bounds [-0.352, 0.601].
- Pre-verifies deployment artifact SHA-256 hash before loading.
- 0 network requests, 0 external API calls, 0 disk logging of raw email content.
- Enforces 2 MB max input size limit.
"""

import sys
import os
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.parser import parse_raw_email
from src.models.explainability import explain_individual_prediction
from src.features.url_security import analyze_email_urls, get_indicator_definitions

from src.models.inference import (
    PROJECT_ROOT,
    DEPLOYMENT_ARTIFACT,
    DEPLOYMENT_MANIFEST,
    EXPECTED_ARTIFACT_SHA256,
    MAX_INPUT_BYTES,
    BINARY_THRESHOLD,
    LOWER_THREE_TIER,
    UPPER_THREE_TIER,
    verify_artifact_hash,
    load_deployment_pipeline,
    evaluate_binary_classification,
    evaluate_three_tier_routing,
    build_feature_record
)

# Preset Demo Emails for Interactive Evaluation
DEMO_EMAILS = {
    "Select a Preset Demo Email...": "",
    "Legitimate Team Sync (HAM)": (
        "From: alex.smith@company.com\n"
        "To: team@company.com\n"
        "Subject: Project Launch Sync Tomorrow at 10am\n"
        "Date: Thu, 09 Oct 2026 10:00:00 +0000\n\n"
        "Hi Team,\n\n"
        "Please review the attached project roadmap for tomorrow's launch meeting.\n"
        "Let me know if you have any questions.\n\n"
        "Best regards,\nAlex"
    ),
    "Promotional HTML Newsletter (HTML FP Case)": (
        "From: news@techdigest.org\n"
        "To: subscriber@domain.com\n"
        "Subject: Weekly Tech Digest & Member Deals\n"
        "Content-Type: text/html\n\n"
        "<html><body>"
        "<h1>Weekly Tech Digest</h1>"
        "<p>Check out our latest articles and exclusive member offers!</p>"
        "<a href='http://techdigest.org/read'>Read Full Issue</a>"
        "</body></html>"
    ),
    "Phishing Scam with IP Host Link (SPAM)": (
        "From: security-alert@bank-update-service.net\n"
        "To: victim@user.com\n"
        "Subject: URGENT: Your Account Has Been Locked\n\n"
        "Dear Customer,\n\n"
        "We detected unauthorized access to your account. You must verify your credentials immediately.\n"
        "Click the link below to restore access:\n"
        "http://192.168.1.100/verify_login\n\n"
        "Failure to verify within 24 hours will result in permanent suspension."
    ),
    "Plain-Text Newsletter Spam (FN Case)": (
        "From: offer@marketdeals.com\n"
        "To: user@domain.com\n"
        "Subject: Exclusive Special Offer Now Available\n\n"
        "Special commercial newsletter update for our valued subscribers.\n"
        "Visit our web site to claim your discount today.\n"
        "Click http://marketdeals.com/claim to get started now."
    )
}


@st.cache_resource
def load_cached_pipeline():
    """Load and cache the verified fitted SpamModelPipeline artifact."""
    return load_deployment_pipeline()


def render_dashboard():
    st.set_page_config(
        page_title="Email Spam Classifier — Enterprise Dashboard",
        page_icon="🛡️",
        layout="wide"
    )

    # Title & Header Banner
    st.title("🛡️ Email Spam Classifier & Security Dashboard")
    st.markdown(
        "**Frozen Deployment Model (E11)** | `LinearSVC` | `10,036 Features` | "
        f"`Binary Threshold: {BINARY_THRESHOLD}` | `3-Tier Bounds: [{LOWER_THREE_TIER}, {UPPER_THREE_TIER}]`"
    )
    st.divider()

    # Load Model Pipeline with Security Verification
    try:
        pipeline = load_cached_pipeline()
    except Exception as e:
        st.error(f"❌ **Startup Error**: Model deployment bundle loading failed. {str(e)}")
        st.stop()

    # Sidebar Controls & Inputs
    st.sidebar.header("📥 Email Input & Options")
    input_mode = st.sidebar.radio("Input Source", ["Paste Raw Email Text", "Upload Email File (.eml, .txt)"])

    raw_bytes_to_parse = b""

    if input_mode == "Paste Raw Email Text":
        preset_choice = st.sidebar.selectbox("Load Preset Demo Email", list(DEMO_EMAILS.keys()))
        default_text = DEMO_EMAILS[preset_choice] if preset_choice != "Select a Preset Demo Email..." else ""

        pasted_text = st.text_area(
            "Paste Email Content (Headers & Body):",
            value=default_text,
            height=250,
            placeholder="From: sender@example.com\nSubject: Hello\n\nEmail body text here..."
        )

        if pasted_text:
            encoded_bytes = pasted_text.encode("utf-8", errors="replace")
            if len(encoded_bytes) > MAX_INPUT_BYTES:
                st.error(f"❌ **Input Size Limit Exceeded**: Input text is {len(encoded_bytes)/1024/1024:.2f} MB. Maximum allowed limit is 2 MB.")
                st.stop()
            raw_bytes_to_parse = encoded_bytes

    else:
        uploaded_file = st.file_uploader("Upload Email File (.eml, .txt)", type=["eml", "txt"])
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > MAX_INPUT_BYTES:
                st.error(f"❌ **File Size Limit Exceeded**: Uploaded file is {len(file_bytes)/1024/1024:.2f} MB. Maximum allowed limit is 2 MB.")
                st.stop()
            raw_bytes_to_parse = file_bytes

    # Main Analysis Trigger
    if not raw_bytes_to_parse or not raw_bytes_to_parse.strip():
        st.info("👋 Select a preset demo email from the sidebar, paste email text, or upload an `.eml` file to begin analysis.")
        st.stop()

    # Process Email
    with st.spinner("Processing email & extracting features..."):
        parsed_email = parse_raw_email(raw_bytes_to_parse)

        record_for_model = build_feature_record(parsed_email)

        explanation = explain_individual_prediction(pipeline, record_for_model, top_k=10, threshold=BINARY_THRESHOLD)
        score = explanation["raw_decision_score"]

        binary_res = evaluate_binary_classification(score)
        three_tier_res = evaluate_three_tier_routing(score)
        url_sec_res = analyze_email_urls(parsed_email.get("urls", []))

    # Top Metric KPI Display Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if binary_res["is_spam"]:
            st.error(f"### Binary: {binary_res['label']}\n**Score $\\ge {BINARY_THRESHOLD}$**")
        else:
            st.success(f"### Binary: {binary_res['label']}\n**Score $< {BINARY_THRESHOLD}$**")

    with col2:
        st.metric(
            label="Raw Decision Score S(x)",
            value=f"{score:+.4f}",
            delta=f"{score - BINARY_THRESHOLD:+.4f} vs threshold",
            delta_color="inverse"
        )

    with col3:
        if three_tier_res["tier"] == "AUTO HAM":
            st.success(f"### Routing: {three_tier_res['tier']}\n{three_tier_res['action']}")
        elif three_tier_res["tier"] == "REVIEW QUEUE":
            st.warning(f"### Routing: {three_tier_res['tier']}\n{three_tier_res['action']}")
        else:
            st.error(f"### Routing: {three_tier_res['tier']}\n{three_tier_res['action']}")

    with col4:
        r_level = url_sec_res["heuristic_risk_level"]
        if r_level == "HIGH":
            st.error(f"### URL Risk: {r_level}\n{url_sec_res['flagged_url_count']} Flagged Links")
        elif r_level == "MEDIUM":
            st.warning(f"### URL Risk: {r_level}\n{url_sec_res['flagged_url_count']} Flagged Links")
        else:
            st.success(f"### URL Risk: {r_level}\n{url_sec_res['flagged_url_count']} Flagged Links")

    st.divider()

    # Detailed Analytical Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Model Explainability & Feature Pushers", "🔗 URL Security Analysis", "📧 Email Metadata & Headers"])

    with tab1:
        st.subheader("LinearSVC Feature Contribution Breakdown")
        st.caption("Exact score reconstruction: $S(x) = \\text{intercept} + \\sum (x_i \\cdot w_i)$")

        e_col1, e_col2 = st.columns(2)

        with e_col1:
            st.write("#### 🔴 Top SPAM Pushers (Positive Contributions)")
            pos_contribs = explanation["top_positive_contributions"]
            if pos_contribs:
                df_pos = pd.DataFrame(pos_contribs)[["feature_name", "feature_family", "feature_value", "coefficient", "contribution"]]
                st.dataframe(df_pos.style.format({"feature_value": "{:.4f}", "coefficient": "{:+.4f}", "contribution": "{:+.4f}"}), use_container_width=True)
            else:
                st.info("No positive feature contributions for this email.")

        with e_col2:
            st.write("#### 🟢 Top HAM Pushers (Negative Contributions)")
            neg_contribs = explanation["top_negative_contributions"]
            if neg_contribs:
                df_neg = pd.DataFrame(neg_contribs)[["feature_name", "feature_family", "feature_value", "coefficient", "contribution"]]
                st.dataframe(df_neg.style.format({"feature_value": "{:.4f}", "coefficient": "{:+.4f}", "contribution": "{:+.4f}"}), use_container_width=True)
            else:
                st.info("No negative feature contributions for this email.")

        st.info("💡 **Explainability Disclaimer**: Feature contributions reflect linear associations learned by the LinearSVC model from training data; they describe how the model calculated score $S(x)$ and do NOT constitute physical proof of email safety or maliciousness.")

    with tab2:
        st.subheader("Offline Static URL Security Analysis")
        st.caption("Heuristic syntactic risk indicators extracted without performing network requests or DNS lookups.")

        u_col1, u_col2 = st.columns([1, 2])

        with u_col1:
            st.metric("Total Extracted URLs", url_sec_res["analyzed_url_count"])
            st.metric("Flagged Anomaly URLs", url_sec_res["flagged_url_count"])
            st.metric("Overall Risk Level", url_sec_res["heuristic_risk_level"])

        with u_col2:
            st.write("#### Detected Security Indicators")
            if url_sec_res["observed_indicators"]:
                for ind in url_sec_res["observed_indicators"]:
                    desc = url_sec_res["indicator_descriptions"].get(ind, "")
                    st.warning(f"**{ind}**: {desc}")
            else:
                st.success("No heuristic URL risk indicators observed.")

        if url_sec_res["url_details"]:
            st.write("#### Per-URL Details")
            for idx, u_det in enumerate(url_sec_res["url_details"], 1):
                with st.expander(f"URL #{idx}: {u_det['url'][:80]}..."):
                    st.json(u_det)

        st.warning("⚠️ **URL Security Disclaimer**: Heuristic structural indicators quantify static pattern risks and syntax anomalies. They do NOT prove whether a URL or domain is malicious or safe.")

    with tab3:
        st.subheader("Extracted Email Headers & Structure")
        h_col1, h_col2 = st.columns(2)

        with h_col1:
            st.write(f"**Subject**: {parsed_email.get('subject', '(None)')}")
            st.write(f"**From**: {parsed_email.get('from', '(None)')}")
            st.write(f"**To**: {parsed_email.get('to', '(None)')}")
            st.write(f"**Date**: {parsed_email.get('date', '(None)')}")

        with h_col2:
            st.write(f"**Sender Domain**: `{parsed_email.get('sender_domain', 'N/A')}`")
            st.write(f"**Content-Type**: `{parsed_email.get('content_type', 'N/A')}`")
            st.write(f"**HTML Present**: `{parsed_email.get('has_html', False)}`")
            st.write(f"**Attachments**: `{parsed_email.get('attachment_count', 0)}`")

        st.write("#### Body Content Preview")
        st.text_area("Extracted Plain Text Body Preview:", value=parsed_email.get("plain_body", "")[:1000], height=150, disabled=True)


if __name__ == "__main__":
    render_dashboard()

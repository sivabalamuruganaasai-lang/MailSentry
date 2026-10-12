# MailSentry: Vercel Deployment & Operation Guide

This document describes the architecture, deployment procedure, security safeguards, and operational settings for deploying the MailSentry Email Spam & Security Analysis engine to Vercel.

---

## 1. Architecture Overview

MailSentry uses a decoupled, serverless single-page application (SPA) architecture designed to operate within Vercel's serverless boundaries:

```
[ Client Browser ]
        │
        ├── (Static Assets) ───────►  Vercel Edge CDN (public/index.html, style.css, app.js)
        │
        └── (HTTPS API Calls) ─────►  Vercel Python Serverless Functions
                                       ├── /api/health   (api/health.py)
                                       └── /api/predict  (api/predict.py)
                                               │
                                               ▼
                                    Frozen LinearSVC Model (models/deployment_e11.joblib)
                                    SHA-256: 7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06
```

### Key Differences from Local Streamlit
| Dimension | Streamlit (`src/ui/app.py`) | Vercel Web App (`public/` + `api/`) |
| :--- | :--- | :--- |
| **Hosting Model** | Persistent stateful Python process | Decoupled static SPA + stateless Serverless Lambdas |
| **Dependencies** | Full (`requirements-dev.txt`: `streamlit`, `matplotlib`, `seaborn`, `pytest`) | Lean (`requirements.txt`: `scikit-learn`, `joblib`, `numpy`, `pandas`, `scipy`) |
| **Bundle Size** | ~383 MB uncompressed | ~247 MB uncompressed (under Vercel 250 MB limit) |
| **Concurrency** | Shared memory server | Ephemeral auto-scaling instances |
| **Privacy / Storage** | Local host only | Cloud serverless; zero disk writes, in-memory processing only |

---

## 2. Pre-Deployment Configuration (`vercel.json` & `.vercelignore`)

### Deployment Configuration (`vercel.json`)
The deployment configuration is maintained in `vercel.json`:
- **Static Output Isolation (`outputDirectory: "public"`)**: Restricts static web serving exclusively to `public/`, preventing unauthorized web access to internal repository assets (`models/`, `src/`, `data/`).
- **Serverless File Inclusion (`includeFiles: "models/**"`)**: Guarantees that `models/deployment_e11.joblib` and `models/deployment_manifest.json` are packaged inside the serverless lambda.
- **Artifact Exclusion (`excludeFiles`)**: Strips training datasets (`data/**`), analysis reports, figures, and tests from the function bundle to keep the uncompressed deployment artifact lean.
- **Max Duration**: `30s` (well above typical ~300ms warm inference latency).
- **Function Memory Allocation**: In modern Vercel, function memory is configured in the **Vercel Project Dashboard** under *Settings > Functions* (recommended: `1024 MB` for optimal scikit-learn cold-start deserialization).
- **Security Headers Injected**:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Permissions-Policy: camera=(), microphone=(), geolocation=()`
- **Routing**: API endpoints routed to `api/*.py`, static assets served directly from `public/`.

### Deployment Filtering (`.vercelignore`)
A `.vercelignore` file at the repository root prevents Vercel from mistaking local development tools for deployment services:
- **Excludes `src/ui/app.py` & `/.streamlit/`**: Ensures Vercel does not detect Streamlit as a separate service (preventing the multi-service `vercel.json` prompt).
- **Excludes `/requirements-dev.txt`, `/tests/`, `/reports/`, and `/data/`**: Prevents development assets and datasets from bloating the deployment artifact, while leading slashes guarantee internal package modules like `src/data/` remain included in the serverless bundle.

---

## 3. GitHub Import & Vercel Project Setup

### Step 1: Push Repository to GitHub
Ensure all deployment configuration files are committed and pushed to `main`:
```bash
git add requirements.txt requirements-dev.txt .vercelignore DEPLOYMENT_VERCEL.md
git commit -m "fix(vercel): configure single-project serverless deployment and .vercelignore"
git push origin main
```

### Step 2: Import into Vercel Dashboard
1. Log in to [Vercel](https://vercel.com).
2. Click **Add New...** > **Project**.
3. Select your GitHub repository (`MailSentry`).
4. In **Project Settings**:
   - **Framework Preset**: Select **`Other`** (crucial: disables auto-detected "Services" mode).
   - **Root Directory**: `./` (leave default repository root; do not change to `api/` or `public/`).
   - **Build Command**: Leave empty / toggle Override OFF (vanilla HTML/CSS/JS requires no build step).
   - **Output Directory**: Toggle Override ON and enter **`public`** (or leave empty to inherit `"outputDirectory": "public"` from `vercel.json`).
   - **Install Command**: Leave default (Vercel automatically installs the minimal runtime from root `requirements.txt`).
5. **Environment Variables**:
   - None required. MailSentry runs completely self-contained with offline heuristics and zero external API keys.
6. Click **Deploy**.

---

## 4. API Endpoints & Contract

### 1. `GET /api/health`
Cryptographically verifies `models/deployment_e11.joblib` SHA-256 against the pinned expected hash before confirming operational readiness. Never exposes host filesystem paths or stack traces.

**Response Schema (`200 OK`)**:
```json
{
  "status": "HEALTHY",
  "model_artifact": "deployment_e11.joblib",
  "model_hash_verified": true,
  "binary_threshold": 0.4013,
  "routing_corridor": {
    "lower_bound": -0.3520,
    "upper_bound": 0.6010
  },
  "max_input_bytes": 2097152,
  "offline_url_heuristics": true,
  "network_calls": false
}
```

### 2. `POST /api/predict`
Accepts email text via JSON (`{"email_text": "..."}`), raw payload, or multipart file uploads (`.eml`, `.txt`).

**Enforcements**:
- **Size Limit**: Enforces a strict server-side **2.0 MiB** (`2,097,152 bytes`) limit before parsing. Payloads exceeding this return `413 Payload Too Large`.
- **File Type Validation**: Only `.eml` and `.txt` files are accepted. Unsupported extensions (e.g., `.pdf`, `.exe`) return `422 Unprocessable Entity` with `UNSUPPORTED_FILE_TYPE`.
- **Pre-Inference Email Validation**: All inputs undergo structural and communicative plausibility checks (`src/data/email_validator.py`) before model invocation. Empty inputs return `400 Bad Request`. Arbitrary non-email text, greetings alone, repeated characters, and generic paragraphs return `422 Unprocessable Entity` (`INVALID_EMAIL_INPUT` / `INSUFFICIENT_CONTENT`) without invoking the model or assigning bogus classifications.
- **In-Memory Only**: Payloads are never saved to `/tmp` or persistent storage.
- **Privacy & Logging**: Email bodies, subject lines, header values, and URLs are never logged to console or logs.
- **Safe Heuristics**: URL safety indicators are purely heuristic and offline; the engine explicitly states that clean heuristic checks do not guarantee an external link is safe.

---

## 5. Security & Privacy Considerations

### Public Cloud Processing Disclosure
> [!IMPORTANT]
> When deployed to a public URL on Vercel, submitted email content is processed by a third-party hosted cloud infrastructure (Vercel / AWS Lambda backend). Users must be advised not to submit confidential, classified, proprietary, or sensitive personal emails unless access control (such as Vercel Deployment Protection or SSO) is enabled.

### Rate Limiting
- **In-Memory Lambda Limiting**: `api/predict.py` implements an in-memory IP rate limiter (30 requests/minute per client IP).
- **Vercel Edge Protection**: For enterprise scale or DDoS mitigation on public URLs, enable **Vercel Attack Challenge Mode** or configure Vercel Edge Firewall rate limiting rules in the Vercel Dashboard under *Settings > Security*.

### HTML Injection & XSS Prevention
- All telemetry, headers, subject lines, and URL strings rendered in the static frontend (`public/app.js`) are sanitized via HTML entity escaping (`escapeHtml()`) before DOM insertion. Raw email HTML is never rendered as executable markup.

---

## 6. Local Testing & Verification

To verify the Vercel serverless API handlers locally using the automated suite:
```bash
python -m pytest tests/test_vercel_api.py tests/test_email_validator.py -v
```

The test suites validate:
1. Cryptographic health checks & SHA-256 integrity
2. Presets and API documentation reflection
3. High-confidence HAM inference and AUTO HAM routing
4. Boundary HAM inference and REVIEW QUEUE corridor routing
5. High-confidence SPAM inference and AUTO SPAM routing
6. Multipart `.eml` and `.txt` file uploads
7. Unsupported file extension rejection (`.pdf`, `.exe` -> HTTP 422)
8. Empty and whitespace-only input rejection (HTTP 400)
9. Pre-inference validation rejecting generic paragraphs, `hello world`, and punctuation (HTTP 422)
10. Preservation of frozen model scores and routing on valid inputs
11. 2 MiB boundary enforcement (HTTP 413)
12. Rate limiting triggers (HTTP 429)
13. Classifier HAM with independent flagged URL indicator alerts
14. Traceback/stack leak suppression on malformed payloads

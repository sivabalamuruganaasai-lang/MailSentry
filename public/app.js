/* ==========================================================================
   MailSentry SOC Threat Intelligence - Vercel Frontend Logic (app.js)
   ========================================================================== */

let currentMode = 'paste'; // 'paste' or 'upload'
let uploadedFile = null;
let presetData = {};

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

document.addEventListener('DOMContentLoaded', () => {
  initApp();
  setupEventListeners();
});

async function initApp() {
  const apiStatusText = document.getElementById('api-status-text');

  try {
    const healthRes = await fetch('/api/health');
    if (healthRes.ok) {
      const healthData = await healthRes.json();
      apiStatusText.textContent = `● API ONLINE (SHA-256 VERIFIED)`;
      const shaCode = document.getElementById('artifact-sha-code');
      if (shaCode && healthData.artifact_sha256) {
        shaCode.textContent = healthData.artifact_sha256;
      }
    } else {
      apiStatusText.textContent = `● API DEGRADED`;
    }
  } catch (err) {
    apiStatusText.textContent = `● API OFFLINE`;
  }

  // Load Presets from API
  try {
    const presetsRes = await fetch('/api/predict');
    if (presetsRes.ok) {
      const data = await presetsRes.json();
      if (data.preset_payloads) {
        presetData = data.preset_payloads;
      }
    }
  } catch (err) {
    console.warn('Could not load dynamic presets:', err);
  }
}

function setupEventListeners() {
  const presetSelect = document.getElementById('preset-select');
  const emailTextarea = document.getElementById('email-textarea');
  const textCharsCount = document.getElementById('text-chars-count');
  const tabPasteBtn = document.getElementById('tab-paste-btn');
  const tabUploadBtn = document.getElementById('tab-upload-btn');
  const pasteContainer = document.getElementById('paste-container');
  const uploadContainer = document.getElementById('upload-container');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const fileInfoBox = document.getElementById('file-info-box');
  const removeFileBtn = document.getElementById('remove-file-btn');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');

  const replaceFileBtn = document.getElementById('replace-file-btn');

  // Textarea telemetry
  emailTextarea.addEventListener('input', () => {
    const text = emailTextarea.value;
    const lines = text ? text.split('\n').length : 0;
    textCharsCount.textContent = `${text.length.toLocaleString()} characters • ${lines.toLocaleString()} lines`;
    hideError();
    resetToStandby();
  });

  // Presets dropdown
  presetSelect.addEventListener('change', (e) => {
    const chosen = e.target.value;
    hideError();
    resetToStandby();

    if (chosen && presetData[chosen]) {
      emailTextarea.value = presetData[chosen];
      switchToMode('paste');
      const lines = emailTextarea.value.split('\n').length;
      textCharsCount.textContent = `${emailTextarea.value.length.toLocaleString()} characters • ${lines.toLocaleString()} lines`;
    } else {
      emailTextarea.value = '';
      textCharsCount.textContent = '0 characters • 0 lines';
    }
  });

  // Mode tabs
  tabPasteBtn.addEventListener('click', () => switchToMode('paste'));
  tabUploadBtn.addEventListener('click', () => switchToMode('upload'));

  function switchToMode(mode) {
    currentMode = mode;
    hideError();
    resetToStandby();

    if (mode === 'paste') {
      tabPasteBtn.classList.add('active');
      tabUploadBtn.classList.remove('active');
      pasteContainer.style.display = 'flex';
      uploadContainer.style.display = 'none';
    } else {
      tabUploadBtn.classList.add('active');
      tabPasteBtn.classList.remove('active');
      uploadContainer.style.display = 'flex';
      pasteContainer.style.display = 'none';
    }
  }

  // Keyboard accessibility for label dropzone
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  // Drag & drop events on dropzone
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  // File input change handler (native picker on desktop & mobile)
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function clearFileInput() {
    uploadedFile = null;
    fileInput.value = '';
    fileInfoBox.style.display = 'none';
    dropzone.style.display = 'block';
  }

  function handleSelectedFile(file) {
    hideError();
    resetToStandby();

    if (!file) return;

    // Validate file extension (.eml or .txt)
    const ext = file.name && file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : '';
    if (ext !== 'eml' && ext !== 'txt') {
      showError(`Unsupported File Type (.${escapeHtml(ext || 'unknown')}): Only .eml and .txt email files are accepted.`);
      clearFileInput();
      return;
    }

    // Validate size limit (2.0 MiB)
    const maxBytes = 2 * 1024 * 1024;
    if (file.size > maxBytes) {
      showError(`File Size Limit Exceeded: ${(file.size / (1024 * 1024)).toFixed(2)} MB exceeds maximum limit of 2.0 MiB.`);
      clearFileInput();
      return;
    }

    // Validate non-empty file
    if (file.size === 0) {
      showError('Selected file is empty (0 bytes). Please choose a valid email file.');
      clearFileInput();
      return;
    }

    uploadedFile = file;
    document.getElementById('loaded-filename').textContent = file.name;
    document.getElementById('loaded-filesize').textContent = `${(file.size / 1024).toFixed(1)} KB (Ready for Analysis)`;
    dropzone.style.display = 'none';
    fileInfoBox.style.display = 'flex';
  }

  // Replace file button
  if (replaceFileBtn) {
    replaceFileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  // Remove file button
  removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFileInput();
    hideError();
    resetToStandby();
  });

  // Reset button
  resetBtn.addEventListener('click', () => {
    presetSelect.value = '';
    emailTextarea.value = '';
    textCharsCount.textContent = '0 characters • 0 lines';
    clearFileInput();
    hideError();
    switchToMode('paste');
    resetToStandby();
  });


  // Run Threat Analysis button
  runBtn.addEventListener('click', executeAnalysis);

  // Tabs navigation inside results
  setupTabs();
}

function setupTabs() {
  const tabs = [
    { btn: 'tab-feat-btn', content: 'tab-feat-content' },
    { btn: 'tab-head-btn', content: 'tab-head-content' },
    { btn: 'tab-mode-btn', content: 'tab-mode-content' }
  ];

  tabs.forEach(t => {
    const btnEl = document.getElementById(t.btn);
    btnEl.addEventListener('click', () => {
      tabs.forEach(other => {
        document.getElementById(other.btn).classList.remove('active');
        document.getElementById(other.content).style.display = 'none';
      });
      btnEl.classList.add('active');
      document.getElementById(t.content).style.display = 'block';
    });
  });
}

function showError(msg, title = 'Error') {
  const errBox = document.getElementById('input-error-box');
  errBox.innerHTML = `<strong>⚠️ ${escapeHtml(title)}:</strong> ${escapeHtml(msg)}`;
  errBox.style.display = 'block';
}

function hideError() {
  const errBox = document.getElementById('input-error-box');
  errBox.style.display = 'none';
  errBox.innerHTML = '';
}

function resetToStandby() {
  document.getElementById('standby-container').style.display = 'flex';
  document.getElementById('evaluated-container').style.display = 'none';
  document.getElementById('eval-header-status').textContent = 'STANDBY';
}

async function executeAnalysis() {
  hideError();

  let bodyData;
  let headers = {};

  if (currentMode === 'paste') {
    const textVal = document.getElementById('email-textarea').value.trim();
    if (!textVal) {
      showError('Please paste email content or select a preset before running analysis.', 'Input Required');
      document.getElementById('eval-header-status').textContent = 'INVALID INPUT';
      return;
    }
    bodyData = JSON.stringify({ email_text: textVal });
    headers['Content-Type'] = 'application/json';
  } else {
    if (!uploadedFile) {
      showError('Please select or drop a .eml or .txt file before running analysis.', 'File Required');
      document.getElementById('eval-header-status').textContent = 'INVALID INPUT';
      return;
    }

    // Pre-flight validation on selected file
    const ext = uploadedFile.name && uploadedFile.name.includes('.') ? uploadedFile.name.split('.').pop().toLowerCase() : '';
    if (ext !== 'eml' && ext !== 'txt') {
      showError(`Only .eml and .txt email files are accepted (got .${ext || 'unknown'}).`, 'Unsupported File Type');
      document.getElementById('eval-header-status').textContent = 'INVALID INPUT';
      return;
    }

    const maxBytes = 2 * 1024 * 1024;
    if (uploadedFile.size > maxBytes) {
      showError(`${(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB exceeds maximum limit of 2.0 MiB.`, 'File Size Limit Exceeded');
      document.getElementById('eval-header-status').textContent = 'INVALID INPUT';
      return;
    }

    if (uploadedFile.size === 0) {
      showError('The selected file is empty (0 bytes). Please choose a valid email file.', 'Empty File');
      document.getElementById('eval-header-status').textContent = 'INVALID INPUT';
      return;
    }

    const formData = new FormData();
    formData.append('email_file', uploadedFile);
    bodyData = formData;
  }

  // Trigger loading UI
  const runBtn = document.getElementById('run-btn');
  const spinner = document.getElementById('run-btn-spinner');
  const btnText = document.getElementById('run-btn-text');

  runBtn.disabled = true;
  spinner.style.display = 'inline';
  btnText.textContent = 'Evaluating Neural Core...';
  document.getElementById('eval-header-status').textContent = 'ANALYZING...';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: headers,
      body: bodyData
    });

    const contentType = response.headers.get('content-type') || '';
    let resJson = null;

    if (contentType.includes('application/json')) {
      try {
        resJson = await response.json();
      } catch (parseErr) {
        resJson = null;
      }
    }

    if (!response.ok) {
      resetToStandby();
      if (response.status === 400 || response.status === 422) {
        document.getElementById('eval-header-status').textContent = 'INVALID INPUT';
        const title = (resJson && resJson.code === 'INSUFFICIENT_CONTENT') ? 'Insufficient email content' : 'Invalid input';
        const message = (resJson && (resJson.message || resJson.error)) || 'Please enter a plausible email or email body with enough content to analyze.';
        showError(message, title);
      } else if (response.status === 413) {
        document.getElementById('eval-header-status').textContent = 'PAYLOAD TOO LARGE';
        showError('Submitted payload exceeds the maximum allowed limit of 2.0 MiB.', 'Payload Too Large');
      } else if (response.status === 429) {
        document.getElementById('eval-header-status').textContent = 'RATE LIMITED';
        showError('Too many requests submitted. Please wait a moment before trying again.', 'Rate Limit Exceeded');
      } else {
        document.getElementById('eval-header-status').textContent = 'SERVER ERROR';
        const serverMsg = (resJson && (resJson.error || resJson.message)) || `Server error (HTTP ${response.status}).`;
        showError(serverMsg, 'Server Error');
      }
      return;
    }

    if (!resJson) {
      resetToStandby();
      document.getElementById('eval-header-status').textContent = 'RESPONSE ERROR';
      showError('Server returned an unexpected non-JSON response.', 'Response Error');
      return;
    }

    renderAnalysisResults(resJson);
  } catch (err) {
    resetToStandby();
    document.getElementById('eval-header-status').textContent = 'CONNECTION ERROR';
    showError(`Could not reach analysis engine (${err.message}). Check your network connection.`, 'Network Error');
  } finally {
    runBtn.disabled = false;
    spinner.style.display = 'none';
    btnText.textContent = '⚡ Run Threat Analysis';
  }
}

function renderAnalysisResults(data) {
  const decision = data.decision;
  const urls = data.url_security;
  const explain = data.explainability;
  const telemetry = data.telemetry;

  document.getElementById('standby-container').style.display = 'none';
  document.getElementById('evaluated-container').style.display = 'flex';
  document.getElementById('eval-header-status').textContent = decision.is_spam ? 'THREAT DETECTED' : 'EVALUATION COMPLETE';

  // 1. Primary Verdict Hero Banner
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictSub = document.getElementById('verdict-sub');
  const advisoryCallout = document.getElementById('advisory-callout');
  const signedMarginVal = document.getElementById('signed-margin-val');
  const absDistVal = document.getElementById('abs-distance-val');

  signedMarginVal.textContent = decision.signed_margin_formatted;
  absDistVal.textContent = `|S(x) - τ| = ${decision.absolute_distance_to_threshold.toFixed(4)}`;

  if (decision.is_spam) {
    verdictBanner.className = 'verdict-hero verdict-hero-spam';
    verdictTitle.className = 'verdict-title verdict-title-spam';
    verdictTitle.textContent = '🚨 SPAM DETECTED';
    verdictSub.textContent = `Binary Classification: POSITIVE • Score S(x) ≥ ${decision.binary_threshold.toFixed(4)}`;
    signedMarginVal.style.color = 'var(--red-threat)';
    advisoryCallout.style.display = 'none';
  } else {
    signedMarginVal.style.color = 'var(--green-safe)';
    if (urls.flagged_count > 0) {
      verdictBanner.className = 'verdict-hero verdict-hero-advisory';
      verdictTitle.className = 'verdict-title verdict-title-advisory';
      verdictTitle.textContent = '⚠️ CLASSIFIED HAM • URL SECURITY ADVISORY';
      verdictSub.textContent = `Binary Classification: NEGATIVE • Score S(x) < ${decision.binary_threshold.toFixed(4)} • Offline URL Heuristics: ${urls.flagged_count} Suspicious Link(s) Flagged`;

      advisoryCallout.style.display = 'block';
      advisoryCallout.innerHTML = `<strong>⚠️ Independent Security Advisory:</strong> The linear classifier categorized this message as <strong>HAM</strong> based on lexical/structural weights (S(x) = ${decision.raw_decision_score > 0 ? '+' : ''}${decision.raw_decision_score.toFixed(4)} &lt; ${decision.binary_threshold.toFixed(4)}), but independent offline URL heuristics flagged <strong>${urls.flagged_count}</strong> link(s) with anomalous or suspicious structure. <em>This message is not verified safe.</em> The binary classification and routing policy remain unchanged, but manual analyst caution is strongly advised.`;
    } else {
      verdictBanner.className = 'verdict-hero verdict-hero-ham';
      verdictTitle.className = 'verdict-title verdict-title-ham';
      verdictTitle.textContent = '✅ CLASSIFIED HAM';
      verdictSub.textContent = `Binary Classification: NEGATIVE • Score S(x) < ${decision.binary_threshold.toFixed(4)}`;
      advisoryCallout.style.display = 'none';
    }
  }

  // 2. Metrics Grid
  const tileScoreVal = document.getElementById('tile-score-val');
  tileScoreVal.textContent = `${decision.raw_decision_score >= 0 ? '+' : ''}${decision.raw_decision_score.toFixed(4)}`;
  tileScoreVal.style.color = decision.is_spam ? 'var(--red-threat)' : 'var(--green-safe)';
  document.getElementById('tile-score-sub').textContent = `Signed margin: ${decision.signed_margin_formatted} vs τ`;

  document.getElementById('tile-thresh-sub').textContent = decision.is_spam
    ? 'Boundary: S(x) ≥ +0.4013 → SPAM'
    : 'Boundary: S(x) < +0.4013 → HAM';

  document.getElementById('tile-dist-val').textContent = decision.absolute_distance_to_threshold.toFixed(4);
  document.getElementById('tile-dist-sub').textContent = `SVM zero-plane: |S(x)| = ${decision.svm_hyperplane_distance.toFixed(4)}`;

  // 3. Operational Routing Dispatch
  const tier = decision.routing_tier;
  const isAutoHam = (tier === 'AUTO HAM');
  const isReview = (tier === 'REVIEW QUEUE');
  const isAutoSpam = (tier === 'AUTO SPAM');

  const tierPill = document.getElementById('route-tier-pill');
  tierPill.textContent = tier;
  tierPill.className = isAutoSpam ? 'pill pill-threat' : (isReview ? 'pill pill-warn' : 'pill pill-safe');

  document.getElementById('route-tile-ham').className = `route-tile ${isAutoHam ? 'active-ham' : ''}`;
  document.getElementById('route-badge-ham').className = `pill ${isAutoHam ? 'pill-safe' : 'pill-dim'}`;
  document.getElementById('route-badge-ham').textContent = isAutoHam ? '● ACTIVE' : 'INACTIVE';

  document.getElementById('route-tile-rev').className = `route-tile ${isReview ? 'active-review' : ''}`;
  document.getElementById('route-badge-rev').className = `pill ${isReview ? 'pill-warn' : 'pill-dim'}`;
  document.getElementById('route-badge-rev').textContent = isReview ? '● ACTIVE' : 'INACTIVE';

  document.getElementById('route-tile-spm').className = `route-tile ${isAutoSpam ? 'active-spam' : ''}`;
  document.getElementById('route-badge-spm').className = `pill ${isAutoSpam ? 'pill-threat' : 'pill-dim'}`;
  document.getElementById('route-badge-spm').textContent = isAutoSpam ? '● ACTIVE' : 'INACTIVE';

  // Position pin in [-1.5, +1.5]
  const minVis = -1.5;
  const maxVis = 1.5;
  const scoreClamped = Math.max(minVis, Math.min(maxVis, decision.raw_decision_score));
  const pinPct = Math.max(2, Math.min(98, ((scoreClamped - minVis) / (maxVis - minVis)) * 100));
  document.getElementById('corridor-pin').style.left = `${pinPct}%`;
  document.getElementById('corridor-active-score').textContent = `Active S(x) = ${decision.raw_decision_score >= 0 ? '+' : ''}${decision.raw_decision_score.toFixed(4)}`;

  document.getElementById('route-explanation-text').innerHTML = `<strong>Routing Disposition:</strong> ${escapeHtml(decision.routing_description)}<br><span style="color: var(--text-dim); font-size: 0.68rem; margin-top: 0.2rem; display: block;">Architectural Note: Binary classification (&tau; = 0.4013) and routing policy are separate decision layers. Borderline messages are held in Review Queue to prevent premature delivery or false quarantines.</span>`;

  // 4. Extracted URLs Card
  document.getElementById('urls-card-title').textContent = `Extracted URLs & Offline Heuristics (${urls.analyzed_count} detected • ${urls.unique_count} unique)`;
  const urlPill = document.getElementById('url-risk-pill');
  urlPill.textContent = `${urls.heuristic_risk_level} RISK`;
  urlPill.className = (urls.heuristic_risk_level === 'HIGH' || urls.heuristic_risk_level === 'CRITICAL')
    ? 'pill pill-threat'
    : (urls.heuristic_risk_level === 'MEDIUM' ? 'pill pill-warn' : 'pill pill-safe');

  const urlListEl = document.getElementById('url-items-list');
  urlListEl.innerHTML = '';

  if (urls.url_details && urls.url_details.length > 0) {
    urls.url_details.forEach((u, i) => {
      const card = document.createElement('div');
      card.className = 'url-card';

      let tagsHtml = '';
      if (u.indicators && u.indicators.length > 0) {
        tagsHtml = u.indicators.map(ind => `<span class="ind-tag">${escapeHtml(ind)}</span>`).join('');
      } else {
        tagsHtml = '<span class="pill pill-safe" style="font-size: 0.65rem;">NO_INDICATORS</span>';
      }

      const statusBadge = u.flagged
        ? '<span class="pill pill-threat">🚨 FLAGGED</span>'
        : '<span class="pill pill-safe">✅ BENIGN</span>';

      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-weight: 700; font-size: 0.74rem; color: var(--cyan-accent); font-family: var(--font-mono);">URL #${i + 1}</span>
          ${statusBadge}
        </div>
        <div class="url-code">${escapeHtml(u.url)}</div>
        <div style="display: flex; gap: 0.8rem; font-size: 0.68rem; color: var(--text-dim); font-family: var(--font-mono); margin-bottom: 0.3rem;">
          <span>SCHEME: <code>${escapeHtml(u.scheme || 'N/A')}</code></span>
          <span>HOST: <code>${escapeHtml(u.hostname || 'N/A')}</code></span>
        </div>
        <div>${tagsHtml}</div>
      `;
      urlListEl.appendChild(card);
    });
  } else {
    urlListEl.innerHTML = '<div style="font-size: 0.75rem; color: var(--text-dim); padding: 0.4rem 0;">No URLs or hyperlinks were extracted from this email payload.</div>';
  }

  // 5. Drill-Down Tabs
  const topSpamTbody = document.getElementById('top-spam-tbody');
  topSpamTbody.innerHTML = '';
  if (explain.top_spam_features && explain.top_spam_features.length > 0) {
    explain.top_spam_features.forEach(f => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="color: #FDA4AF;">${escapeHtml(f.feature_name)}</td>
        <td style="color: var(--text-muted);">${f.coefficient >= 0 ? '+' : ''}${f.coefficient.toFixed(4)}</td>
        <td style="color: var(--red-threat); font-weight: 700;">${f.contribution >= 0 ? '+' : ''}${f.contribution.toFixed(4)}</td>
      `;
      topSpamTbody.appendChild(tr);
    });
  } else {
    topSpamTbody.innerHTML = '<tr><td colspan="3" style="color: var(--text-dim);">No positive feature contributions</td></tr>';
  }

  const topHamTbody = document.getElementById('top-ham-tbody');
  topHamTbody.innerHTML = '';
  if (explain.top_ham_features && explain.top_ham_features.length > 0) {
    explain.top_ham_features.forEach(f => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="color: #6EE7B7;">${escapeHtml(f.feature_name)}</td>
        <td style="color: var(--text-muted);">${f.coefficient >= 0 ? '+' : ''}${f.coefficient.toFixed(4)}</td>
        <td style="color: var(--green-safe); font-weight: 700;">${f.contribution >= 0 ? '+' : ''}${f.contribution.toFixed(4)}</td>
      `;
      topHamTbody.appendChild(tr);
    });
  } else {
    topHamTbody.innerHTML = '<tr><td colspan="3" style="color: var(--text-dim);">No negative feature contributions</td></tr>';
  }

  // Headers Tab
  const headersTbody = document.getElementById('headers-tbody');
  headersTbody.innerHTML = '';
  const headerKeys = ['subject', 'from', 'to', 'date', 'message_id', 'content_type', 'is_multipart', 'attachment_count', 'raw_byte_length'];
  headerKeys.forEach(k => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <th style="width: 25%; text-transform: uppercase;">${escapeHtml(k.replace('_', ' '))}</th>
      <td style="color: var(--text-main); word-break: break-all;">${escapeHtml(String(telemetry[k] !== undefined ? telemetry[k] : 'N/A'))}</td>
    `;
    headersTbody.appendChild(tr);
  });
}

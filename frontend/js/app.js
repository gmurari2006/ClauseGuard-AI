/**
 * ClauseGuard AI - Main Frontend Application Orchestrator
 * Coordinates document uploads, session state, sample loading, PDF export,
 * and ML evaluation modals.
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let loadedDocuments = []; // Array of { id, title, filename, size, domain, analysis }
  let activeDocId = null;
  let activeAnalysis = null;

  // Initialize sub-modules
  window.ViewerModule.init();
  window.QAModule.init();

  // Setup DOM Event Listeners
  setupUploadDropzone();
  setupSampleSelector();
  setupPasteModal();
  setupMlEvalModal();
  setupCompareModal();
  setupPdfDownload();
  setupClearAllDocs();
  setupGlobalKeyboardHandlers();

  // Hydrate ML navbar badge honestly from real /api/ml-eval metrics (does not load any document)
  hydrateMlNavbarLabel();

  // Defensively clear any stale browser storage state on fresh launch
  try {
    Object.keys(localStorage).forEach((k) => {
      if (k.toLowerCase().includes("clauseguard") || k.toLowerCase().includes("doc")) {
        localStorage.removeItem(k);
      }
    });
    Object.keys(sessionStorage).forEach((k) => {
      if (k.toLowerCase().includes("clauseguard") || k.toLowerCase().includes("doc")) {
        sessionStorage.removeItem(k);
      }
    });
  } catch (e) {
    // Ignore storage access restrictions
  }

  // Start in clean empty state (no preloaded or default agreement)
  loadedDocuments = [];
  resetAppToEmptyState();

  // Ensure bfcache / soft reloads never restore stale dropdown or panel state
  window.addEventListener("pageshow", (event) => {
    if (event.persisted || !loadedDocuments.length) {
      resetAppToEmptyState();
    }
  });

  // ==========================================
  // DOCUMENT UPLOAD (Drag-and-Drop + File Input)
  // ==========================================
  function setupUploadDropzone() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    if (!dropZone || !fileInput) return;

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFiles(e.target.files);
        fileInput.value = "";
      }
    });
  }

  function handleFiles(files) {
    Array.from(files).forEach((file) => {
      uploadFileToServer(file);
    });
  }

  function uploadFileToServer(file) {
    setProcessingStatus("loading", `Uploading & parsing ${file.name}...`);
    const formData = new FormData();
    formData.append("file", file);

    fetch("/api/upload", {
      method: "POST",
      body: formData
    })
      .then((res) => {
        if (!res.ok) {
          return res.json().then((d) => {
            throw new Error(d.detail || "Upload failed");
          });
        }
        return res.json();
      })
      .then((analysis) => {
        setProcessingStatus("ready", `Analyzed ${file.name}`);
        const docEntry = {
          id: analysis.document_id || `doc_${Date.now()}`,
          title: file.name,
          filename: file.name,
          size: formatFileSize(file.size),
          domain: analysis.domain || "GENERAL",
          analysis: analysis
        };
        addDocumentToState(docEntry);
        setActiveDocument(docEntry.id);
      })
      .catch((err) => {
        setProcessingStatus("ready", "Error processing file");
        alert(`Failed to analyze file: ${err.message}`);
      });
  }

  // ==========================================
  // SAMPLE AGREEMENT LOADING
  // ==========================================
  function setupSampleSelector() {
    const select = document.getElementById("sampleSelect");
    if (!select) return;

    select.addEventListener("change", (e) => {
      const sampleId = e.target.value;
      if (sampleId) {
        loadSampleAgreement(sampleId);
      }
    });
  }

  function loadSampleAgreement(sampleId) {
    // Check if sample already loaded in loadedDocuments
    const existing = loadedDocuments.find((d) => d.sample_id === sampleId || d.id === sampleId);
    if (existing) {
      setActiveDocument(existing.id);
      return;
    }

    setProcessingStatus("loading", `Classifying sample agreement...`);
    fetch(`/api/samples/${sampleId}`)
      .then((res) => res.json())
      .then((analysis) => {
        setProcessingStatus("ready", "Ready for analysis");
        const docEntry = {
          id: analysis.document_id,
          sample_id: sampleId,
          title: analysis.sample_title || analysis.document_info.filename,
          filename: analysis.document_info.filename,
          size: `${analysis.document_info.word_count || 500} words`,
          domain: analysis.domain || "INTERNSHIP",
          analysis: analysis
        };
        addDocumentToState(docEntry);
        setActiveDocument(docEntry.id);
      })
      .catch((err) => {
        setProcessingStatus("ready", "Ready");
        console.error("Failed to load sample agreement", err);
      });
  }

  // ==========================================
  // DOCUMENT STATE MANAGEMENT
  // ==========================================
  function addDocumentToState(docEntry) {
    // Avoid duplicates by filename
    const existingIdx = loadedDocuments.findIndex((d) => d.filename === docEntry.filename);
    if (existingIdx >= 0) {
      loadedDocuments[existingIdx] = docEntry;
    } else {
      loadedDocuments.push(docEntry);
    }
    renderDocumentsList();
  }

  function setActiveDocument(docId) {
    activeDocId = docId;
    const doc = loadedDocuments.find((d) => d.id === docId);
    if (!doc) return;

    activeAnalysis = doc.analysis;

    // Update Header
    const titleEl = document.getElementById("headerDocTitle");
    const domainEl = document.getElementById("headerDocDomain");
    const dlBtn = document.getElementById("btnDownloadReport");
    if (titleEl) titleEl.innerText = doc.filename;
    if (domainEl) domainEl.innerText = (doc.domain || "AGREEMENT").replace(/_/g, " ");
    if (dlBtn) dlBtn.disabled = false;

    // Synchronize Top Sample Dropdown
    const sampleSelect = document.getElementById("sampleSelect");
    if (sampleSelect) {
      const sampleOption = Array.from(sampleSelect.options).find(
        (opt) => opt.value === doc.id || opt.value === doc.sample_id || opt.text.includes(doc.filename)
      );
      if (sampleOption) {
        sampleSelect.value = sampleOption.value;
      } else {
        sampleSelect.value = "";
      }
    }

    // Notify modules
    if (window.ViewerModule) window.ViewerModule.renderDocument(activeAnalysis);
    if (window.InsightsModule) window.InsightsModule.renderInsights(activeAnalysis);
    if (window.QAModule) {
      if (typeof window.QAModule.setActiveDocument === "function") {
        window.QAModule.setActiveDocument(docId, doc.domain);
      } else {
        window.QAModule.setActiveDocumentId(docId);
      }
    }

    // Update list visual state
    renderDocumentsList();
  }

  function renderDocumentsList() {
    const container = document.getElementById("docsListContainer");
    const countBadge = document.getElementById("docCountBadge");
    const compareBtn = document.getElementById("btnCompareDocs");
    if (compareBtn) {
      compareBtn.disabled = loadedDocuments.length < 2;
      compareBtn.title =
        loadedDocuments.length >= 2
          ? `Compare ${loadedDocuments.length} loaded agreements side by side`
          : "Load at least 2 agreements to compare side by side";
    }
    if (!container) return;

    if (countBadge) countBadge.innerText = loadedDocuments.length;

    if (!loadedDocuments.length) {
      container.innerHTML = `
        <div class="empty-docs-placeholder">
          <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
            <polyline points="13 2 13 9 20 9"/>
          </svg>
          <p>No documents uploaded yet</p>
          <span class="helper-text">Select a sample contract above or drop a file to start auditing</span>
        </div>
      `;
      return;
    }

    let html = "";
    loadedDocuments.forEach((doc) => {
      const isActive = doc.id === activeDocId;
      const score = doc.analysis ? doc.analysis.risk_score : 50;
      const riskClass = score >= 70 ? "high" : score >= 40 ? "medium" : "low";

      html += `
        <div class="doc-item-card ${isActive ? 'active' : ''}" data-doc-id="${doc.id}">
          <div class="doc-item-header">
            <div class="doc-title-wrapper" title="${escapeHtml(doc.filename)}">
              <svg class="doc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
              <span class="doc-name">${escapeHtml(doc.filename)}</span>
            </div>
            <div class="doc-actions-inline">
              <button class="doc-action-btn btn-dl" title="Download text" data-doc-id="${doc.id}">
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7 10 12 15 17 10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
              </button>
              <button class="doc-action-btn btn-del" title="Remove" data-doc-id="${doc.id}">
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
          </div>
          <div class="doc-meta-row">
            <span>${doc.size} • ${(doc.domain || "general").replace(/_/g, " ").toLowerCase()}</span>
            <span class="doc-risk-badge ${riskClass}">Risk ${score}/100</span>
          </div>
        </div>
      `;
    });

    container.innerHTML = html;

    // Attach click listeners
    container.querySelectorAll(".doc-item-card").forEach((card) => {
      card.addEventListener("click", (e) => {
        if (e.target.closest(".doc-action-btn")) return;
        const id = card.getAttribute("data-doc-id");
        setActiveDocument(id);
      });
    });

    // Delete handlers
    container.querySelectorAll(".btn-del").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const id = btn.getAttribute("data-doc-id");
        removeDocument(id);
      });
    });

    // Download handlers
    container.querySelectorAll(".btn-dl").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const id = btn.getAttribute("data-doc-id");
        downloadDocRaw(id);
      });
    });
  }

  function removeDocument(id) {
    loadedDocuments = loadedDocuments.filter((d) => d.id !== id);
    if (activeDocId === id) {
      if (loadedDocuments.length > 0) {
        setActiveDocument(loadedDocuments[0].id);
      } else {
        resetAppToEmptyState();
      }
    } else {
      renderDocumentsList();
    }
  }

  function resetAppToEmptyState() {
    activeDocId = null;
    activeAnalysis = null;
    const titleEl = document.getElementById("headerDocTitle");
    const domainEl = document.getElementById("headerDocDomain");
    const dlBtn = document.getElementById("btnDownloadReport");
    const sampleSelect = document.getElementById("sampleSelect");

    if (titleEl) titleEl.innerText = "No agreement loaded";
    if (domainEl) domainEl.innerText = "Standby";
    if (dlBtn) dlBtn.disabled = true;
    if (sampleSelect) {
      sampleSelect.selectedIndex = 0;
      sampleSelect.value = "";
    }

    // Reset active centre tab to Document tab
    document.querySelectorAll(".nav-tab").forEach((t) => {
      t.classList.toggle("active", t.getAttribute("data-tab") === "tab-document");
    });
    document.querySelectorAll(".tab-content").forEach((c) => {
      c.classList.toggle("active", c.id === "tab-document");
    });

    setProcessingStatus("ready", "Ready for analysis");

    if (window.ViewerModule && window.ViewerModule.resetViewer) {
      window.ViewerModule.resetViewer();
    }

    if (window.InsightsModule && window.InsightsModule.resetInsights) {
      window.InsightsModule.resetInsights();
    }

    if (window.QAModule && window.QAModule.resetQA) {
      window.QAModule.resetQA();
    }

    renderDocumentsList();
  }

  function downloadDocRaw(id) {
    const doc = loadedDocuments.find((d) => d.id === id);
    if (!doc || !doc.analysis) return;
    const blob = new Blob([doc.analysis.raw_text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = doc.filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function setupClearAllDocs() {
    const clearBtn = document.getElementById("btnClearAllDocs");
    if (!clearBtn) return;
    clearBtn.addEventListener("click", () => {
      loadedDocuments = [];
      fetch("/api/sessions/clear", { method: "POST" }).catch(() => {});
      resetAppToEmptyState();
    });
  }

  // ==========================================
  // PDF REPORT GENERATION
  // ==========================================
  function setupPdfDownload() {
    const btn = document.getElementById("btnDownloadReport");
    if (!btn) return;

    btn.addEventListener("click", () => {
      if (!activeAnalysis) return;

      const originalText = btn.innerHTML;
      btn.innerHTML = `<span>Generating PDF...</span>`;
      btn.disabled = true;

      fetch("/api/generate-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: activeDocId,
          analysis: activeAnalysis
        })
      })
        .then((res) => {
          if (!res.ok) throw new Error("PDF generation failed");
          return res.blob();
        })
        .then((blob) => {
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          const fname = `ClauseGuard_Report_${(activeAnalysis.document_info.filename || "Agreement").replace(/\.[^/.]+$/, "")}.pdf`;
          a.download = fname;
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
          btn.innerHTML = originalText;
          btn.disabled = false;
        })
        .catch((err) => {
          alert(`Error generating PDF: ${err.message}`);
          btn.innerHTML = originalText;
          btn.disabled = false;
        });
    });
  }

  // ==========================================
  // PASTE CONTRACT MODAL
  // ==========================================
  function setupPasteModal() {
    const modal = document.getElementById("pasteModal");
    const openBtn = document.getElementById("btnOpenPasteModal");
    const closeBtn = document.getElementById("btnClosePasteModal");
    const cancelBtn = document.getElementById("btnCancelPaste");
    const submitBtn = document.getElementById("btnSubmitPaste");
    const titleInput = document.getElementById("pasteDocTitle");
    const textInput = document.getElementById("pasteTextInput");

    if (!modal) return;

    const closeModal = () => modal.classList.remove("active");
    if (openBtn) openBtn.addEventListener("click", () => modal.classList.add("active"));
    if (closeBtn) closeBtn.addEventListener("click", closeModal);
    if (cancelBtn) cancelBtn.addEventListener("click", closeModal);
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });

    if (submitBtn) {
      submitBtn.addEventListener("click", () => {
        const text = textInput.value.trim();
        if (!text) {
          alert("Please enter agreement text to analyze.");
          return;
        }
        const title = titleInput.value.trim() || `Custom_Agreement_${Date.now()}.txt`;
        closeModal();

        setProcessingStatus("loading", "Analyzing pasted contract...");
        fetch("/api/analyze-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: text,
            filename: title
          })
        })
          .then((res) => res.json())
          .then((analysis) => {
            setProcessingStatus("ready", "Ready");
            textInput.value = "";
            titleInput.value = "";
            const docEntry = {
              id: analysis.document_id,
              title: title,
              filename: title,
              size: `${analysis.document_info.word_count || 300} words`,
              domain: analysis.domain || "GENERAL",
              analysis: analysis
            };
            addDocumentToState(docEntry);
            setActiveDocument(docEntry.id);
          })
          .catch((err) => {
            setProcessingStatus("ready", "Ready");
            alert(`Failed to analyze: ${err.message}`);
          });
      });
    }
  }

  // ==========================================
  // ML EVALUATION & DATASET READINESS MODAL
  // ==========================================
  function hydrateMlNavbarLabel() {
    const labelEl = document.getElementById("mlPipelineBtnLabel");
    if (!labelEl) return;

    fetch("/api/ml-eval")
      .then((res) => {
        if (!res.ok) throw new Error("ML eval unavailable");
        return res.json();
      })
      .then((metrics) => {
        if (typeof metrics?.weighted_avg_f1 === "number" && !Number.isNaN(metrics.weighted_avg_f1)) {
          labelEl.innerText = `ML Baseline: ${(metrics.weighted_avg_f1 * 100).toFixed(1)}% F1`;
        } else {
          labelEl.innerText = "ML Evaluation";
        }
      })
      .catch(() => {
        labelEl.innerText = "ML Evaluation";
      });
  }

  function setupMlEvalModal() {
    const modal = document.getElementById("mlEvalModal");
    const openBtn = document.getElementById("btnMlEval");
    const closeBtn = document.getElementById("btnCloseMlModal");
    const closeBottomBtn = document.getElementById("btnCloseMlModalBtn");

    if (!modal) return;

    const closeModal = () => modal.classList.remove("active");
    if (openBtn) {
      openBtn.addEventListener("click", () => {
        fetchMlEvaluationMetrics();
        fetchDatasetReadinessStatus();
        modal.classList.add("active");
      });
    }
    if (closeBtn) closeBtn.addEventListener("click", closeModal);
    if (closeBottomBtn) closeBottomBtn.addEventListener("click", closeModal);
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
  }

  function fetchMlEvaluationMetrics() {
    fetch("/api/ml-eval")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch ML evaluation metrics");
        return res.json();
      })
      .then((metrics) => {
        renderMlModalContent(metrics);
        // Keep navbar label synchronized with real fetched metrics
        const labelEl = document.getElementById("mlPipelineBtnLabel");
        if (labelEl && typeof metrics?.weighted_avg_f1 === "number" && !Number.isNaN(metrics.weighted_avg_f1)) {
          labelEl.innerText = `ML Baseline: ${(metrics.weighted_avg_f1 * 100).toFixed(1)}% F1`;
        }
      })
      .catch((err) => {
        console.error("Failed to load ML metrics", err);
        renderMlModalContent({});
      });
  }

  function fetchDatasetReadinessStatus() {
    const section = document.getElementById("mlDatasetStatusSection");
    if (!section) return;

    fetch("/api/dataset/status")
      .then((res) => {
        if (!res.ok) {
          // Gracefully ignore 404 or pending endpoint
          section.style.display = "none";
          section.innerHTML = "";
          return null;
        }
        return res.json();
      })
      .then((statusData) => {
        if (!statusData || typeof statusData !== "object") return;
        renderDatasetStatusSection(section, statusData);
      })
      .catch(() => {
        section.style.display = "none";
        section.innerHTML = "";
      });
  }

  function renderDatasetStatusSection(section, data) {
    const corpusSize =
      data?.master_corpus_size ??
      data?.total_clauses ??
      data?.prepared_corpus_size ??
      data?.corpus_size ??
      data?.total_rows;

    const classCount =
      data?.class_count ??
      data?.num_classes ??
      (Array.isArray(data?.classes) ? data.classes.length : undefined);

    const splitsObj = data?.splits || data?.split_sizes || {};
    const trainCnt = splitsObj?.train ?? data?.train_size ?? data?.train_count;
    const valCnt = splitsObj?.val ?? splitsObj?.validation ?? data?.val_size ?? data?.validation_size;
    const testCnt = splitsObj?.test ?? data?.test_size ?? data?.test_count;
    const hasSplits =
      typeof trainCnt === "number" || typeof valCnt === "number" || typeof testCnt === "number";
    const splitSummary = hasSplits
      ? `${trainCnt ?? "N/A"} / ${valCnt ?? "N/A"} / ${testCnt ?? "N/A"}`
      : "N/A";

    // Leakage & duplicate check status
    let leakageLabel = "N/A";
    const qc = data?.quality_checks || data?.leakage_check || {};
    if (typeof data?.leakage_status === "string") {
      leakageLabel = data.leakage_status;
    } else if (typeof data?.leakage_check === "string") {
      leakageLabel = data.leakage_check;
    } else if (typeof data?.zero_leakage === "boolean") {
      leakageLabel = data.zero_leakage ? "Passed (Zero Leakage)" : "Attention Needed";
    } else if (typeof qc?.passed === "boolean") {
      leakageLabel = qc.passed ? "Passed (0 Overlaps)" : "Attention Needed";
    } else if (typeof data?.duplicate_count === "number" || typeof data?.leakage_count === "number") {
      const dups = data?.duplicate_count ?? 0;
      const leaks = data?.leakage_count ?? 0;
      leakageLabel = dups === 0 && leaks === 0 ? "Passed (0 Duplicates / 0 Leakage)" : `${dups} dups / ${leaks} leaks`;
    }

    const statusBadgeText =
      data?.retraining_status ||
      data?.pipeline_status_badge ||
      "Model Retraining Pending Authorization";

    const noteText = data?.governance_note || data?.notes || "";

    section.style.display = "block";
    section.innerHTML = `
      <div class="dataset-status-header">
        <h4>Dataset Preparation Pipeline Status</h4>
        <span class="badge badge-gold dataset-pending-badge">${escapeHtml(statusBadgeText)}</span>
      </div>
      <div class="dataset-status-grid">
        <div class="dataset-stat-tile">
          <div class="tile-val">${corpusSize !== undefined && corpusSize !== null ? escapeHtml(String(corpusSize)) : "N/A"}</div>
          <div class="tile-lbl">Prepared Master Corpus</div>
        </div>
        <div class="dataset-stat-tile">
          <div class="tile-val">${classCount !== undefined && classCount !== null ? escapeHtml(String(classCount)) : "N/A"}</div>
          <div class="tile-lbl">Clause Classes</div>
        </div>
        <div class="dataset-stat-tile">
          <div class="tile-val tile-val-sm">${escapeHtml(splitSummary)}</div>
          <div class="tile-lbl">Train / Val / Test Splits</div>
        </div>
        <div class="dataset-stat-tile">
          <div class="tile-val tile-val-sm">${escapeHtml(leakageLabel)}</div>
          <div class="tile-lbl">Leakage &amp; Duplicate Check</div>
        </div>
      </div>
      ${noteText ? `<p class="dataset-status-note">${escapeHtml(noteText)}</p>` : ""}
    `;
  }

  function renderMlModalContent(metrics) {
    const grid = document.getElementById("mlMetricsGrid");
    const tableCont = document.getElementById("mlReportTableContainer");
    const featuresCont = document.getElementById("mlTopFeaturesContainer");

    if (grid) {
      const acc =
        typeof metrics?.test_accuracy === "number" && !Number.isNaN(metrics.test_accuracy)
          ? `${(metrics.test_accuracy * 100).toFixed(1)}%`
          : "N/A";
      const cvAcc =
        typeof metrics?.cv_accuracy_mean === "number" && !Number.isNaN(metrics.cv_accuracy_mean)
          ? `${(metrics.cv_accuracy_mean * 100).toFixed(1)}%`
          : "N/A";
      const f1 =
        typeof metrics?.weighted_avg_f1 === "number" && !Number.isNaN(metrics.weighted_avg_f1)
          ? metrics.weighted_avg_f1.toFixed(3)
          : "N/A";
      const count =
        typeof metrics?.dataset_size === "number" && !Number.isNaN(metrics.dataset_size)
          ? metrics.dataset_size
          : "N/A";

      grid.innerHTML = `
        <div class="ml-metric-tile">
          <div class="tile-val">${acc}</div>
          <div class="tile-lbl">Test Accuracy (Holdout)</div>
        </div>
        <div class="ml-metric-tile">
          <div class="tile-val">${cvAcc}</div>
          <div class="tile-lbl">5-Fold CV Mean</div>
        </div>
        <div class="ml-metric-tile">
          <div class="tile-val">${f1}</div>
          <div class="tile-lbl">Weighted F1-Score</div>
        </div>
        <div class="ml-metric-tile">
          <div class="tile-val">${count}</div>
          <div class="tile-lbl">Annotated Clauses</div>
        </div>
      `;
    }

    if (tableCont) {
      if (metrics?.classification_report) {
        let rows = "";
        for (const [cls, vals] of Object.entries(metrics.classification_report)) {
          if (!vals || typeof vals !== "object" || typeof vals.precision !== "number") continue;
          const f1 = typeof vals["f1-score"] === "number" ? (vals["f1-score"] * 100).toFixed(1) + "%" : "N/A";
          const prec = (vals["precision"] * 100).toFixed(1) + "%";
          const rec = typeof vals["recall"] === "number" ? (vals["recall"] * 100).toFixed(1) + "%" : "N/A";
          const supp = vals["support"] ?? "N/A";

          rows += `
            <tr>
              <td style="font-weight:600; color:var(--color-navy);">${escapeHtml(cls.replace(/_/g, " "))}</td>
              <td>${prec}</td>
              <td>${rec}</td>
              <td style="font-weight:700; color:var(--color-teal);">${f1}</td>
              <td>${supp}</td>
            </tr>
          `;
        }

        tableCont.innerHTML = `
          <table class="ml-eval-table">
            <thead>
              <tr>
                <th>Clause Category</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Test Support</th>
              </tr>
            </thead>
            <tbody>
              ${rows}
            </tbody>
          </table>
        `;
      } else {
        tableCont.innerHTML = `<p class="empty-list-notice" style="padding:10px;">Classification report unavailable.</p>`;
      }
    }

    if (featuresCont) {
      if (metrics?.top_features) {
        let fHtml = "";
        for (const [cls, words] of Object.entries(metrics.top_features)) {
          const wordList = Array.isArray(words) ? words.join(", ") : String(words || "");
          fHtml += `
            <div class="feature-row">
              <span class="feature-cls">${escapeHtml(cls.replace(/_/g, " "))}:</span>
              <span class="feature-words">${escapeHtml(wordList)}</span>
            </div>
          `;
        }
        featuresCont.innerHTML = fHtml;
      } else {
        featuresCont.innerHTML = `<p class="empty-list-notice">Feature importance data unavailable.</p>`;
      }
    }
  }

  // ==========================================
  // MULTI-DOCUMENT COMPARISON MODAL
  // ==========================================
  function setupCompareModal() {
    const modal = document.getElementById("compareModal");
    const openBtn = document.getElementById("btnCompareDocs");
    const closeBtn = document.getElementById("btnCloseCompareModal");
    const closeBottomBtn = document.getElementById("btnCloseCompareModalBtn");

    if (!modal) return;

    const closeModal = () => modal.classList.remove("active");
    if (openBtn) {
      openBtn.addEventListener("click", () => {
        if (loadedDocuments.length < 2) return;
        renderComparisonModal();
        modal.classList.add("active");
      });
    }
    if (closeBtn) closeBtn.addEventListener("click", closeModal);
    if (closeBottomBtn) closeBottomBtn.addEventListener("click", closeModal);
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
  }

  function renderComparisonModal() {
    const body = document.getElementById("compareModalBody");
    const badge = document.getElementById("compareCountBadge");
    if (!body) return;

    if (badge) {
      badge.innerText = `${loadedDocuments.length} Agreements`;
    }

    if (loadedDocuments.length < 2) {
      body.innerHTML = `<p class="empty-list-notice">Load at least 2 agreements to compare them side by side.</p>`;
      return;
    }

    const numericScores = loadedDocuments
      .map((d) => d?.analysis?.risk_score)
      .filter((s) => typeof s === "number" && !Number.isNaN(s));
    const lowestRiskScore = numericScores.length > 0 ? Math.min(...numericScores) : null;

    let rowsHtml = "";
    loadedDocuments.forEach((doc) => {
      const a = doc.analysis || {};
      const isActive = doc.id === activeDocId;
      const score = typeof a.risk_score === "number" ? a.risk_score : null;
      const scoreClass = score === null ? "neutral" : score >= 70 ? "high" : score >= 40 ? "medium" : "low";
      const isLowest = score !== null && lowestRiskScore !== null && score === lowestRiskScore && numericScores.length >= 2;
      const riskCategory = a.risk_category || "Evaluated";
      const domainLabel = (doc.domain || a.domain || "GENERAL").replace(/_/g, " ");

      const fairness = a.fairness || {};
      const sPct = typeof fairness.student_favor_pct === "number" ? `${fairness.student_favor_pct}%` : "N/A";
      const cPct = typeof fairness.counterparty_favor_pct === "number" ? `${fairness.counterparty_favor_pct}%` : "N/A";
      const fairnessStr = sPct !== "N/A" && cPct !== "N/A" ? `${sPct} / ${cPct}` : "N/A";
      const fairnessLabel = fairness.fairness_label || "";

      const dist = a.distribution || {};
      const subTotal = dist.total ?? a.document_info?.clause_count ?? 0;
      const highCnt = dist.high ?? 0;
      const medCnt = dist.medium ?? 0;
      const lowCnt = dist.low ?? 0;

      const redFlagsCnt = Array.isArray(a.top_issues) ? a.top_issues.length : 0;
      const missingCnt = Array.isArray(a.missing_information) ? a.missing_information.length : 0;
      const crossLinksCnt = a.document_info?.cross_clause_links ?? 0;

      rowsHtml += `
        <tr class="${isActive ? "compare-active-row" : ""}">
          <td>
            <div class="compare-filename-cell">
              <strong>${escapeHtml(doc.filename)}</strong>
              <div class="compare-row-tags">
                ${isActive ? `<span class="badge badge-teal">Active</span>` : ""}
                ${isLowest ? `<span class="badge badge-success">Lowest Risk</span>` : ""}
              </div>
            </div>
          </td>
          <td><span class="compare-domain-pill">${escapeHtml(domainLabel)}</span></td>
          <td>
            <span class="doc-risk-badge ${scoreClass}">${score !== null ? `${score}/100` : "N/A"}</span>
            <div class="compare-submeta">${escapeHtml(riskCategory)}</div>
          </td>
          <td>
            <div class="compare-balance-val">${escapeHtml(fairnessStr)}</div>
            ${fairnessLabel ? `<div class="compare-submeta">${escapeHtml(fairnessLabel)}</div>` : ""}
          </td>
          <td>
            <div class="compare-clauses-total">${subTotal} total</div>
            <div class="compare-hml-pills">
              <span class="hml-tag hml-high">${highCnt}H</span>
              <span class="hml-tag hml-med">${medCnt}M</span>
              <span class="hml-tag hml-low">${lowCnt}L</span>
            </div>
          </td>
          <td class="compare-num-cell">
            <span class="badge ${redFlagsCnt > 0 ? "badge-danger" : "badge-success"}">${redFlagsCnt}</span>
          </td>
          <td class="compare-num-cell">
            <span class="badge ${missingCnt > 0 ? "badge-warning" : "badge-success"}">${missingCnt}</span>
          </td>
          <td class="compare-num-cell">
            <span class="badge ${crossLinksCnt > 0 ? "badge-gold" : "badge-structural"}">${crossLinksCnt}</span>
          </td>
          <td>
            <button type="button" class="btn ${isActive ? "btn-ghost" : "btn-secondary"} btn-xs btn-compare-select" data-doc-id="${escapeHtml(doc.id)}" ${isActive ? "disabled" : ""}>
              ${isActive ? "Viewing" : "Select"}
            </button>
          </td>
        </tr>
      `;
    });

    body.innerHTML = `
      <p class="compare-intro-note">Comparing real analysis metrics across all agreements currently loaded in this session.</p>
      <div class="compare-table-wrapper">
        <table class="compare-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Domain</th>
              <th>Risk Score &amp; Category</th>
              <th>Student / Counterparty</th>
              <th>Substantive Clauses</th>
              <th>Red Flags</th>
              <th>Missing Terms</th>
              <th>Cross-Clause Links</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            ${rowsHtml}
          </tbody>
        </table>
      </div>
    `;

    body.querySelectorAll(".btn-compare-select").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-doc-id");
        if (targetId) {
          setActiveDocument(targetId);
          const modal = document.getElementById("compareModal");
          if (modal) modal.classList.remove("active");
        }
      });
    });
  }

  // ==========================================
  // GLOBAL KEYBOARD POLISH (Escape to Close)
  // ==========================================
  function setupGlobalKeyboardHandlers() {
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;

      ["pasteModal", "mlEvalModal", "compareModal"].forEach((modalId) => {
        const modal = document.getElementById(modalId);
        if (modal && modal.classList.contains("active")) {
          modal.classList.remove("active");
        }
      });

      const inspector = document.getElementById("clauseInspector");
      if (inspector && inspector.classList.contains("active")) {
        if (window.ViewerModule && typeof window.ViewerModule.closeInspector === "function") {
          window.ViewerModule.closeInspector();
        } else {
          inspector.classList.remove("active");
        }
      }
    });
  }

  // ==========================================
  // HELPERS
  // ==========================================
  function setProcessingStatus(state, label) {
    const icon = document.getElementById("statusIcon");
    const lbl = document.getElementById("statusLabel");
    if (!icon || !lbl) return;

    icon.className = `status-indicator-icon ${state}-state`;
    lbl.innerText = label;
  }

  function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return "12 KB";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});

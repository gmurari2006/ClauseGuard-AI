/**
 * ClauseGuard AI - Document Viewer & Tab Workspace Module
 * Renders highlighted clauses, handles tab switching, clause inspector drawer,
 * summary view, and the contract review audit table.
 */

window.ViewerModule = (function () {
  let currentClauses = [];
  let currentRawText = "";
  let activeFilter = "all";
  let activeClauseIndex = null;

  function init() {
    setupTabNavigation();
    setupFilterChips();
    setupSearchInput();
    setupInspectorDrawer();
  }

  function setupTabNavigation() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

        tab.classList.add("active");
        const tabId = tab.getAttribute("data-tab");
        const target = document.getElementById(tabId);
        if (target) {
          target.classList.add("active");
        }
      });
    });
  }

  function setupFilterChips() {
    const chips = document.querySelectorAll(".filter-chip");
    chips.forEach((chip) => {
      chip.addEventListener("click", () => {
        chips.forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        activeFilter = chip.getAttribute("data-filter");
        applyClauseFilter();
      });
    });
  }

  function setupSearchInput() {
    const searchInput = document.getElementById("viewerSearchInput");
    if (!searchInput) return;
    searchInput.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      const blocks = document.querySelectorAll(".clause-block");
      blocks.forEach((b) => {
        if (!q) {
          b.style.display = "";
          return;
        }
        const text = b.textContent.toLowerCase();
        if (text.includes(q)) {
          b.style.display = "";
        } else {
          b.style.display = "none";
        }
      });
    });
  }

  function setupInspectorDrawer() {
    const closeBtn = document.getElementById("btnCloseInspector");
    const drawer = document.getElementById("clauseInspector");
    const copyBtn = document.getElementById("btnCopyCounterProposal");

    if (closeBtn && drawer) {
      closeBtn.addEventListener("click", () => {
        closeInspector();
      });
    }

    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        const text = document.getElementById("inspActionText").innerText;
        copyTextToClipboard(text).then(() => {
          const originalHtml = copyBtn.innerHTML;
          copyBtn.innerHTML = `<span>Copied!</span>`;
          setTimeout(() => {
            copyBtn.innerHTML = originalHtml;
          }, 1500);
        });
      });
    }
  }

  function closeInspector() {
    const drawer = document.getElementById("clauseInspector");
    if (drawer) drawer.classList.remove("active");
    clearSelectedClauseHighlight();
  }

  function copyTextToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).catch(() => fallbackCopyText(text));
    }
    return fallbackCopyText(text);
  }

  function fallbackCopyText(text) {
    return new Promise((resolve) => {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
      } catch (e) {
        // Ignore copy failure
      }
      ta.remove();
      resolve();
    });
  }

  function clearSelectedClauseHighlight() {
    document.querySelectorAll(".clause-block").forEach((b) => b.classList.remove("selected-clause"));
    activeClauseIndex = null;
  }

  function renderDocument(analysis) {
    if (!analysis) return;
    currentClauses = analysis.clauses || [];
    currentRawText = analysis.raw_text || "";

    const sheet = document.getElementById("documentSheet");
    if (!sheet) return;

    if (!currentClauses.length) {
      sheet.innerHTML = `
        <div class="empty-sheet-state">
          <p>No clauses identified in this document.</p>
        </div>
      `;
      return;
    }

    // Render clauses with highlights and clear spacing
    let html = "";
    currentClauses.forEach((clause) => {
      const isHeading = clause.is_heading;
      const isStructural = Boolean(clause.is_structural);
      const risk = isStructural && !isHeading ? "informational" : (clause.final_risk_level || "low").toLowerCase();
      const cat = clause.category || "STANDARD_FAIR";
      const catFormatted = cat === "STRUCTURAL_INFORMATIONAL" ? "STRUCTURAL / INFORMATIONAL" : cat.replace(/_/g, " ");

      const riskPillText =
        risk === "high"
          ? "HIGH RISK"
          : risk === "medium"
          ? "MEDIUM RISK"
          : risk === "informational"
          ? "INFO"
          : "LOW RISK";
      const conf = typeof clause.ml_confidence === "number" ? Math.round(clause.ml_confidence * 100) : null;
      const sourceLabel = clause.source === "STATUTORY_RULE"
        ? "Statutory Rule"
        : clause.source === "COMBINED_EVALUATION"
        ? "Statutory + ML"
        : clause.source === "CROSS_CLAUSE_ANALYSIS"
        ? "Cross-Clause Analysis"
        : clause.source === "DOCUMENT_STRUCTURE"
        ? "Structure"
        : conf !== null
        ? `ML Model (${conf}%)`
        : "ML Model";

      const clauseRefLabel = clause.clause_number ? `Clause ${clause.clause_number}` : `#${clause.index}`;
      const relatedClauses = clause.related_clauses || [];
      let crossRefHtml = "";
      if (relatedClauses.length > 0) {
        const chips = relatedClauses.map((rc) => `
          <button type="button" class="crossref-chip" data-target-index="${rc.target_index}" title="Jump to ${escapeHtml(rc.target_clause_ref)}">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.2">
              <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
              <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
            </svg>
            <span>${escapeHtml(rc.display_text)}</span>
          </button>
        `).join("");
        crossRefHtml = `<div class="clause-crossref-bar">${chips}</div>`;
      }

      if (isHeading) {
        html += `
          <div class="clause-block heading-block" data-index="${clause.index}">
            <div class="heading-title-text">${escapeHtml(clause.original_text)}</div>
          </div>
        `;
      } else {
        html += `
          <div class="clause-block risk-${risk}" data-index="${clause.index}" data-risk="${risk}" title="Click to view legal analysis and counter-proposal">
            <div class="clause-header-row">
              <div class="clause-header-left">
                <span class="clause-index-badge">${escapeHtml(clauseRefLabel)}</span>
                <span class="clause-category-badge cat-${risk}">${escapeHtml(catFormatted)}</span>
                <span class="clause-risk-pill pill-${risk}">${riskPillText}</span>
              </div>
              <div class="clause-header-right">
                <span class="clause-source-badge">${escapeHtml(sourceLabel)}</span>
              </div>
            </div>
            <div class="clause-body-text">${formatClauseBody(clause.original_text)}</div>
            ${crossRefHtml}
            ${clause.rule_violations && clause.rule_violations.length > 0 ? `
              <div class="clause-statutory-notice">
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <span><strong>Statutory Flag:</strong> ${escapeHtml(clause.rule_violations[0].rule_name)} (${escapeHtml(clause.rule_violations[0].statute)})</span>
              </div>
            ` : ''}
          </div>
        `;
      }
    });

    sheet.innerHTML = html;

    // Attach cross-reference jump listeners
    sheet.querySelectorAll(".crossref-chip").forEach((chip) => {
      chip.addEventListener("click", (e) => {
        e.stopPropagation();
        const targetIdx = parseInt(chip.getAttribute("data-target-index"), 10);
        if (!isNaN(targetIdx)) {
          scrollToClause(targetIdx);
        }
      });
    });

    // Attach click listeners to highlighted clauses
    sheet.querySelectorAll(".clause-block").forEach((block) => {
      if (block.classList.contains("heading-block")) return;
      block.addEventListener("click", () => {
        const idx = parseInt(block.getAttribute("data-index"), 10);
        const clause = currentClauses.find((c) => c.index === idx);
        if (clause) {
          selectClause(clause, block);
        }
      });
    });

    // Update filter counts
    updateFilterCounts();
    applyClauseFilter();

    // Render other tabs
    renderSummaryTab(analysis);
    renderReviewTable(analysis);
  }

  function selectClause(clause, element) {
    clearSelectedClauseHighlight();
    activeClauseIndex = clause.index;
    if (element) {
      element.classList.add("selected-clause");
    }

    const drawer = document.getElementById("clauseInspector");
    if (!drawer) return;

    // Populate drawer
    const inspCat = clause.category === "STRUCTURAL_INFORMATIONAL"
      ? "STRUCTURAL / INFORMATIONAL"
      : (clause.category || "STANDARD").replace(/_/g, " ");
    document.getElementById("inspCategoryBadge").innerText = inspCat;
    
    const riskBadge = document.getElementById("inspRiskBadge");
    const risk = clause.is_structural ? "INFORMATIONAL" : (clause.final_risk_level || "low").toUpperCase();
    riskBadge.innerText = risk === "INFORMATIONAL" ? "INFORMATIONAL" : `${risk} RISK`;
    riskBadge.style.background =
      risk === "HIGH"
        ? "var(--color-high-risk)"
        : risk === "MEDIUM"
        ? "var(--color-med-risk)"
        : risk === "INFORMATIONAL"
        ? "var(--color-text-muted)"
        : "var(--color-low-risk)";

    // Source & Confidence
    const conf = typeof clause.ml_confidence === "number" ? Math.round(clause.ml_confidence * 100) : null;
    const sourceLabel = clause.source === "STATUTORY_RULE"
      ? "Statutory Rule"
      : clause.source === "COMBINED_EVALUATION"
      ? "Statutory + ML"
      : clause.source === "CROSS_CLAUSE_ANALYSIS"
      ? "Cross-Clause Analysis"
      : clause.source === "DOCUMENT_STRUCTURE"
      ? "Structure"
      : conf !== null
      ? `ML Model (${conf}%)`
      : "ML Model";
    document.getElementById("inspSourceBadge").innerText = sourceLabel;

    document.getElementById("inspOriginalText").innerText = `"${clause.original_text}"`;
    document.getElementById("inspExplanation").innerText = clause.explanation || "No specific warning detected.";
    document.getElementById("inspActionText").innerText = clause.recommended_action || "Standard clause. Keep as is.";

    // Populate linked cross-references in inspector
    const crossRefSec = document.getElementById("inspCrossRefsSection");
    const crossRefList = document.getElementById("inspCrossRefsList");
    const related = clause.related_clauses || [];
    if (crossRefSec && crossRefList) {
      if (related.length > 0) {
        crossRefSec.style.display = "block";
        crossRefList.innerHTML = related.map((rc) => `
          <button type="button" class="crossref-chip" data-target-index="${rc.target_index}">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.2">
              <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
              <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
            </svg>
            <span>${escapeHtml(rc.display_text)}</span>
          </button>
        `).join("");
        crossRefList.querySelectorAll(".crossref-chip").forEach((chip) => {
          chip.addEventListener("click", () => {
            const idx = parseInt(chip.getAttribute("data-target-index"), 10);
            if (!isNaN(idx)) scrollToClause(idx);
          });
        });
      } else {
        crossRefSec.style.display = "none";
        crossRefList.innerHTML = "";
      }
    }

    drawer.classList.add("active");
  }

  function scrollToClause(index) {
    // Switch to Document tab
    const docTab = document.querySelector('[data-tab="tab-document"]');
    if (docTab) docTab.click();

    setTimeout(() => {
      const block = document.querySelector(`.clause-block[data-index="${index}"]`);
      if (block) {
        block.scrollIntoView({ behavior: "smooth", block: "center" });
        const clause = currentClauses.find((c) => c.index === index);
        if (clause) {
          selectClause(clause, block);
        }
      }
    }, 150);
  }

  function updateFilterCounts() {
    const substantive = currentClauses.filter((c) => !c.is_heading && !c.is_structural);
    const high = substantive.filter((c) => c.final_risk_level === "high").length;
    const med = substantive.filter((c) => c.final_risk_level === "medium").length;
    const low = substantive.filter((c) => c.final_risk_level === "low").length;

    const elHigh = document.getElementById("countFilterHigh");
    const elMed = document.getElementById("countFilterMed");
    const elLow = document.getElementById("countFilterLow");
    const tabCount = document.getElementById("tabReviewCount");

    if (elHigh) elHigh.innerText = high;
    if (elMed) elMed.innerText = med;
    if (elLow) elLow.innerText = low;
    if (tabCount) tabCount.innerText = substantive.length;
  }

  function applyClauseFilter() {
    const blocks = document.querySelectorAll(".clause-block");
    blocks.forEach((b) => {
      if (b.classList.contains("heading-block")) {
        b.style.display = "";
        return;
      }
      const risk = b.getAttribute("data-risk");
      if (activeFilter === "all" || risk === activeFilter) {
        b.style.display = "";
      } else {
        b.style.display = "none";
      }
    });
  }

  function formatClauseBody(text) {
    if (!text) return "";
    const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    if (lines.length <= 1) {
      return `<p class="clause-para">${escapeHtml(text)}</p>`;
    }
    return lines.map((line) => `<p class="clause-para">${escapeHtml(line)}</p>`).join("");
  }

  function formatSignedPoints(val) {
    if (typeof val !== "number" || Number.isNaN(val)) return "N/A";
    if (val > 0) return `+${val}`;
    return `${val}`;
  }

  function renderRiskBreakdownCard(breakdown) {
    if (!breakdown || typeof breakdown !== "object") return "";

    const baseline = typeof breakdown.baseline === "number" ? breakdown.baseline : null;
    const highCount = breakdown.high_count ?? 0;
    const highWeight = breakdown.high_weight ?? 18;
    const highPoints = typeof breakdown.high_points === "number" ? breakdown.high_points : null;

    const ruleCount = breakdown.rule_flag_count ?? 0;
    const ruleWeight = breakdown.rule_flag_weight ?? 8;
    const rulePoints = typeof breakdown.rule_flag_points === "number" ? breakdown.rule_flag_points : null;

    const medCount = breakdown.medium_count ?? 0;
    const medWeight = breakdown.medium_weight ?? 6;
    const medPoints = typeof breakdown.medium_points === "number" ? breakdown.medium_points : null;

    const missCount = breakdown.missing_count ?? 0;
    const missWeight = breakdown.missing_weight ?? 4;
    const missPoints = typeof breakdown.missing_points === "number" ? breakdown.missing_points : null;

    const lowCount = breakdown.low_count ?? 0;
    const lowWeight = breakdown.low_weight ?? -2;
    const lowOffset = typeof breakdown.low_offset === "number" ? breakdown.low_offset : null;

    const rawScore = typeof breakdown.raw_score === "number" ? breakdown.raw_score : null;
    const finalScore = typeof breakdown.final_score === "number" ? breakdown.final_score : null;
    const bounds = Array.isArray(breakdown.bounds) && breakdown.bounds.length === 2 ? breakdown.bounds : [10, 98];
    const cappedNote = breakdown.was_capped
      ? `Bounded to [${bounds[0]}–${bounds[1]}] scale`
      : `Within [${bounds[0]}–${bounds[1]}] scale`;

    return `
      <div class="summary-transparency-card">
        <div class="transparency-card-header">
          <div>
            <h3>Transparent Risk Score Calculation</h3>
            <p class="transparency-subtitle">Deterministic weighted audit formula derived strictly from classified substantive clauses and statutory rules.</p>
          </div>
          <span class="transparency-score-badge">${finalScore !== null ? `${finalScore}/100` : "N/A"}</span>
        </div>

        <div class="breakdown-ledger">
          <div class="breakdown-row">
            <span class="breakdown-label">Baseline Risk Floor</span>
            <span class="breakdown-formula">Starting baseline</span>
            <span class="breakdown-points pts-neutral">${formatSignedPoints(baseline)}</span>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-label">High-Risk Clauses</span>
            <span class="breakdown-formula">${highCount} &times; +${highWeight}</span>
            <span class="breakdown-points ${highPoints > 0 ? "pts-high" : "pts-neutral"}">${formatSignedPoints(highPoints)}</span>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-label">Statutory Rule Flags</span>
            <span class="breakdown-formula">${ruleCount} &times; +${ruleWeight}</span>
            <span class="breakdown-points ${rulePoints > 0 ? "pts-high" : "pts-neutral"}">${formatSignedPoints(rulePoints)}</span>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-label">Medium-Risk Clauses</span>
            <span class="breakdown-formula">${medCount} &times; +${medWeight}</span>
            <span class="breakdown-points ${medPoints > 0 ? "pts-med" : "pts-neutral"}">${formatSignedPoints(medPoints)}</span>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-label">Missing Protective Terms</span>
            <span class="breakdown-formula">${missCount} &times; +${missWeight}</span>
            <span class="breakdown-points ${missPoints > 0 ? "pts-med" : "pts-neutral"}">${formatSignedPoints(missPoints)}</span>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-label">Low-Risk / Standard Offset</span>
            <span class="breakdown-formula">${lowCount} &times; ${lowWeight}</span>
            <span class="breakdown-points ${lowOffset < 0 ? "pts-low" : "pts-neutral"}">${formatSignedPoints(lowOffset)}</span>
          </div>
          <div class="breakdown-row breakdown-subtotal">
            <span class="breakdown-label">Raw Unbounded Total</span>
            <span class="breakdown-formula">Sum of weighted factors</span>
            <span class="breakdown-points pts-neutral">${rawScore !== null ? rawScore : "N/A"}</span>
          </div>
          <div class="breakdown-row breakdown-final">
            <span class="breakdown-label">Final Bounded Score</span>
            <span class="breakdown-formula">${escapeHtml(cappedNote)}</span>
            <span class="breakdown-points pts-final">${finalScore !== null ? `${finalScore}/100` : "N/A"}</span>
          </div>
        </div>
      </div>
    `;
  }

  function extractClusterClauseItems(cluster) {
    const items = [];
    if (Array.isArray(cluster?.clauses)) {
      cluster.clauses.forEach((c) => {
        if (typeof c === "number") {
          const found = currentClauses.find((x) => x.index === c);
          const label = found?.clause_number ? `Clause ${found.clause_number}` : `Clause #${c}`;
          items.push({ index: c, label });
        } else if (c && typeof c === "object") {
          const idx = typeof c.index === "number" ? c.index : typeof c.clause_index === "number" ? c.clause_index : null;
          const label = c.clause_ref || (c.clause_number ? `Clause ${c.clause_number}` : c.label || (idx !== null ? `Clause #${idx}` : "Clause"));
          items.push({ index: idx, label });
        }
      });
    } else if (Array.isArray(cluster?.clause_indices)) {
      const refs = Array.isArray(cluster?.clause_refs) ? cluster.clause_refs : [];
      cluster.clause_indices.forEach((idx, i) => {
        if (typeof idx === "number") {
          const found = currentClauses.find((x) => x.index === idx);
          const label = refs[i] || (found?.clause_number ? `Clause ${found.clause_number}` : `Clause #${idx}`);
          items.push({ index: idx, label });
        }
      });
    } else if (Array.isArray(cluster?.members)) {
      cluster.members.forEach((m) => {
        if (m && typeof m === "object") {
          const idx = typeof m.index === "number" ? m.index : typeof m.clause_index === "number" ? m.clause_index : null;
          const label = m.clause_ref || (m.clause_number ? `Clause ${m.clause_number}` : (idx !== null ? `Clause #${idx}` : "Clause"));
          items.push({ index: idx, label });
        }
      });
    }
    return items;
  }

  function renderCrossClauseClustersCard(crossSummary) {
    const clusters = crossSummary?.clusters;
    if (!Array.isArray(clusters) || clusters.length === 0) return "";

    const introNote = crossSummary?.summary || crossSummary?.overview || "";
    const clusterCardsHtml = clusters.map((cl) => {
      const title = cl?.title || cl?.name || cl?.cluster_name || cl?.pattern || "Compound Risk Cluster";
      const sevRaw = (cl?.severity || cl?.risk_level || "high").toLowerCase();
      const sevBadgeClass =
        sevRaw === "high" || sevRaw === "critical"
          ? "badge-danger"
          : sevRaw === "medium" || sevRaw === "elevated"
          ? "badge-warning"
          : "badge-success";
      const desc = cl?.description || cl?.summary || cl?.explanation || "";
      const clauseItems = extractClusterClauseItems(cl);
      const chipsHtml = clauseItems
        .map((item) => {
          if (typeof item.index === "number") {
            return `
              <button type="button" class="crossref-chip cluster-clause-chip" data-target-index="${item.index}" title="Jump to ${escapeHtml(item.label)}">
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.2">
                  <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
                  <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
                </svg>
                <span>${escapeHtml(item.label)}</span>
              </button>
            `;
          }
          return `<span class="review-crossref-pill">${escapeHtml(item.label)}</span>`;
        })
        .join("");

      return `
        <div class="cluster-item-card">
          <div class="cluster-item-header">
            <span class="cluster-item-title">${escapeHtml(title)}</span>
            <span class="badge ${sevBadgeClass}">${escapeHtml(sevRaw.toUpperCase())}</span>
          </div>
          ${desc ? `<p class="cluster-item-desc">${escapeHtml(desc)}</p>` : ""}
          ${chipsHtml ? `<div class="cluster-chips-row">${chipsHtml}</div>` : ""}
        </div>
      `;
    }).join("");

    return `
      <div class="summary-clusters-card">
        <div class="transparency-card-header">
          <div>
            <h3>Cross-Clause Risk Clusters</h3>
            <p class="transparency-subtitle">${escapeHtml(introNote || "Interacting provisions that amplify student risk when read together across sections.")}</p>
          </div>
          <span class="badge badge-gold">${clusters.length} ${clusters.length === 1 ? "Cluster" : "Clusters"}</span>
        </div>
        <div class="clusters-stack">
          ${clusterCardsHtml}
        </div>
      </div>
    `;
  }

  function renderSummaryTab(analysis) {
    const container = document.getElementById("summaryContainer");
    if (!container) return;

    const meta = analysis.document_info || {};
    const domainRaw = (analysis.domain || "AGREEMENT").replace(/_/g, " ");
    const domain = domainRaw.charAt(0).toUpperCase() + domainRaw.slice(1).toLowerCase();
    const fairness = analysis.fairness || {};
    const dist = analysis.distribution || {};
    const highCount = dist.high || 0;
    const medCount = dist.medium || 0;
    const lowCount = dist.low || 0;
    const totalSub = dist.total || meta.clause_count || 0;
    const structuralCnt = meta.structural_count || 0;
    const structNote = structuralCnt > 0 ? ` • ${structuralCnt} structural/informational excluded` : "";
    const riskColor =
      analysis.risk_score >= 70
        ? "var(--color-high-risk)"
        : analysis.risk_score >= 40
        ? "var(--color-med-risk)"
        : "var(--color-low-risk)";

    const breakdownHtml = renderRiskBreakdownCard(analysis?.risk_score_breakdown);
    const clustersHtml = renderCrossClauseClustersCard(analysis?.cross_clause_summary);

    container.innerHTML = `
      <div class="summary-card-deck">
        <div class="summary-metric-card">
          <div class="metric-header">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            </svg>
            <span class="metric-title">Agreement Domain</span>
          </div>
          <div class="metric-value">${escapeHtml(domain)}</div>
          <div class="metric-desc">${meta.word_count || 0} words • ${totalSub} substantive clauses (${highCount}H / ${medCount}M / ${lowCount}L)${escapeHtml(structNote)}</div>
        </div>

        <div class="summary-metric-card">
          <div class="metric-header">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 14 14"/>
            </svg>
            <span class="metric-title">Risk Posture</span>
          </div>
          <div class="metric-value" style="color: ${riskColor}">
            ${analysis.risk_score}/100
          </div>
          <div class="metric-desc">${escapeHtml(analysis.risk_category || "Evaluated")}</div>
        </div>

        <div class="summary-metric-card">
          <div class="metric-header">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
            </svg>
            <span class="metric-title">Contract Balance</span>
          </div>
          <div class="metric-value">${fairness.student_favor_pct ?? "N/A"}% Student</div>
          <div class="metric-desc">${escapeHtml(fairness.fairness_label || "Balanced")}</div>
        </div>
      </div>

      <div class="executive-brief-box">
        <h3>Executive Student Brief</h3>
        <p>This document has been parsed and classified by ClauseGuard AI's Machine Learning, Cross-Clause, and Student Protection Rules engine. Across <strong>${totalSub} substantive contractual clauses</strong>, <strong>${highCount} high-risk</strong>, <strong>${medCount} medium-risk</strong>, and <strong>${lowCount} low-risk</strong> clauses were identified.</p>
        <p><strong>Primary Finding:</strong> ${escapeHtml(fairness.summary_verdict || "Standard contract terms apply.")}</p>
        <p><strong>Next Steps:</strong> Review the highlighted clauses in the <em>Document</em> and <em>Contract Review</em> tabs, inspect the proposed counter-clauses, and download the official PDF report to share with your university legal aid or placement cell.</p>
      </div>

      ${breakdownHtml}
      ${clustersHtml}

      <div class="summary-disclaimer-note">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="16" x2="12" y2="12"/>
          <line x1="12" y1="8" x2="12.01" y2="8"/>
        </svg>
        <span>ClauseGuard AI provides automated educational risk analysis and does not constitute formal legal advice.</span>
      </div>
    `;

    // Wire clickable cluster clause jump chips
    container.querySelectorAll(".cluster-clause-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const idx = parseInt(chip.getAttribute("data-target-index"), 10);
        if (!Number.isNaN(idx)) {
          scrollToClause(idx);
        }
      });
    });
  }

  function renderReviewTable(analysis) {
    const container = document.getElementById("reviewTableContainer");
    if (!container) return;

    const allRows = (analysis.clauses || []).filter((c) => !c.is_heading);
    if (!allRows.length) {
      container.innerHTML = `<p class="empty-list-notice">No clauses available for review.</p>`;
      return;
    }

    const substantive = allRows.filter((c) => !c.is_structural);
    const structural = allRows.filter((c) => c.is_structural);
    const highCnt = substantive.filter((c) => c.final_risk_level === "high").length;
    const medCnt = substantive.filter((c) => c.final_risk_level === "medium").length;
    const lowCnt = substantive.filter((c) => c.final_risk_level === "low").length;

    let rows = "";
    allRows.forEach((c, idx) => {
      const isStruct = Boolean(c.is_structural);
      const rowRiskKey = isStruct ? "structural" : (c.final_risk_level || "low").toLowerCase();
      const risk = isStruct ? "INFO" : (c.final_risk_level || "low").toUpperCase();
      const badgeClass =
        risk === "HIGH"
          ? "badge-danger"
          : risk === "MEDIUM"
          ? "badge-warning"
          : risk === "INFO"
          ? "badge-structural"
          : "badge-success";
      const cat = c.category === "STRUCTURAL_INFORMATIONAL"
        ? "STRUCTURAL / INFORMATIONAL"
        : (c.category || "GENERAL").replace(/_/g, " ");
      const refNum = c.clause_number ? `Clause ${c.clause_number}` : (isStruct ? "Info" : `#${idx + 1}`);
      const related = c.related_clauses || [];
      const crossRefHtml = related.length
        ? `<div class="review-crossref-list">${related.map((rc) => `<span class="review-crossref-pill">${escapeHtml(rc.display_text)}</span>`).join("")}</div>`
        : "";
      const rowClass = isStruct ? "review-structural-row" : "";
      const actionLabel = isStruct ? "Note" : "Action";

      rows += `
        <tr class="${rowClass}" data-review-risk="${rowRiskKey}">
          <td style="width: 74px; font-weight: 700; color: var(--color-text-light);">${escapeHtml(refNum)}</td>
          <td style="width: 165px;">
            <span class="badge ${badgeClass}">${risk}</span>
            <div style="font-size: 0.71rem; font-weight: 700; color: ${isStruct ? 'var(--color-text-muted)' : 'var(--color-navy)'}; margin-top: 4px;">${escapeHtml(cat)}</div>
          </td>
          <td class="clause-text-cell" style="width: 43%;">
            ${escapeHtml(c.original_text)}
          </td>
          <td class="advice-cell">
            ${crossRefHtml}
            <div style="color: var(--color-text-main);">${escapeHtml(c.explanation || "")}</div>
            <span class="action-highlight">${actionLabel}: ${escapeHtml(c.recommended_action || "")}</span>
          </td>
        </tr>
      `;
    });

    const checklistItems = substantive.filter(
      (c) => (c.final_risk_level === "high" || c.final_risk_level === "medium") && c.recommended_action
    );

    container.innerHTML = `
      <div class="review-summary-strip">
        <span><strong>Substantive Classified Clauses:</strong> ${substantive.length} (${highCnt} High • ${medCnt} Medium • ${lowCnt} Low)</span>
        <span class="review-summary-meta"><strong>Structural / Informational:</strong> ${structural.length} (Excluded from risk scoring)</span>
      </div>
      <div class="review-controls-bar">
        <div class="review-filter-group" role="group" aria-label="Filter contract review rows">
          <button type="button" class="review-filter-chip active" data-review-filter="all">All (${allRows.length})</button>
          <button type="button" class="review-filter-chip review-filter-high" data-review-filter="high">High (${highCnt})</button>
          <button type="button" class="review-filter-chip review-filter-med" data-review-filter="medium">Medium (${medCnt})</button>
          <button type="button" class="review-filter-chip review-filter-low" data-review-filter="low">Low (${lowCnt})</button>
          <button type="button" class="review-filter-chip review-filter-struct" data-review-filter="structural">Structural (${structural.length})</button>
        </div>
        <button type="button" class="btn btn-ghost btn-xs review-copy-btn" id="btnCopyReviewChecklist" title="Copy High & Medium negotiation recommendations">
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
          </svg>
          <span>Copy Negotiation Checklist</span>
        </button>
      </div>
      <table class="review-table">
        <thead>
          <tr>
            <th>Ref</th>
            <th>Severity & Category</th>
            <th>Original Contract Text</th>
            <th>Student Impact & Specific Recommendation</th>
          </tr>
        </thead>
        <tbody id="reviewTableBody">
          ${rows}
          <tr class="review-empty-filter-row" id="reviewEmptyFilterRow" style="display: none;">
            <td colspan="4" class="review-empty-filter-cell">No clauses match the selected filter.</td>
          </tr>
        </tbody>
      </table>
    `;

    // Wire independent inline review filter bar (does NOT touch Document tab's activeFilter)
    const reviewChips = container.querySelectorAll(".review-filter-chip");
    const tableBody = container.querySelector("#reviewTableBody");
    const emptyFilterRow = container.querySelector("#reviewEmptyFilterRow");

    reviewChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        reviewChips.forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        const selected = chip.getAttribute("data-review-filter");
        let visibleCount = 0;
        if (tableBody) {
          tableBody.querySelectorAll("tr[data-review-risk]").forEach((tr) => {
            const rowRisk = tr.getAttribute("data-review-risk");
            const show = selected === "all" || rowRisk === selected;
            tr.style.display = show ? "" : "none";
            if (show) visibleCount += 1;
          });
        }
        if (emptyFilterRow) {
          emptyFilterRow.style.display = visibleCount === 0 ? "" : "none";
        }
      });
    });

    // Wire Copy Negotiation Checklist button
    const copyChecklistBtn = container.querySelector("#btnCopyReviewChecklist");
    if (copyChecklistBtn) {
      copyChecklistBtn.addEventListener("click", () => {
        const docName = analysis?.document_info?.filename || "Agreement";
        let checklistText = `ClauseGuard AI – Student Negotiation Checklist (${docName})\n`;
        checklistText += "------------------------------------------------------------\n";
        if (checklistItems.length === 0) {
          checklistText += "• No High or Medium risk negotiation items were flagged in this agreement.\n";
        } else {
          checklistItems.forEach((c, i) => {
            const ref = c.clause_number ? `Clause ${c.clause_number}` : `Clause #${c.index ?? i + 1}`;
            const sev = (c.final_risk_level || "medium").toUpperCase();
            const cat = (c.category || "GENERAL").replace(/_/g, " ");
            checklistText += `• [${sev}] ${ref} (${cat}): ${c.recommended_action}\n`;
          });
        }

        copyTextToClipboard(checklistText).then(() => {
          const originalHtml = copyChecklistBtn.innerHTML;
          copyChecklistBtn.classList.add("copied-state");
          copyChecklistBtn.innerHTML = `<span>Copied!</span>`;
          setTimeout(() => {
            copyChecklistBtn.classList.remove("copied-state");
            copyChecklistBtn.innerHTML = originalHtml;
          }, 1800);
        });
      });
    }
  }

  function resetViewer() {
    currentClauses = [];
    currentRawText = "";
    activeFilter = "all";
    activeClauseIndex = null;

    // Reset filter chips active state
    document.querySelectorAll(".filter-chip").forEach((c) => {
      c.classList.toggle("active", c.getAttribute("data-filter") === "all");
    });

    // Clear search input
    const searchInput = document.getElementById("viewerSearchInput");
    if (searchInput) searchInput.value = "";

    // Close inspector drawer
    const drawer = document.getElementById("clauseInspector");
    if (drawer) drawer.classList.remove("active");

    // Reset filter counts and review badge
    const elHigh = document.getElementById("countFilterHigh");
    const elMed = document.getElementById("countFilterMed");
    const elLow = document.getElementById("countFilterLow");
    const tabCount = document.getElementById("tabReviewCount");
    if (elHigh) elHigh.innerText = "0";
    if (elMed) elMed.innerText = "0";
    if (elLow) elLow.innerText = "0";
    if (tabCount) tabCount.innerText = "0";

    // Neutral empty state in Document tab
    const sheet = document.getElementById("documentSheet");
    if (sheet) {
      sheet.innerHTML = `
        <div class="empty-sheet-state">
          <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" stroke-width="1">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
          </svg>
          <h3>Upload an agreement to begin analysis.</h3>
          <p>Drag and drop a PDF, DOCX, or TXT file in the left panel, paste text, or choose one from Sample Contracts.</p>
        </div>
      `;
    }

    // Neutral empty state in Summary tab
    const summaryContainer = document.getElementById("summaryContainer");
    if (summaryContainer) {
      summaryContainer.innerHTML = `
        <div class="tab-empty-state">
          <h3>Upload an agreement to begin analysis.</h3>
          <p>Executive summary metrics and student protection findings will appear here after analysis.</p>
        </div>
      `;
    }

    // Neutral empty state in Contract Review tab
    const reviewContainer = document.getElementById("reviewTableContainer");
    if (reviewContainer) {
      reviewContainer.innerHTML = `
        <div class="tab-empty-state">
          <h3>Upload an agreement to begin analysis.</h3>
          <p>Clause-by-clause audit results and recommended negotiation actions will appear here after analysis.</p>
        </div>
      `;
    }
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

  // Capitalize title helper
  String.prototype.title = function () {
    return this.split(" ")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
      .join(" ");
  };

  return {
    init: init,
    renderDocument: renderDocument,
    resetViewer: resetViewer,
    scrollToClause: scrollToClause,
    selectClause: selectClause,
    closeInspector: closeInspector
  };
})();

/**
 * ClauseGuard AI - Live Risk Insights & Analytics Module
 * Renders the circular risk score gauge, student fairness balance meter,
 * clause distribution charts, top issues, missing terms, and negotiation playbook.
 */

window.InsightsModule = (function () {
  const GAUGE_CIRCUMFERENCE = 264; // 2 * PI * 42

  function renderInsights(analysis) {
    if (!analysis) {
      resetInsights();
      return;
    }

    const emptyBanner = document.getElementById("insightsEmptyBanner");
    if (emptyBanner) emptyBanner.style.display = "none";

    renderRiskScore(analysis.risk_score || 0, analysis.risk_category || "Standard");
    renderFairnessMeter(analysis.fairness || {});
    renderClauseDistribution(analysis.distribution || {}, analysis.clauses || [], analysis.document_info || {});
    renderTopIssues(analysis.top_issues || []);
    renderMissingInformation(analysis.missing_information || []);
    renderRecommendations(analysis.clauses || []);
  }

  function renderRiskScore(score, category) {
    const valElem = document.getElementById("riskScoreValue");
    const headingElem = document.getElementById("riskCategoryHeading");
    const fillElem = document.getElementById("gaugeFill");
    const subtextElem = document.getElementById("riskScoreSubtext");

    if (valElem) valElem.innerText = score;
    if (headingElem) headingElem.innerText = category;

    // Calculate gauge stroke offset
    if (fillElem) {
      fillElem.classList.remove("high", "medium", "low");
      const offset = GAUGE_CIRCUMFERENCE - (score / 100) * GAUGE_CIRCUMFERENCE;
      fillElem.style.strokeDashoffset = offset;

      if (score >= 70) {
        fillElem.classList.add("high");
        fillElem.style.stroke = "var(--color-high-risk)";
        headingElem.style.color = "var(--color-high-risk)";
        if (subtextElem) {
          subtextElem.innerText = "Severe red flags detected! Do not sign without written amendments.";
        }
      } else if (score >= 40) {
        fillElem.classList.add("medium");
        fillElem.style.stroke = "var(--color-med-risk)";
        headingElem.style.color = "var(--color-med-risk)";
        if (subtextElem) {
          subtextElem.innerText = "Moderate caution advised. Key clauses require clarification.";
        }
      } else {
        fillElem.classList.add("low");
        fillElem.style.stroke = "var(--color-low-risk)";
        headingElem.style.color = "var(--color-low-risk)";
        if (subtextElem) {
          subtextElem.innerText = "Contract terms are balanced and within standard student norms.";
        }
      }
    }
  }

  function renderFairnessMeter(fairness) {
    const sPct = fairness.student_favor_pct || 50;
    const cPct = fairness.counterparty_favor_pct || 50;

    const sBar = document.getElementById("fairnessStudentBar");
    const cBar = document.getElementById("fairnessCounterpartyBar");
    const sText = document.getElementById("studentPctText");
    const cText = document.getElementById("counterpartyPctText");
    const ratioBadge = document.getElementById("fairnessRatioBadge");
    const verdictText = document.getElementById("fairnessVerdictText");

    if (sBar) sBar.style.width = `${sPct}%`;
    if (cBar) cBar.style.width = `${cPct}%`;
    if (sText) sText.innerText = `${sPct}%`;
    if (cText) cText.innerText = `${cPct}%`;
    if (ratioBadge) ratioBadge.innerText = `${sPct}% / ${cPct}%`;
    if (verdictText) {
      verdictText.innerText = fairness.summary_verdict || "Evaluating rights and obligation reciprocity.";
    }
  }

  function renderClauseDistribution(dist, clauses, docInfo) {
    const total = dist.total || clauses.length || 1;
    const high = dist.high || 0;
    const med = dist.medium || 0;
    const low = dist.low || 0;

    const highPct = Math.round((high / total) * 100);
    const medPct = Math.round((med / total) * 100);
    const lowPct = Math.max(0, 100 - highPct - medPct);

    const dHigh = document.getElementById("distHighBar");
    const dMed = document.getElementById("distMedBar");
    const dLow = document.getElementById("distLowBar");

    if (dHigh) dHigh.style.width = `${highPct}%`;
    if (dMed) dMed.style.width = `${medPct}%`;
    if (dLow) dLow.style.width = `${lowPct}%`;

    const cHigh = document.getElementById("distHighCount");
    const cMed = document.getElementById("distMedCount");
    const cLow = document.getElementById("distLowCount");
    const cTotal = document.getElementById("totalClausesPill");

    if (cHigh) cHigh.innerText = high;
    if (cMed) cMed.innerText = med;
    if (cLow) cLow.innerText = low;
    if (cTotal) cTotal.innerText = `${total} clauses`;

    // Render category tags
    const tagsFlow = document.getElementById("categoryTagsFlow");
    if (tagsFlow && dist.categories) {
      let tagsHtml = "";
      for (const [cat, count] of Object.entries(dist.categories)) {
        const label = cat.replace(/_/g, " ").toLowerCase();
        tagsHtml += `<span class="cat-chip">${escapeHtml(label)} (${count})</span>`;
      }
      tagsFlow.innerHTML = tagsHtml;
    }

    // Compact cross-clause link summary badge when links exist
    const crossRow = document.getElementById("crossClauseDistRow");
    const linkCount = docInfo?.cross_clause_links || 0;
    if (crossRow) {
      if (linkCount > 0) {
        crossRow.style.display = "flex";
        crossRow.innerHTML = `
          <span class="cross-link-summary-badge" title="Substantive clauses referencing or linked to risk provisions across sections">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.2">
              <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
              <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
            </svg>
            <span>${linkCount} Cross-Clause ${linkCount === 1 ? "Link" : "Links"} Detected</span>
          </span>
        `;
      } else {
        crossRow.style.display = "none";
        crossRow.innerHTML = "";
      }
    }
  }

  function renderTopIssues(issues) {
    const list = document.getElementById("topIssuesList");
    const badge = document.getElementById("topIssuesCountBadge");
    if (!list) return;

    if (badge) badge.innerText = `${issues.length} Red Flags`;

    if (!issues.length) {
      list.innerHTML = `<p class="empty-list-notice">No critical contract vulnerabilities detected.</p>`;
      return;
    }

    let html = "";
    issues.forEach((issue) => {
      const related = issue.related_clauses || [];
      const refLine = related.length
        ? `<div class="issue-crossref-tag">${escapeHtml(related[0].display_text)}</div>`
        : "";
      html += `
        <div class="top-issue-item" data-clause-index="${issue.clause_index}">
          <div class="issue-title-row">
            <span>${escapeHtml(issue.title)}</span>
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="9 18 15 12 9 6"/>
            </svg>
          </div>
          ${refLine}
          <p class="issue-snippet">"${escapeHtml(issue.snippet)}"</p>
        </div>
      `;
    });

    list.innerHTML = html;

    // Click issue to scroll to that clause in document viewer
    list.querySelectorAll(".top-issue-item").forEach((item) => {
      item.addEventListener("click", () => {
        const idx = parseInt(item.getAttribute("data-clause-index"), 10);
        if (!isNaN(idx) && window.ViewerModule) {
          window.ViewerModule.scrollToClause(idx);
        }
      });
    });
  }

  function renderMissingInformation(missing) {
    const list = document.getElementById("missingItemsList");
    const badge = document.getElementById("missingTermsCountBadge");
    if (!list) return;

    if (badge) badge.innerText = `${missing.length} Items`;

    if (!missing.length) {
      list.innerHTML = `<p class="empty-list-notice">All standard protective baseline terms present.</p>`;
      return;
    }

    let html = "";
    missing.forEach((item) => {
      const sev = (item?.severity || "").toLowerCase();
      const sevBadgeClass =
        sev === "high"
          ? "badge-danger"
          : sev === "medium"
          ? "badge-warning"
          : sev === "low"
          ? "badge-success"
          : "";
      const sevPillHtml = sev
        ? `<span class="badge missing-severity-pill ${sevBadgeClass}">${escapeHtml(sev.toUpperCase())}</span>`
        : "";
      const recHtml = item?.recommendation
        ? `<div class="missing-rec"><strong>Recommended Clause:</strong> ${escapeHtml(item.recommendation)}</div>`
        : "";

      html += `
        <div class="missing-info-item">
          <div class="missing-header-row">
            <div class="missing-title">${escapeHtml(item?.topic || "Missing Term")}</div>
            ${sevPillHtml}
          </div>
          <p class="missing-desc">${escapeHtml(item?.description || "")}</p>
          ${recHtml}
        </div>
      `;
    });

    list.innerHTML = html;
  }

  function renderRecommendations(clauses) {
    const list = document.getElementById("recommendationsList");
    if (!list) return;

    // Filter to substantive high and medium clauses with actionable steps
    const actionable = clauses.filter(
      (c) => !c.is_heading && !c.is_structural && (c.final_risk_level === "high" || c.final_risk_level === "medium") && c.recommended_action
    );

    if (!actionable.length) {
      list.innerHTML = `<p class="empty-list-notice">No special negotiation counters required.</p>`;
      return;
    }

    // Deduplicate by category
    const seenCats = new Set();
    const uniqueActions = [];
    actionable.forEach((c) => {
      if (!seenCats.has(c.category)) {
        seenCats.add(c.category);
        uniqueActions.push(c);
      }
    });

    let html = "";
    uniqueActions.slice(0, 4).forEach((item) => {
      const cat = (item.category || "General").replace(/_/g, " ");
      html += `
        <div class="action-item-box">
          <strong>${escapeHtml(cat)}</strong>
          ${escapeHtml(item.recommended_action)}
        </div>
      `;
    });

    list.innerHTML = html;
  }

  function resetInsights() {
    const emptyBanner = document.getElementById("insightsEmptyBanner");
    if (emptyBanner) emptyBanner.style.display = "flex";

    const valElem = document.getElementById("riskScoreValue");
    const headingElem = document.getElementById("riskCategoryHeading");
    const fillElem = document.getElementById("gaugeFill");
    const subtextElem = document.getElementById("riskScoreSubtext");

    if (valElem) valElem.innerText = "--";
    if (headingElem) {
      headingElem.innerText = "Awaiting Document";
      headingElem.style.color = "var(--color-navy)";
    }
    if (fillElem) {
      fillElem.classList.remove("high", "medium", "low");
      fillElem.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
      fillElem.style.stroke = "var(--color-border)";
    }
    if (subtextElem) {
      subtextElem.innerText = "Insights will appear after document analysis.";
    }

    // Fairness meter reset to neutral/empty values
    const sBar = document.getElementById("fairnessStudentBar");
    const cBar = document.getElementById("fairnessCounterpartyBar");
    const sText = document.getElementById("studentPctText");
    const cText = document.getElementById("counterpartyPctText");
    const ratioBadge = document.getElementById("fairnessRatioBadge");
    const verdictText = document.getElementById("fairnessVerdictText");

    if (sBar) sBar.style.width = "0%";
    if (cBar) cBar.style.width = "0%";
    if (sText) sText.innerText = "--";
    if (cText) cText.innerText = "--";
    if (ratioBadge) ratioBadge.innerText = "-- / --";
    if (verdictText) verdictText.innerText = "Insights will appear after document analysis.";

    // Breakdown reset
    const dHigh = document.getElementById("distHighBar");
    const dMed = document.getElementById("distMedBar");
    const dLow = document.getElementById("distLowBar");
    if (dHigh) dHigh.style.width = "0%";
    if (dMed) dMed.style.width = "0%";
    if (dLow) dLow.style.width = "0%";

    const cHigh = document.getElementById("distHighCount");
    const cMed = document.getElementById("distMedCount");
    const cLow = document.getElementById("distLowCount");
    const cTotal = document.getElementById("totalClausesPill");
    if (cHigh) cHigh.innerText = "0";
    if (cMed) cMed.innerText = "0";
    if (cLow) cLow.innerText = "0";
    if (cTotal) cTotal.innerText = "0 clauses";

    const tagsFlow = document.getElementById("categoryTagsFlow");
    if (tagsFlow) tagsFlow.innerHTML = "";

    const crossRow = document.getElementById("crossClauseDistRow");
    if (crossRow) {
      crossRow.style.display = "none";
      crossRow.innerHTML = "";
    }

    const listIssues = document.getElementById("topIssuesList");
    const badgeIssues = document.getElementById("topIssuesCountBadge");
    if (listIssues) listIssues.innerHTML = `<p class="empty-list-notice">Insights will appear after document analysis.</p>`;
    if (badgeIssues) badgeIssues.innerText = "0 Red Flags";

    const listMissing = document.getElementById("missingItemsList");
    const badgeMissing = document.getElementById("missingTermsCountBadge");
    if (listMissing) listMissing.innerHTML = `<p class="empty-list-notice">Insights will appear after document analysis.</p>`;
    if (badgeMissing) badgeMissing.innerText = "0 Items";

    const listRec = document.getElementById("recommendationsList");
    if (listRec) listRec.innerHTML = `<p class="empty-list-notice">Insights will appear after document analysis.</p>`;
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

  return {
    renderInsights: renderInsights,
    resetInsights: resetInsights
  };
})();

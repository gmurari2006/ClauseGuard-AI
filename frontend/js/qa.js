/**
 * ClauseGuard AI - Interactive Student Q&A Module
 * Powers natural language question answering over the active agreement.
 */

window.QAModule = (function () {
  let activeDocumentId = null;
  let activeDomain = null;
  let defaultPresetsCache = [];
  let inFlightQueryKey = null;
  let lastAnsweredKey = null;
  let lastSubmittedAt = 0;

  function init() {
    setupPresets(null);
    setupInputHandlers();
  }

  function setActiveDocument(docId, domain) {
    if (activeDocumentId !== docId) {
      lastAnsweredKey = null;
      inFlightQueryKey = null;
    }
    activeDocumentId = docId;
    const normalizedDomain = domain || null;
    if (normalizedDomain !== activeDomain) {
      activeDomain = normalizedDomain;
      setupPresets(activeDomain);
    }
  }

  function setActiveDocumentId(docId) {
    setActiveDocument(docId, activeDomain);
  }

  function resetQA() {
    activeDocumentId = null;
    const hadDomain = Boolean(activeDomain);
    activeDomain = null;
    inFlightQueryKey = null;
    lastAnsweredKey = null;
    lastSubmittedAt = 0;
    const stream = document.getElementById("qaMessagesStream");
    if (stream) {
      stream.innerHTML = `
        <div class="qa-welcome-bubble">
          <div class="ai-avatar">CG</div>
          <div class="qa-bubble-content">
            <h4>ClauseGuard Q&amp;A Assistant</h4>
            <p>Upload an agreement or select a sample contract, then ask anything about it in plain English. For example: <em>"Can I quit if I have semester exams?"</em>, <em>"Are they allowed to keep my original marksheets?"</em>, or <em>"Who owns the code I write?"</em></p>
          </div>
        </div>
      `;
    }
    const input = document.getElementById("qaInput");
    if (input) input.value = "";
    if (hadDomain) {
      setupPresets(null);
    }
  }

  function renderPresetChips(container, presets) {
    let html = "";
    presets.forEach((p) => {
      html += `<button type="button" class="preset-chip" data-question="${escapeHtml(p.question)}">${escapeHtml(p.topic)}</button>`;
    });
    container.innerHTML = html;

    container.querySelectorAll(".preset-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        const q = btn.getAttribute("data-question");
        askQuestion(q);
      });
    });
  }

  function setupPresets(domain) {
    const container = document.getElementById("qaPresetsList");
    if (!container) return;

    const url = domain
      ? `/api/qa/presets?domain=${encodeURIComponent(domain)}`
      : "/api/qa/presets";

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error("Presets request failed");
        return res.json();
      })
      .then((data) => {
        const presets = Array.isArray(data?.presets) ? data.presets : [];
        if (!domain && presets.length > 0) {
          defaultPresetsCache = presets;
        }
        if (presets.length > 0) {
          renderPresetChips(container, presets);
        } else if (defaultPresetsCache.length > 0) {
          renderPresetChips(container, defaultPresetsCache);
        } else if (domain) {
          // Fallback to default endpoint if domain-specific query returned empty
          return setupPresets(null);
        }
      })
      .catch((err) => {
        if (domain) {
          if (defaultPresetsCache.length > 0) {
            renderPresetChips(container, defaultPresetsCache);
          } else {
            setupPresets(null);
          }
        } else {
          console.error("Failed to load QA presets", err);
        }
      });
  }

  function setupInputHandlers() {
    const input = document.getElementById("qaInput");
    const sendBtn = document.getElementById("btnSendQa");

    if (sendBtn && input) {
      sendBtn.addEventListener("click", () => {
        const text = input.value.trim();
        if (text) {
          askQuestion(text);
          input.value = "";
        }
      });

      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          const text = input.value.trim();
          if (text) {
            askQuestion(text);
            input.value = "";
          }
        }
      });
    }
  }

  function askQuestion(query) {
    const cleanQuery = (query || "").trim();
    if (!cleanQuery) return;

    const stream = document.getElementById("qaMessagesStream");
    if (!stream) return;

    // Prevent accidental duplicate submissions (same question in-flight or immediately repeated)
    const normalizedKey = `${activeDocumentId || "none"}::${cleanQuery.toLowerCase().replace(/\s+/g, " ")}`;
    const now = Date.now();
    if (inFlightQueryKey === normalizedKey) {
      return;
    }
    if (lastAnsweredKey === normalizedKey && (now - lastSubmittedAt) < 10000) {
      stream.scrollTop = stream.scrollHeight;
      return;
    }
    lastSubmittedAt = now;

    // Append User Message
    const userRow = document.createElement("div");
    userRow.className = "qa-message-row user-row";
    userRow.innerHTML = `
      <div class="user-avatar">You</div>
      <div class="qa-message-bubble">
        <p>${escapeHtml(cleanQuery)}</p>
      </div>
    `;
    stream.appendChild(userRow);

    if (!activeDocumentId) {
      const noticeRow = document.createElement("div");
      noticeRow.className = "qa-message-row";
      noticeRow.innerHTML = `
        <div class="ai-avatar">CG</div>
        <div class="qa-message-bubble">
          <p>Please upload an agreement or select a sample contract before asking questions.</p>
        </div>
      `;
      stream.appendChild(noticeRow);
      stream.scrollTop = stream.scrollHeight;
      return;
    }

    inFlightQueryKey = normalizedKey;

    // Append Pending AI Message
    const aiRow = document.createElement("div");
    aiRow.className = "qa-message-row";
    const pendingId = "pending_" + Date.now() + "_" + Math.floor(Math.random() * 1000);
    aiRow.innerHTML = `
      <div class="ai-avatar">CG</div>
      <div class="qa-message-bubble" id="${pendingId}">
        <p><em>Analyzing contract clauses...</em></p>
      </div>
    `;
    stream.appendChild(aiRow);
    stream.scrollTop = stream.scrollHeight;

    // Send to backend
    fetch("/api/qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: cleanQuery,
        document_id: activeDocumentId
      })
    })
      .then((res) => {
        if (!res.ok) throw new Error("Q&A request failed");
        return res.json();
      })
      .then((data) => {
        if (inFlightQueryKey === normalizedKey) {
          inFlightQueryKey = null;
        }
        lastAnsweredKey = normalizedKey;

        const targetBubble = document.getElementById(pendingId);
        if (targetBubble) {
          const groundingHtml = buildGroundingPillHtml(data);
          const crossNotesHtml = buildCrossClauseNotesHtml(data?.cross_clause_notes);

          let citedHtml = "";
          if (data?.cited_clauses && data.cited_clauses.length > 0) {
            citedHtml = `<div class="cited-clauses-container"><strong style="font-size:0.72rem; color:var(--color-text-muted);">SUPPORTING CONTRACT CLAUSES:</strong>`;
            data.cited_clauses.forEach((c) => {
              const riskColor =
                c.risk_level === "high"
                  ? "var(--color-high-risk)"
                  : c.risk_level === "medium"
                  ? "var(--color-med-risk)"
                  : "var(--color-low-risk)";
              const refPrefix = c.clause_ref ? `${escapeHtml(c.clause_ref)} • ` : "";
              const isClickable = typeof c.index === "number";
              const clickAttrs = isClickable
                ? `data-clause-index="${c.index}" role="button" tabindex="0" title="Jump to ${escapeHtml(c.clause_ref || `Clause #${c.index}`)} in Document"`
                : "";
              const jumpCue = isClickable
                ? `<span class="citation-jump-cue">View in Document &rarr;</span>`
                : "";

              citedHtml += `
                <div class="cited-clause-card ${isClickable ? "clickable-citation" : ""}" ${clickAttrs}>
                  <div class="cited-clause-header">
                    <span style="font-weight:700; color:${riskColor}; font-size:0.72rem;">${refPrefix}${escapeHtml((c.category || "").replace(/_/g, " "))} (${escapeHtml((c.risk_level || "low").toUpperCase())})</span>
                    ${jumpCue}
                  </div>
                  <div style="font-family:var(--font-doc); margin-top:3px; font-size:0.80rem;">"${escapeHtml(c.text)}"</div>
                </div>
              `;
            });
            citedHtml += `</div>`;
          }

          // Safely render markdown bold and line breaks
          const formattedAnswer = escapeHtml(data?.answer || "")
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/\n\n/g, "<br/><br/>")
            .replace(/\n/g, "<br/>");
          targetBubble.innerHTML = `
            ${groundingHtml}
            <div>${formattedAnswer}</div>
            ${crossNotesHtml}
            ${citedHtml}
          `;

          // Wire clickable cited clauses to jump directly in Document tab
          targetBubble.querySelectorAll(".cited-clause-card[data-clause-index]").forEach((card) => {
            const jump = () => {
              const idx = parseInt(card.getAttribute("data-clause-index"), 10);
              if (!Number.isNaN(idx) && window.ViewerModule && window.ViewerModule.scrollToClause) {
                window.ViewerModule.scrollToClause(idx);
              }
            };
            card.addEventListener("click", jump);
            card.addEventListener("keydown", (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                jump();
              }
            });
          });

          stream.scrollTop = stream.scrollHeight;
        }
      })
      .catch((err) => {
        if (inFlightQueryKey === normalizedKey) {
          inFlightQueryKey = null;
        }
        const targetBubble = document.getElementById(pendingId);
        if (targetBubble) {
          targetBubble.innerHTML = `<p style="color:var(--color-high-risk)">Could not retrieve an answer. Ensure a document is loaded.</p>`;
        }
      });
  }

  function formatConfidenceLabel(confidence) {
    if (confidence === null || confidence === undefined || confidence === "") return "";
    if (typeof confidence === "number" && !Number.isNaN(confidence)) {
      const pct = confidence <= 1 ? Math.round(confidence * 100) : Math.round(confidence);
      const tier = pct >= 75 ? "High" : pct >= 45 ? "Medium" : "Low";
      return `${tier} Confidence (${pct}%)`;
    }
    const raw = String(confidence).trim();
    if (!raw) return "";
    const formatted = raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase();
    return formatted.toLowerCase().includes("confidence") ? formatted : `${formatted} Confidence`;
  }

  function buildGroundingPillHtml(data) {
    const statusRaw = data?.grounding_status;
    const confRaw = data?.confidence;
    if (!statusRaw && (confRaw === null || confRaw === undefined || confRaw === "")) {
      return "";
    }

    const norm = String(statusRaw || "").toLowerCase().replace(/[\s-]+/g, "_");
    let pillClass = "grounded";
    let statusText = "Grounded";

    if (norm.includes("insufficient") || norm.includes("ungrounded") || norm.includes("not_found") || norm.includes("none")) {
      pillClass = "insufficient";
      statusText = "Insufficient Evidence in Document";
    } else if (norm.includes("partial") || norm.includes("limited") || norm.includes("weak")) {
      pillClass = "partial";
      statusText = "Partially Grounded";
    } else if (norm) {
      pillClass = "grounded";
      statusText = "Grounded";
    }

    const confText = formatConfidenceLabel(confRaw);
    const label =
      pillClass === "insufficient"
        ? statusText
        : statusText && confText
        ? `${statusText} • ${confText}`
        : statusText || confText;

    return `
      <div class="qa-grounding-row">
        <span class="qa-grounding-pill ${pillClass}">
          <span class="qa-grounding-dot"></span>
          <span>${escapeHtml(label)}</span>
        </span>
      </div>
    `;
  }

  function buildCrossClauseNotesHtml(notes) {
    if (!notes) return "";
    const items = [];
    if (typeof notes === "string" && notes.trim()) {
      items.push(notes.trim());
    } else if (Array.isArray(notes)) {
      notes.forEach((n) => {
        if (typeof n === "string" && n.trim()) {
          items.push(n.trim());
        } else if (n && typeof n === "object") {
          const text = n.note || n.text || n.summary || n.display_text || "";
          if (text) items.push(text);
        }
      });
    }
    if (!items.length) return "";

    return `
      <div class="qa-cross-clause-notes">
        <div class="qa-cross-notes-title">
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2">
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
          </svg>
          <span>Cross-Clause Context</span>
        </div>
        <ul class="qa-cross-notes-list">
          ${items.map((txt) => `<li>${escapeHtml(txt)}</li>`).join("")}
        </ul>
      </div>
    `;
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
    init: init,
    setActiveDocument: setActiveDocument,
    setActiveDocumentId: setActiveDocumentId,
    resetQA: resetQA,
    askQuestion: askQuestion
  };
})();

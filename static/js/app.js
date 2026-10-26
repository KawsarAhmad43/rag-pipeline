// static/js/app.js
// Workbench static client - dark themed & Streamlit-like layout
// Supports: Mode (Local/API), Pipeline selection (GET /api/pipelines), chat (POST /api/pipelines/{id}/chat),
// file upload/ingest (POST /api/pipelines/{id}/ingest), New Session, Clear Chat.

// Configure API base (by default same origin)
const API_BASE = "";

// fallback pipelines if API unreachable
const FALLBACK_PIPELINES = [
  { id: "simple_rag", name: "Simple RAG", capabilities: { upload: true } },
  { id: "graph_rag", name: "Graph RAG", capabilities: { upload: true } },
  { id: "multimodal_rag", name: "Multimodal RAG", capabilities: { upload: true, multimodal: true } },
  { id: "agentic_rag", name: "Agentic RAG", capabilities: { upload: true, tools: true } },
];

// small helper: escape HTML
function escapeHtml(s) {
  return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

document.addEventListener("DOMContentLoaded", () => {
  // UI elements
  const modeEl = document.getElementById("mode");
  const pipelineSelect = document.getElementById("pipeline-select");
  const selectedLbl = document.getElementById("selected-pipeline");
  const modelSelect = document.getElementById("model-select");
  const toolCheckboxes = document.querySelectorAll(".tool-checkbox");

  const fileInput = document.getElementById("fileInput");
  const btnUpload = document.getElementById("btnUpload");
  const uploadProgress = document.getElementById("uploadProgress");
  const docList = document.getElementById("doc-list");
  const resetKbBtn = document.getElementById("reset-kb");
  const selectedFilesPreview = document.getElementById("selected-files-preview");
  const selectedFilesList = document.getElementById("selected-files-list");
  const processingStatus = document.getElementById("processing-status");
  const ragStatusList = document.getElementById("rag-status-list");

  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatWindow = document.getElementById("chat-window");
  const chatHistoryList = document.getElementById("chat-history");

  const btnNewSession = document.getElementById("btnNewSession");
  const btnClearChat = document.getElementById("btnClearChat");
  const kbVersionEl = document.getElementById("kb-version");

  const btnRecord = document.getElementById("btnRecord");
  const voiceSelectPanel = document.getElementById("voice-settings-panel");
  const voiceSelect = document.getElementById("voice-select");
  const ttsEnabled = document.getElementById("tts-enabled");

  // const sidebar = document.getElementById("sidebar"); // Sidebar is static in this version
  const menuToggle = document.getElementById("menu-toggle");
  const sidebar = document.querySelector(".sidebar");
  const uploadArea = document.getElementById("upload-area");
  const readyPopup = document.getElementById("ready-popup");
  const systemStatus = document.getElementById("system-status");
  const usePdfContext = document.getElementById("use-pdf-context");

  // New elements for status and attachments
  const kbMonitor = document.getElementById("kb-monitor");
  const kbStatusDot = document.getElementById("kb-status-dot");
  const kbStats = document.getElementById("kb-stats");

  // Chat attachments
  const attachInput = document.getElementById("attachInput");
  const attachPreview = document.getElementById("attach-preview");
  const attachName = document.getElementById("attach-name");
  const clearAttachBtn = document.getElementById("clear-attach");



  let mediaRecorder = null;
  let audioChunks = [];

  // state (persist in localStorage to be helpful)
  const state = {
    sessionId: localStorage.getItem("workbench_session_id") || null,
    pipeline: localStorage.getItem("workbench_selected_pipeline") || "simple_rag",
    mode: localStorage.getItem("workbench_mode") || "remote",
    model: localStorage.getItem("workbench_model") || "Gemma 3-4B",
    tools: JSON.parse(localStorage.getItem("workbench_tools") || "[]"),
    docs: JSON.parse(localStorage.getItem("workbench_docs") || "[]"),
    chats: JSON.parse(localStorage.getItem("workbench_chats") || "[]"),
    kb_version: Number(localStorage.getItem("workbench_kb_version") || "1"),
  };

  // FORCE API MODE for first-time use with the new Docker stack
  if (!localStorage.getItem("workbench_docker_migrated_v3")) {
    console.log("Migrating state to Docker API mode...");
    state.mode = "remote";
    state.sessionId = null; // Clear old session ID to force fresh dashed UUID
    localStorage.setItem("workbench_docker_migrated_v3", "true");
    saveState();
  } else if (state.sessionId && !state.sessionId.includes("-") && !state.sessionId.startsWith("local-")) {
    console.log("Clearing legacy hexadecimal session ID...");
    state.sessionId = null;
    saveState();
  }

  // UI sync helpers
  function saveState() {
    localStorage.setItem("workbench_session_id", state.sessionId || "");
    localStorage.setItem("workbench_selected_pipeline", state.pipeline);
    localStorage.setItem("workbench_mode", state.mode);
    localStorage.setItem("workbench_model", state.model);
    localStorage.setItem("workbench_tools", JSON.stringify(state.tools));
    localStorage.setItem("workbench_docs", JSON.stringify(state.docs));
    localStorage.setItem("workbench_chats", JSON.stringify(state.chats));
    localStorage.setItem("workbench_kb_version", String(state.kb_version));
  }

  // Migration: If the user has old "Local" or "API" strings in storage, flip them to "remote"
  if (state.mode === "Local" || state.mode === "API") {
    state.mode = "remote";
    saveState();
  }

  async function renderSelectedPipelineLabel() {
    const opt = pipelineSelect.options[pipelineSelect.selectedIndex];
    const name = opt ? opt.text : state.pipeline;

    let statusHtml = `<span style="color: #FFC107;">● Waiting for context...</span>`;

    // Check readiness if session exists and documents are tracked
    if (state.pipeline === "multimodal_rag") {
      kbMonitor.style.display = "block";
      if (state.sessionId) {
        try {
          const r = await fetch(`${API_BASE}/api/pipelines/multimodal_rag/status/${state.sessionId}`);
          if (r.ok) {
            const json = await r.json();
            if (json.ready) {
              const countText = json.doc_count > 0 ? `${json.doc_count} segments` : "Empty";
              statusHtml = `<span style="color: #4CAF50;">● Ready (${countText})</span>`;
              kbStatusDot.style.background = "var(--success)";
              kbStats.textContent = `Indexed: ${json.doc_count} chunks stored.`;

              // Trigger highlight if previously was not ready
              if (!kbStatusDot.classList.contains('ready-glow')) {
                kbStatusDot.classList.add('ready-glow');
                // Optional: also glow the label
                document.getElementById('system-status').classList.add('ready-glow');
                setTimeout(() => {
                  document.getElementById('system-status').classList.remove('ready-glow');
                }, 4000);
              }

            } else {
              kbStatusDot.style.background = "#FFC107";
              kbStats.textContent = json.doc_count > 0 ? `Embedding: ${json.doc_count} chunks...` : "Empty collection...";
              kbStatusDot.classList.remove('ready-glow');
            }
          }
        } catch (e) {
          console.warn("Status check failed", e);
        }
      } else {
        kbStats.textContent = "Start a chat to initialize monitor.";
      }
    } else {
      kbMonitor.style.display = "none";
      // Fallback for non-multimodal pipelines
      if (state.docs.length > 0) {
        statusHtml = `<span style="color: #4CAF50;">● Ready (${state.docs.length} file${state.docs.length > 1 ? 's' : ''})</span>`;
      } else {
        // Even if system is ready, warn if no docs (but don't say Waiting...)
        statusHtml = `<span style="color: #4CAF50;">● System Ready (No files)</span>`;
      }
    }

    selectedLbl.innerHTML = `${name} | ${statusHtml}`;
  }

  function showReadyPopup() {
    readyPopup.classList.add("visible");
    setTimeout(() => {
      readyPopup.classList.remove("visible");
    }, 4000);
  }

  function renderDocList() {
    docList.innerHTML = "";
    if (!state.docs.length) {
      const li = document.createElement("li");
      li.textContent = "No uploaded documents";
      li.className = "muted";
      docList.appendChild(li);
      return;
    }
    state.docs.forEach((d, i) => {
      const li = document.createElement("li");
      const left = document.createElement("div");
      // Add a green checkmark if the file is confirmed uploaded/indexed
      const checkmark = d.uploaded ? '<span style="color:#4CAF50; font-weight:bold; margin-right:6px;" title="Indexed & Ready">✓</span>' : '';
      left.innerHTML = `${checkmark}${escapeHtml(d.name)}`;
      const actions = document.createElement("div");
      actions.innerHTML = `<button data-i="${i}" class="view-doc btn ghost">👁</button>
                           <button data-i="${i}" class="del-doc btn ghost">❌</button>`;
      li.appendChild(left);
      li.appendChild(actions);
      docList.appendChild(li);
    });

    // attach handlers
    docList.querySelectorAll(".view-doc").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const i = Number(btn.dataset.i);
        alert(`View not implemented for ${state.docs[i].name}. Path: ${state.docs[i].backend_paths || []}`);
      });
    });
    docList.querySelectorAll(".del-doc").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const i = Number(btn.dataset.i);
        state.docs.splice(i, 1);
        state.kb_version++;
        kbVersionEl.textContent = `KB version: ${state.kb_version}`;
        saveState();
        renderDocList();
      });
    });
  }

  function renderChatHistory() {
    chatHistoryList.innerHTML = "";
    if (!state.chats.length) {
      const li = document.createElement("li");
      li.textContent = "No chat history";
      li.className = "muted";
      chatHistoryList.appendChild(li);
      return;
    }
    state.chats.forEach((c, i) => {
      const li = document.createElement("li");
      li.textContent = `${i + 1}. ${c.user}`;
      li.addEventListener("click", () => {
        chatInput.value = c.user;
      });
      chatHistoryList.appendChild(li);
    });
  }

  function renderChatWindow() {
    chatWindow.innerHTML = "";
    state.chats.forEach(c => {
      // User message
      const userBox = document.createElement("div");
      userBox.className = "message user";
      userBox.innerHTML = `
        <div class="bubble">${escapeHtml(c.user)}</div>
        <div class="info">Just now</div>
      `;
      chatWindow.appendChild(userBox);

      // Bot message
      const botBox = document.createElement("div");
      botBox.className = "message bot";

      // NEW LOGIC: Support "Thinking..." as a distinct state
      let bubbleClass = "bubble";
      if (c.answer === "Thinking...") bubbleClass += " thinking-bubble";

      let botContent = `<div class="${bubbleClass}">${escapeHtml(c.answer || "...")}</div>`;

      // show sources if present
      if (c.sources && c.sources.length) {
        let srcHtml = `<div class="info" style="margin-top:4px; font-weight:600;">Sources:</div>`;
        c.sources.forEach((s, idx) => {
          srcHtml += `<div class="info" style="font-size:10px;">${idx + 1}) ${s.title || "source"}${s.page ? ` (p.${s.page})` : ''}</div>`;
        });
        botContent += srcHtml;
      }

      // debug
      if (c.debug && Object.keys(c.debug).length > 0) {
        botContent += `<details class="info"><summary style="cursor:pointer">Debug Info</summary><pre style="font-size:10px; background:rgba(0,0,0,0.2); padding:4px; border-radius:4px; margin-top:4px;">${JSON.stringify(c.debug, null, 2)}</pre></details>`;
      }

      botBox.innerHTML = botContent;
      chatWindow.appendChild(botBox);
    });
    // Ensure auto-scroll
    setTimeout(() => {
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }, 50);
  }

  // API helpers
  async function fetchPipelines() {
    try {
      systemStatus.innerHTML = '<span class="status-dot"></span> Checking API...';
      const r = await fetch(`${API_BASE}/api/pipelines`);
      if (!r.ok) throw new Error("no api");
      const data = await r.json();
      // Orchestrator returns {"pipelines": ["id1", "id2"]}
      const rawList = data.pipelines || data;
      // Normalize to [{id, name, ...}]
      const normalized = (Array.isArray(rawList) ? rawList : []).map(p => {
        if (typeof p === "string") {
          return { id: p, name: p.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') };
        }
        return p;
      });

      populatePipelines(normalized);
      systemStatus.classList.remove("error");
      systemStatus.innerHTML = '<span class="status-dot"></span> System Online';
      systemStatus.style.color = "var(--success)";
      systemStatus.style.borderColor = "#d1fae5";
      systemStatus.style.background = "#ecfdf5";
      return normalized;
    } catch (e) {
      console.warn("fetchPipelines failed:", e);
      populatePipelines(FALLBACK_PIPELINES);
      systemStatus.innerHTML = '<span class="status-dot"></span> System Offline';
      systemStatus.style.color = "var(--danger)";
      systemStatus.style.borderColor = "#fecaca";
      systemStatus.style.background = "#fee2e2";
      return FALLBACK_PIPELINES;
    }
  }

  function populatePipelines(list) {
    pipelineSelect.innerHTML = "";
    list.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name || p.id;
      pipelineSelect.appendChild(opt);
    });
    // restore selection if possible
    const idx = Array.from(pipelineSelect.options).findIndex(o => o.value === state.pipeline);
    pipelineSelect.selectedIndex = idx >= 0 ? idx : 0;
    state.pipeline = pipelineSelect.value;
    renderSelectedPipelineLabel();
  }

  async function createSession() {
    if (state.sessionId && state.mode === "remote") return state.sessionId;
    if (state.mode !== "remote") {
      // make a local pseudo session
      state.sessionId = `local-${Date.now()}`;
      saveState();
      return state.sessionId;
    }
    try {
      const r = await fetch(`${API_BASE}/api/sessions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      const j = await r.json();
      state.sessionId = j.session_id;
      saveState();
      return state.sessionId;
    } catch (e) {
      console.error("createSession failed", e);
      alert("Failed to create session: " + e);
      return null;
    }
  }

  async function callChat(pipeline, sessionId, message, options = {}) {
    const isMultimodal = pipeline === "multimodal_rag";


    if (isMultimodal) {
      // Use FormData for multimodal or when attachment is present
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("user_input", message);
      formData.append("generation_mode", state.mode === "remote" ? "api" : "local");

      if (options.attachment) {
        formData.append("attachment", options.attachment);
      }



      const r = await fetch(`${API_BASE}/api/pipelines/${encodeURIComponent(pipeline)}/chat`, {
        method: "POST",
        body: formData
      });
      if (!r.ok) {
        const txt = await r.text();
        throw new Error(txt || `status ${r.status}`);
      }
      return r.json();
    } else {
      // Standard JSON chat
      const payload = { session_id: sessionId, message, options };
      console.log("📤 Sending to orchestrator:", { url: `${API_BASE}/api/pipelines/${pipeline}/chat`, payload });
      const r = await fetch(`${API_BASE}/api/pipelines/${encodeURIComponent(pipeline)}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!r.ok) {
        const txt = await r.text();
        throw new Error(txt || `status ${r.status}`);
      }
      return r.json();
    }
  }

  function uploadFilesXHR(pipeline, sessionId, files, onProgress) {
    return new Promise((resolve, reject) => {
      const url = `${API_BASE}/api/pipelines/${encodeURIComponent(pipeline)}/ingest`;
      const xhr = new XMLHttpRequest();
      const form = new FormData();
      for (let i = 0; i < files.length; i++) {
        form.append("files", files[i]);
      }
      form.append("session_id", sessionId || "");
      xhr.open("POST", url, true);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const json = JSON.parse(xhr.responseText);
            resolve(json);
          } catch (err) {
            reject("Invalid JSON from server");
          }
        } else {
          reject(`Upload failed (${xhr.status}) ${xhr.responseText}`);
        }
      };
      xhr.onerror = () => reject("Network error during upload");
      xhr.send(form);
    });
  }

  // Event handlers
  pipelineSelect.addEventListener("change", () => {
    state.pipeline = pipelineSelect.value;
    renderSelectedPipelineLabel();
    saveState();
  });

  modeEl.addEventListener("change", () => {
    state.mode = modeEl.value;
    saveState();
  });

  if (modelSelect) {
    modelSelect.addEventListener("change", () => {
      state.model = modelSelect.value;
      saveState();
    });
  }

  toolCheckboxes.forEach(cb => {
    cb.addEventListener("change", () => {
      state.tools = Array.from(document.querySelectorAll(".tool-checkbox:checked")).map(x => x.value);
      saveState();
    });
    // set initial checked if present
    if (state.tools.includes(cb.value)) cb.checked = true;
  });

  // File input change listener to show selected files
  fileInput.addEventListener("change", () => {
    updateSelectedFilesPreview();
  });

  function updateSelectedFilesPreview() {
    const files = fileInput.files;
    if (files && files.length > 0) {
      selectedFilesPreview.style.display = "block";
      selectedFilesList.innerHTML = "";
      Array.from(files).forEach((file) => {
        const fileItem = document.createElement("div");
        fileItem.textContent = `📄 ${file.name}`;
        fileItem.style.padding = "2px 0";
        selectedFilesList.appendChild(fileItem);
      });
    } else {
      selectedFilesPreview.style.display = "none";
      selectedFilesList.innerHTML = "";
    }
  }

  // drag & drop
  // uploadArea is already defined at top
  uploadArea.addEventListener("dragover", (e) => { e.preventDefault(); uploadArea.classList.add("drag"); });
  uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag"));
  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault(); uploadArea.classList.remove("drag");
    const files = Array.from(e.dataTransfer.files);
    if (files && files.length) {
      // put files into file input (not strictly necessary) and handle upload
      fileInput.files = e.dataTransfer.files;
      // show filenames after drag & drop
      updateSelectedFilesPreview();
    }
  });

  // Upload button with parallel RAG type processing
  btnUpload.addEventListener("click", async () => {
    const files = fileInput.files;
    if (!files || files.length === 0) {
      alert("Select files first (drag-drop or Browse)");
      return;
    }
    if (state.mode !== "remote") {
      // local: add to docs state only
      for (let i = 0; i < files.length; i++) {
        if (!state.docs.find(d => d.name === files[i].name)) {
          state.docs.push({ name: files[i].name, backend_paths: [], uploaded: false });
        }
      }
      state.kb_version++;
      kbVersionEl.textContent = `KB version: ${state.kb_version}`;
      saveState();
      renderDocList();
      return;
    }

    // remote: upload to ALL RAG types (except graph_rag)
    const sess = await createSession();
    if (!sess) return;

    // Define all RAG types (excluding graph_rag as requested)
    const ragTypes = [
      { id: "simple_rag", name: "Simple RAG" },
      { id: "multimodal_rag", name: "Multimodal RAG" },
      { id: "agentic_rag", name: "Agentic RAG" }
    ];

    // Show processing status section
    processingStatus.style.display = "block";
    ragStatusList.innerHTML = "";
    selectedFilesPreview.style.display = "none";

    // Create status items for each RAG type
    const statusElements = {};
    ragTypes.forEach(rag => {
      const statusItem = document.createElement("div");
      statusItem.className = "rag-status-item processing";
      statusItem.innerHTML = `
        <div class="status-spinner"></div>
        <div class="rag-status-name">${rag.name}</div>
        <div class="rag-status-message">Processing...</div>
      `;
      ragStatusList.appendChild(statusItem);
      statusElements[rag.id] = statusItem;
    });

    // Upload to all RAG types in parallel
    const uploadPromises = ragTypes.map(async (rag) => {
      const statusItem = statusElements[rag.id];
      try {
        const res = await uploadFilesXHR(rag.id, sess, files, null);

        // Extract chunk count from various response formats
        let chunkCount = 0;

        // Try to get chunk_count from ingest_result
        if (res.ingest_result) {
          // Direct chunk_count field
          chunkCount = res.ingest_result.chunk_count || 0;

          // Parse from message like "PDF processed and stored successfully - 42 chunks extracted"
          if (!chunkCount && res.ingest_result.message) {
            const match = res.ingest_result.message.match(/(\d+)\s+chunks?/i);
            if (match) {
              chunkCount = parseInt(match[1], 10);
            }
          }
        }

        // Fallback to top-level chunk_count
        if (!chunkCount) {
          chunkCount = res.chunk_count || 0;
        }

        // Update status to success
        statusItem.className = "rag-status-item success";
        statusItem.innerHTML = `
          <div class="status-icon">✓</div>
          <div class="rag-status-name">${rag.name}</div>
          <div class="rag-status-message">Indexed ${chunkCount} chunks</div>
        `;
        return { success: true, ragType: rag.id, chunkCount };
      } catch (err) {
        // Update status to error
        statusItem.className = "rag-status-item error";
        statusItem.innerHTML = `
          <div class="status-icon">✗</div>
          <div class="rag-status-name">${rag.name}</div>
          <div class="rag-status-message">Failed: ${err.message || err}</div>
        `;
        return { success: false, ragType: rag.id, error: err };
      }
    });

    // Wait for all uploads to complete
    const results = await Promise.all(uploadPromises);

    // Check if all succeeded
    const allSucceeded = results.every(r => r.success);
    const successCount = results.filter(r => r.success).length;

    // Update doc list
    for (let i = 0; i < files.length; i++) {
      const name = files[i].name;
      if (!state.docs.find(d => d.name === name)) {
        state.docs.push({ name, backend_paths: [], uploaded: allSucceeded });
      }
    }
    state.kb_version++;
    kbVersionEl.textContent = `KB version: ${state.kb_version}`;
    saveState();
    renderDocList();

    if (allSucceeded) {
      uploadProgress.textContent = "All uploads complete!";
      showReadyPopup();

      // Success message in chat
      const totalChunks = results.reduce((sum, r) => sum + (r.chunkCount || 0), 0);
      const info = document.createElement("div");
      info.className = "message bot";
      info.innerHTML = `<div class="bubble">
        <strong>Ingestion Complete</strong><br>
        Processed ${files.length} file(s) across ${successCount} RAG pipeline(s).<br>
        ✅ <strong>Total Embeddings Created:</strong> ${totalChunks} chunks indexed successfully.
      </div>`;
      chatWindow.appendChild(info);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    } else {
      uploadProgress.textContent = `${successCount}/${ragTypes.length} pipelines succeeded`;
    }

    // Clear file input and hide processing after delay
    fileInput.value = "";
    setTimeout(() => {
      uploadProgress.textContent = "";
      processingStatus.style.display = "none";
    }, 5000);
  });

  // Click upload area to trigger file input
  uploadArea.addEventListener("click", () => {
    fileInput.click();
  });

  menuToggle.addEventListener("click", () => {
    if (window.innerWidth <= 800) {
      sidebar.classList.toggle("open");
      sidebar.classList.remove("collapsed"); // Ensure no conflict
    } else {
      sidebar.classList.toggle("collapsed");
      sidebar.classList.remove("open"); // Ensure no conflict
    }
  });

  // Close sidebar when clicking outside (optional UX improvement)
  document.addEventListener("click", (e) => {
    if (window.innerWidth <= 800 &&
      sidebar.classList.contains("open") &&
      !sidebar.contains(e.target) &&
      e.target !== menuToggle) {
      sidebar.classList.remove("open");
    }
  });

  resetKbBtn.addEventListener("click", async () => {
    if (!confirm("⚠️ WARNING: This will delete ALL vector database data from ALL RAG types (Simple RAG, Multimodal RAG, Agentic RAG).\n\nThis action cannot be undone. Continue?")) return;

    const sess = state.sessionId;
    if (sess && state.mode === "remote") {
      uploadProgress.textContent = "Resetting all RAG databases...";

      try {
        // Single DELETE call to orchestrator which handles all RAG types
        const resp = await fetch(`${API_BASE}/api/sessions/${sess}`, {
          method: "DELETE",
          headers: { "Content-Type": "application/json" }
        });

        if (resp.ok) {
          const result = await resp.json();
          console.log("Reset result:", result);

          const successCount = result.summary?.success || 0;
          const total = result.summary?.total || 0;

          if (result.status === "completed") {
            uploadProgress.textContent = `✅ All RAG databases reset successfully! (${successCount}/${total})`;
          } else {
            uploadProgress.textContent = `⚠️ Reset completed: ${successCount}/${total} succeeded`;
            if (result.summary?.errors?.length > 0) {
              console.error("Reset errors:", result.summary.errors);
            }
          }
        } else {
          const errorText = await resp.text();
          console.error("Reset failed:", errorText);
          uploadProgress.textContent = "❌ Reset failed. Check console for details.";
        }
      } catch (e) {
        console.error("Error during reset:", e);
        uploadProgress.textContent = "❌ Error during reset: " + e.message;
      }

      setTimeout(() => {
        uploadProgress.textContent = "";
      }, 3000);
    }

    // Clear local state
    state.docs = [];
    state.chats = [];
    state.sessionId = null;
    state.kb_version = 1;
    kbVersionEl.textContent = `KB version: ${state.kb_version}`;
    saveState();
    renderDocList();
    renderChatWindow();
    renderSelectedPipelineLabel();

    // Show confirmation in chat
    const info = document.createElement("div");
    info.className = "message bot";
    info.innerHTML = `<div class="bubble">
      <strong>🗑️ Reset Complete</strong><br>
      All vector database data has been cleared from all RAG pipelines.<br>
      You can now upload new documents.
    </div>`;
    chatWindow.appendChild(info);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  });

  // Chat submit
  chatForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;
    // add user bubble
    state.chats.push({ user: text, answer: "", sources: [], debug: null });
    saveState();
    renderChatWindow();
    chatInput.value = "";

    // remote or local
    console.log("Current Mode:", state.mode);

    // RESTORED: Need session ID for API calls
    const sess = await createSession();
    if (!sess) { alert("No session available"); return; }

    // # OLD LOGIC (Manual thinking bubble - Re-commented for duplication fix)
    // const thinking = document.createElement("div");
    // thinking.className = "message bot";
    // thinking.innerHTML = `<div class="bubble">Thinking...</div>`;
    // chatWindow.appendChild(thinking);
    // chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
      // Get enabled tools
      const enabledTools = Array.from(document.querySelectorAll(".tool-checkbox:checked"))
        .map(checkbox => checkbox.value);

      const pdfAnalyzerEnabled = enabledTools.includes("pdf_analyzer");
      const webSearchEnabled = enabledTools.includes("web_search");

      const options = {
        top_k: 5,
        pdf_context: usePdfContext.checked,  // Original feature: use uploaded knowledge base
        pdf_analyzer: pdfAnalyzerEnabled,     // New tool: advanced PDF analysis
        web_search: webSearchEnabled          // Web search tool
      };

      // Handle attachment if any
      if (attachInput && attachInput.files && attachInput.files.length > 0) {
        options.attachment = attachInput.files[0];
      }



      // NEW LOGIC: State-driven Thinking... bubble
      state.chats[state.chats.length - 1].answer = "Thinking...";
      renderChatWindow();

      console.log("🔍 DEBUG: About to call chat with session_id:", sess);
      console.log("🔍 DEBUG: Pipeline:", state.pipeline);
      console.log("🔍 DEBUG: Options:", options);

      const out = await callChat(state.pipeline, sess, text, options);

      // # OLD LOGIC (Manual DOM update)
      // if (out.answer) {
      //   thinking.querySelector(".bubble").innerText = out.answer;
      // } else {
      //   thinking.querySelector(".bubble").innerText = "(no answer)";
      // }

      // NEW LOGIC: Update state and re-render
      state.chats[state.chats.length - 1].answer = out.answer || "(no answer)";
      state.chats[state.chats.length - 1].sources = out.sources || [];
      state.chats[state.chats.length - 1].debug = out.debug || {};

      saveState();
      renderChatWindow();

      // Clear attachment after successful send
      if (attachInput) attachInput.value = "";
      if (attachPreview) attachPreview.style.display = "none";
    } catch (err) {
      // # OLD LOGIC
      // thinking.querySelector(".bubble").innerText = "Error: " + (err.message || err);

      // NEW LOGIC
      state.chats[state.chats.length - 1].answer = "Error: " + (err.message || err);
      renderChatWindow();
    }
  });

  // New session & clear chat buttons
  btnNewSession.addEventListener("click", async () => {
    state.sessionId = null;
    state.chats = [];
    state.docs = [];
    state.kb_version = 1;
    saveState();

    // Explicitly create a new session on the backend
    await createSession();

    renderChatWindow();
    renderDocList();
    renderChatHistory();
    renderSelectedPipelineLabel();
    kbVersionEl.textContent = `KB version: ${state.kb_version}`;
    console.log("New session initialized:", state.sessionId);
  });

  btnClearChat.addEventListener("click", () => {
    if (!confirm("Clear current chat display?")) return;
    state.chats = [];
    saveState();
    renderChatWindow();
  });

  // --- Voice Recording Logic ---
  btnRecord.addEventListener("click", async () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
      btnRecord.classList.remove("recording");
      btnRecord.innerText = "🎤";
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
          await sendVoiceChat(audioBlob);
        };

        mediaRecorder.start();
        btnRecord.classList.add("recording");
        btnRecord.innerText = "🛑";
      } catch (err) {
        alert("Microphone access denied or error: " + err);
      }
    }
  });

  async function sendVoiceChat(blob) {
    const sess = await createSession();
    if (!sess) return;

    // Show thinking bubble
    const thinking = document.createElement("div");
    thinking.className = "bubble bot";
    thinking.innerText = "Transcribing & Thinking...";
    chatWindow.appendChild(thinking);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    const formData = new FormData();
    formData.append("session_id", sess);
    formData.append("audio_file", blob, "input.wav");
    formData.append("generation_mode", state.mode === "remote" ? "api" : "local");
    formData.append("voice", voiceSelect.value);

    try {
      const resp = await fetch(`${API_BASE}/api/pipelines/multimodal_rag/voice_chat`, {
        method: "POST",
        body: formData
      });

      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();

      thinking.innerText = data.text || "(no answer)";

      // Play audio
      if (data.audio_path) {
        // The backend returns a local path, but Nginx serves it at /audio/
        const filename = data.audio_path.split('/').pop();
        const audioUrl = `${API_BASE}/audio/${filename}`;
        const audio = new Audio(audioUrl);
        audio.play();

        // Add audio player to chat
        const player = document.createElement("audio");
        player.controls = true;
        player.src = audioUrl;
        player.className = "voice-audio-msg";
        chatWindow.appendChild(player);
      }

      state.chats.push({ user: "🎤 (Voice Message)", answer: data.text, audio_url: data.audio_path });
      saveState();
    } catch (err) {
      thinking.innerText = "Voice Error: " + err.message;
    }
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  // Attachment listeners
  if (attachInput) {
    attachInput.addEventListener("change", () => {
      const file = attachInput.files[0];
      if (file) {
        attachPreview.style.display = "flex";
        attachName.innerText = file.name;
      } else {
        attachPreview.style.display = "none";
      }
    });
  }

  if (clearAttachBtn) {
    clearAttachBtn.addEventListener("click", () => {
      attachInput.value = "";
      attachPreview.style.display = "none";
    });
  }

  // Selective UI

  pipelineSelect.addEventListener("change", () => {
    const isMultimodal = pipelineSelect.value === "multimodal_rag";
    if (voiceSelectPanel) voiceSelectPanel.style.display = isMultimodal ? "block" : "none";
    if (btnRecord) btnRecord.style.display = isMultimodal ? "inline-block" : "none";

    // Toggle Tools panel for Agentic/Tools capability
    const toolsPanel = document.getElementById("tools-panel");
    const currentPipeline = FALLBACK_PIPELINES.find(p => p.id === pipelineSelect.value) || { capabilities: {} };
    if (toolsPanel) {
      const hasTools = currentPipeline.capabilities && currentPipeline.capabilities.tools;
      toolsPanel.style.display = hasTools ? "block" : "none";
    }

    state.pipeline = pipelineSelect.value;
    saveState();
    renderSelectedPipelineLabel();
  });

  // Status Polling Loop (Always active, checks state)
  setInterval(() => {
    // Only poll if we are in a pipeline that supports status checks
    const statusCapable = ["multimodal_rag", "agentic_rag", "simple_rag", "graph_rag"];
    if (statusCapable.includes(state.pipeline) && state.sessionId) {
      renderSelectedPipelineLabel();
    }
  }, 3000);

  // Initial trigger
  pipelineSelect.dispatchEvent(new Event("change"));
  renderChatWindow();
  renderChatHistory();
  // initial load
  (async () => {
    // fetch pipelines if API mode
    await fetchPipelines();

    // UI initial values
    if (modeEl) modeEl.value = state.mode || "remote";
    if (modelSelect) modelSelect.value = state.model || "Gemma 3-4B";
    if (pipelineSelect) pipelineSelect.value = state.pipeline || pipelineSelect.value;
    renderSelectedPipelineLabel();
    if (kbVersionEl) kbVersionEl.textContent = `KB version: ${state.kb_version}`;

    renderDocList();
    renderChatHistory();
    renderChatWindow();
  })();
});

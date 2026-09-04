const INTENTS = ["decipher", "kite", "rescue", "rotate", "travel", "search", "gate", "idle"];
const state = { selected: null, summary: null, sessions: [], frameIndex: 0, playTimer: null };

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function setStatus(message, kind = "") {
  const node = $("operation-status");
  node.textContent = message || "";
  node.className = `operation-status ${kind}`;
}

function showError(error) {
  const panel = $("error-panel");
  const list = $("error-list");
  const messages = String(error.message || error).split("; ");
  list.replaceChildren(...messages.map((message) => { const item = document.createElement("div"); item.textContent = message; return item; }));
  panel.hidden = false;
  setStatus("Operation failed", "error");
}

function clearError() { $("error-panel").hidden = true; $("error-list").replaceChildren(); }

function renderSessions(sessions) {
  const list = $("session-list");
  list.replaceChildren();
  if (!sessions.length) { list.innerHTML = '<div class="empty">No sessions found.</div>'; return; }
  sessions.forEach((session) => {
    const button = document.createElement("button");
    button.className = `session-item ${state.selected === session.name ? "active" : ""}`;
    button.dataset.session = session.name;
    const status = session.raw_valid ? "valid" : "warning";
    button.innerHTML = `<span class="session-icon">${session.raw_valid ? "●" : "!"}</span><span class="session-copy"><strong></strong><small>${session.frame_count} frames · ${session.duration_s.toFixed(1)}s</small></span><span class="file-status"><i class="${session.has_actions ? "on" : ""}">A</i><i class="${session.has_intents ? "on" : ""}">I</i><i class="${session.has_v5_chunks ? "on" : ""}">V5</i></span>`;
    button.querySelector("strong").textContent = session.name;
    button.querySelector(".session-icon").classList.add(status);
    button.addEventListener("click", () => selectSession(session.name));
    list.append(button);
  });
}

async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    state.sessions = data.sessions || [];
    renderSessions(data.sessions || []);
    if (!state.selected && data.sessions && data.sessions.length) selectSession(data.sessions[0].name);
  } catch (error) { showError(error); }
}

function renderActionChart(counts) {
  const chart = $("action-chart");
  chart.replaceChildren();
  const entries = Object.entries(counts || {});
  if (!entries.length) { chart.innerHTML = '<div class="empty">No action file yet.</div>'; return; }
  const max = Math.max(...entries.map(([, value]) => value), 1);
  entries.sort((a, b) => b[1] - a[1]).slice(0, 9).forEach(([name, count]) => {
    const row = document.createElement("div"); row.className = "bar-row";
    row.innerHTML = `<div class="bar-label"><span></span><strong>${count}</strong></div><div class="bar-track"><i style="width:${Math.max(3, count / max * 100)}%"></i></div>`;
    row.querySelector("span").textContent = name;
    chart.append(row);
  });
}

function focusFrame(frameId) {
  if (frameId == null || frameId === "") return;
  const frameIds = state.summary?.frame_ids || [];
  const numericId = Number(frameId);
  if (!Number.isFinite(numericId) || !frameIds.length) return;
  let index = frameIds.indexOf(numericId);
  if (index < 0) {
    index = frameIds.reduce((best, value, candidate) => Math.abs(value - numericId) < Math.abs(frameIds[best] - numericId) ? candidate : best, 0);
  }
  state.frameIndex = index;
  const currentId = frameIds[index];
  $("frame-input").value = currentId;
  $("frame-slider").value = index;
  $("frame-current").textContent = `Frame ${currentId}`;
  $("frame-total").textContent = frameIds.length ? `${frameIds[frameIds.length - 1]}` : "—";
  $("frame-time").textContent = state.summary.frame_times?.[String(currentId)] != null ? `${Number(state.summary.frame_times[String(currentId)]).toFixed(3)}s` : "—";
  const frameEvents = (state.summary.events || []).filter((event) => event.frame_id === currentId);
  $("frame-events").textContent = frameEvents.length ? `${frameEvents.length} input event${frameEvents.length === 1 ? "" : "s"} on this frame` : "No input events on this frame";
  $("frame-image").hidden = false;
  $("frame-empty").hidden = true;
  $("frame-image").src = `/api/sessions/${encodeURIComponent(state.selected)}/frames/${currentId}`;
  document.querySelectorAll("#event-table tr.selected").forEach((node) => node.classList.remove("selected"));
  document.querySelectorAll(`#event-table tr[data-frame-id="${currentId}"]`).forEach((node) => node.classList.add("selected"));
}

function renderFrameViewer(summary) {
  stopFramePlayback();
  const frameIds = summary.frame_ids || [];
  const slider = $("frame-slider");
  slider.min = "0";
  slider.max = String(Math.max(frameIds.length - 1, 0));
  slider.disabled = !frameIds.length;
  $("frame-input").disabled = !frameIds.length;
  $("frame-previous").disabled = !frameIds.length;
  $("frame-next").disabled = !frameIds.length;
  $("frame-play").disabled = !frameIds.length;
  if (!frameIds.length) {
    $("frame-image").hidden = true;
    $("frame-empty").hidden = false;
    $("frame-current").textContent = "Frame —";
    $("frame-total").textContent = "—";
    return;
  }
  focusFrame(frameIds[0]);
}

function stepFrame(delta) {
  const frameIds = state.summary?.frame_ids || [];
  if (!frameIds.length) return;
  const nextIndex = Math.max(0, Math.min(frameIds.length - 1, state.frameIndex + delta));
  focusFrame(frameIds[nextIndex]);
  if (nextIndex === frameIds.length - 1 && delta > 0) stopFramePlayback();
}

function stopFramePlayback() {
  if (state.playTimer) window.clearInterval(state.playTimer);
  state.playTimer = null;
  $("frame-play").textContent = "▶ Play";
  $("frame-play").setAttribute("aria-label", "Play frames");
}

function toggleFramePlayback() {
  if (state.playTimer) { stopFramePlayback(); return; }
  const frameIds = state.summary?.frame_ids || [];
  if (!frameIds.length) return;
  if (state.frameIndex >= frameIds.length - 1) focusFrame(frameIds[0]);
  state.playTimer = window.setInterval(() => stepFrame(1), 1000 / Math.max(Number(state.summary.fps) || 10, 1));
  $("frame-play").textContent = "⏸ Pause";
  $("frame-play").setAttribute("aria-label", "Pause frames");
}

function renderKeyStats(counts) {
  const stats = $("key-stats"); stats.replaceChildren();
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) { stats.innerHTML = '<div class="empty">No key events.</div>'; return; }
  entries.slice(0, 16).forEach(([key, count]) => {
    const chip = document.createElement("div"); chip.className = "key-chip";
    const label = document.createElement("span"); label.textContent = key;
    const value = document.createElement("strong"); value.textContent = count;
    chip.append(label, value); stats.append(chip);
  });
}

function renderEvents(events) {
  const body = $("event-table").querySelector("tbody"); body.replaceChildren();
  if (!events || !events.length) { body.innerHTML = '<tr><td colspan="5" class="empty">No input events yet.</td></tr>'; return; }
  events.forEach((event) => {
    const row = document.createElement("tr"); row.dataset.frameId = event.frame_id ?? "";
    const frameLabel = event.end_frame_id != null && event.end_frame_id !== event.frame_id ? `${event.frame_id}–${event.end_frame_id}` : (event.frame_id ?? "—");
    const eventName = String(event.kind || "").replace("key_", "").toUpperCase() || "—";
    const eventLabel = event.repeat_count > 1 ? `${eventName} ×${event.repeat_count}` : eventName;
    const values = [frameLabel, event.time_s == null ? "—" : `${Number(event.time_s).toFixed(3)}s`, event.key || "—", eventLabel, event.value ?? "—"];
    values.forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
    row.addEventListener("click", () => {
      focusFrame(event.frame_id);
      row.classList.add("selected");
      row.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    body.append(row);
  });
}

function intentSelect(value) {
  const select = document.createElement("select");
  INTENTS.forEach((intent) => { const option = new Option(intent, intent, intent === value, intent === value); select.add(option); });
  return select;
}

function renderIntents(rows) {
  const body = $("intent-table").querySelector("tbody"); body.replaceChildren();
  if (!rows || !rows.length) { body.innerHTML = '<tr><td colspan="6" class="empty">No intent segments yet.</td></tr>'; return; }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const id = document.createElement("td"); id.textContent = row.segment_id || "";
    const start = document.createElement("input"); start.type = "number"; start.value = row.start_frame ?? ""; start.min = "0"; start.dataset.field = "start_frame";
    const end = document.createElement("input"); end.type = "number"; end.value = row.end_frame ?? ""; end.min = "0"; end.dataset.field = "end_frame";
    const startCell = document.createElement("td"); startCell.append(start);
    const endCell = document.createElement("td"); endCell.append(end);
    const intentCell = document.createElement("td"); const select = intentSelect(row.intent); select.dataset.field = "intent"; intentCell.append(select);
    const reason = document.createElement("td"); reason.className = "reason"; reason.textContent = row.candidate_reason || "";
    const notesCell = document.createElement("td"); const notes = document.createElement("input"); notes.type = "text"; notes.value = row.notes || ""; notes.dataset.field = "notes"; notesCell.append(notes);
    tr.dataset.segmentId = row.segment_id || "";
    tr.append(id, startCell, endCell, intentCell, reason, notesCell); body.append(tr);
  });
}

function collectIntents() {
  return [...$("intent-table").querySelectorAll("tbody tr")].map((row) => {
    const value = (field) => row.querySelector(`[data-field="${field}"]`)?.value ?? "";
    return { segment_id: row.dataset.segmentId, start_frame: value("start_frame"), end_frame: value("end_frame"), intent: value("intent"), candidate_reason: row.cells[4].textContent, notes: value("notes") };
  });
}

function renderSummary(summary) {
  state.summary = summary;
  $("session-title").textContent = state.selected;
  $("selected-session").textContent = state.selected;
  $("session-meta").textContent = `${summary.metadata.task_instruction || summary.metadata.task_name || "raw VLA recording"} · ${summary.metadata.mode || "unknown mode"}`;
  $("metric-frames").textContent = summary.frame_count;
  $("metric-range").textContent = `frames ${summary.frame_start ?? "—"}–${summary.frame_end ?? "—"}`;
  $("metric-duration").textContent = `${Number(summary.duration_s || 0).toFixed(1)}s`;
  $("metric-fps").textContent = `${Number(summary.fps || 0).toFixed(1)} FPS`;
  $("metric-raw").textContent = summary.raw_errors.length ? "Needs review" : "Valid";
  $("metric-raw").className = summary.raw_errors.length ? "bad" : "good";
  $("metric-outputs").textContent = `${summary.has_actions ? "A" : "–"} / ${summary.has_intents ? "I" : "–"} / ${summary.v5_chunks?.exists ? "V5" : "–"} / ${summary.v5_chunks?.audit_exists ? "✓" : "–"}`;
  const auditText = !summary.v5_chunks?.audit_exists ? "audit missing" : summary.v5_chunks.audit_gate_pass ? "audit gate passed" : `audit gate blocked (${summary.v5_chunks.audit_conflicts || 0} conflicts)`;
  $("metric-output-detail").textContent = summary.v5_chunks?.exists ? `${summary.v5_chunks.count} vla_chunks_v5 records · ${auditText}` : "actions / intents / v5 chunks / audit";
  $("move-frames").textContent = `${summary.movement_frames} / ${summary.frame_count}`;
  $("camera-frames").textContent = `${summary.camera_frames} / ${summary.frame_count}`;
  $("event-total").textContent = summary.event_count || 0;
  $("key-total").textContent = summary.key_event_count || 0;
  $("mean-values").innerHTML = Object.entries(summary.mean_abs || {}).map(([key, value]) => `<span>${key}<strong>${Number(value).toFixed(3)}</strong></span>`).join("");
  renderActionChart(summary.action_categories); renderKeyStats(summary.key_counts); renderEvents(summary.events); renderFrameViewer(summary); renderIntents(summary.intent_rows);
  if (summary.raw_errors.length) showError({ message: summary.raw_errors.join("; ") }); else clearError();
}

async function selectSession(name) {
  stopFramePlayback(); state.selected = name; renderSessions(state.sessions);
  try { $("empty-state").hidden = true; $("session-view").hidden = false; setStatus("Loading session..."); renderSummary(await api(`/api/sessions/${encodeURIComponent(name)}/summary`)); setStatus("Ready", "success"); }
  catch (error) { showError(error); }
  loadSessions();
}

async function runOperation(button, path, method = "POST", payload = {}) {
  if (!state.selected) return;
  button.disabled = true; clearError(); setStatus("Working...");
  try { await api(`/api/sessions/${encodeURIComponent(state.selected)}${path}`, { method, body: JSON.stringify(payload) }); await selectSession(state.selected); setStatus("Saved", "success"); }
  catch (error) { showError(error); }
  finally { button.disabled = false; }
}

$("refresh-sessions").addEventListener("click", loadSessions);
$("frame-previous").addEventListener("click", () => stepFrame(-1));
$("frame-next").addEventListener("click", () => stepFrame(1));
$("frame-play").addEventListener("click", toggleFramePlayback);
$("frame-slider").addEventListener("input", (event) => {
  const frameIds = state.summary?.frame_ids || [];
  if (frameIds.length) focusFrame(frameIds[Number(event.currentTarget.value)]);
});
$("frame-input").addEventListener("change", (event) => focusFrame(event.currentTarget.value));
$("frame-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); focusFrame(event.currentTarget.value); event.currentTarget.blur(); }
});
$("extract-actions").addEventListener("click", (event) => runOperation(event.currentTarget, "/actions"));
$("init-intents").addEventListener("click", (event) => runOperation(event.currentTarget, "/intents/init"));
$("build-chunks").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true; clearError(); setStatus("Building vla_chunks_v5 and audit...");
  try {
    const result = await api(`/api/sessions/${encodeURIComponent(state.selected)}/chunks/build`, { method: "POST", body: JSON.stringify({}) });
    await selectSession(state.selected);
    setStatus(`Generated ${result.chunk_count} chunks → ${result.output} · ${result.audit_output}`, result.gate_pass ? "success" : "error");
  } catch (error) { showError(error); }
  finally { button.disabled = false; }
});
$("save-intents").addEventListener("click", (event) => runOperation(event.currentTarget, "/intents", "PUT", collectIntents()));
loadSessions();

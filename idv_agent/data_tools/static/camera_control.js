const OPTIONS = {
  camera_control_phase: ["search", "target_acquire", "target_align", "hold"],
  camera_target_id: ["target_cipher", "other_visible", "none"],
  camera_steering_mode: ["target_center", "path_follow", "search_sweep", "hold"],
  path_strategy: ["direct", "detour_left", "detour_right", "unknown"],
  desired_turn_dx: [-2, -1, 0, 1, 2], desired_turn_dy: [-2, -1, 0, 1, 2],
  status: ["pending", "complete"],
};
const state = { data: null, split: "train", rowIndex: 0, context: [] };
const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}
function error(message = "") { $("error").hidden = !message; $("error").textContent = message; }
function status(message, kind = "") { $("save-status").textContent = message; $("save-status").className = kind; }
function selected() { return state.data?.splits?.[state.split]?.rows?.[state.rowIndex] || null; }
function selectField(field, value) {
  const row = selected(); if (!row) return;
  row[field] = value;
  if (field === "camera_control_phase" && value === "hold") {
    row.camera_steering_mode = "hold"; row.path_strategy = "unknown";
    row.desired_turn_dx = 0; row.desired_turn_dy = 0;
  }
  if (field === "camera_steering_mode") {
    if (value === "hold") { row.camera_control_phase = "hold"; row.path_strategy = "unknown"; row.desired_turn_dx = 0; row.desired_turn_dy = 0; }
    if (value !== "path_follow" && value !== "hold") row.path_strategy = "unknown";
  }
  renderFields();
}

function renderQueue() {
  const tabs = $("split-tabs"); tabs.replaceChildren();
  for (const split of ["train", "val"]) { const meta = state.data.splits[split]; const b = document.createElement("button"); b.className = `button small ${split === state.split ? "primary" : "secondary"}`; b.textContent = `${split} ${meta.complete}/${meta.rows.length}`; b.onclick = () => { state.split = split; state.rowIndex = 0; loadContext(); render(); }; tabs.append(b); }
  const meta = state.data.splits[state.split]; $("queue-stats").textContent = `当前 ${state.split}：${meta.complete} 完成 / ${meta.pending} 待标注\n读取：${meta.source}\n输出：${meta.output}`;
}
function renderChoices(field) {
  const box = document.querySelector(`[data-field="${field}"]`); const row = selected(); box.replaceChildren();
  OPTIONS[field].forEach((option) => { const button = document.createElement("button"); const value = String(option); const disabled = field === "path_strategy" && row.camera_steering_mode !== "path_follow" && option !== "unknown"; button.className = `choice ${String(row[field]) === value ? "selected" : ""}`; button.textContent = value; button.disabled = disabled; button.onclick = () => selectField(field, option); box.append(button); });
}
function renderFields() { Object.keys(OPTIONS).forEach(renderChoices); }
function renderTarget() {
  const row = selected(), g = row.existing_grounding;
  $("row-id").textContent = row.id; $("frame-title").textContent = `${row.session} · frame ${row.frame}`;
  const actionStart = row.frame + 2, actionEnd = actionStart + 5;
  $("grounding").textContent = `观察窗口 frame ${Math.max(0, row.frame - 21)}–${row.frame}；现有 grounding：${g.cipher_bbox_xyxy_norm ? "cipher bbox" : "无 bbox"} · target_side=${g.target_side}`;
  $("target-image").src = `/api/camera-control/${state.split}/rows/${encodeURIComponent(row.id)}/frames/${row.frame}`;
  $("frame-caption").innerHTML = `<strong>观察帧 ${row.frame}</strong><span>上下文结束点；标签应描述此刻可得信息下的控制意图</span>`;
  $("replay-window").textContent = `frame ${actionStart}–${actionEnd}`;
  $("replay-camera").textContent = `dx ${row.replay_camera_dx}, dy ${row.replay_camera_dy}`; $("target-side").textContent = g.target_side; $("interact-prompt").textContent = g.interact_prompt ? "yes" : "no"; $("reachable").textContent = g.cipher_reachable ? "yes" : "no";
  const bbox = $("bbox"), b = g.cipher_bbox_xyxy_norm; bbox.hidden = !b;
  if (b) { bbox.style.left = `${b[0] * 100}%`; bbox.style.top = `${b[1] * 100}%`; bbox.style.width = `${(b[2] - b[0]) * 100}%`; bbox.style.height = `${(b[3] - b[1]) * 100}%`; }
}
function renderContext() { const box = $("context"); box.replaceChildren(); const row = selected(); state.context.filter(x => x.exists).forEach((item, index) => { const figure = document.createElement("figure"); const img = document.createElement("img"); img.loading = "lazy"; img.src = `/api/camera-control/${state.split}/rows/${encodeURIComponent(row.id)}/frames/${item.frame}`; img.alt = `frame ${item.frame}`; const c = document.createElement("figcaption"); c.textContent = `frame ${item.frame}${index === 0 ? " · h0 start" : index === 5 ? " · h0 end" : ""}`; figure.append(img, c); box.append(figure); }); }
function render() { renderQueue(); const row = selected(); $("empty").hidden = Boolean(row); $("editor").hidden = !row; if (!row) return; renderTarget(); renderContext(); renderFields(); }
async function loadContext() { const row = selected(); if (!row) { state.context = []; return; } state.context = (await api(`/api/camera-control/${state.split}/rows/${encodeURIComponent(row.id)}/context`)).frames; }
async function load() { error(); status("Loading"); state.data = await api("/api/camera-control/sets"); const rows = state.data.splits[state.split].rows; const pending = rows.findIndex(x => x.status !== "complete"); state.rowIndex = pending < 0 ? 0 : pending; await loadContext(); render(); status("Ready", "success"); }
async function save(goNext = false) { const row = selected(); if (!row) return; error(); status("Saving"); const annotation = Object.fromEntries(Object.keys(OPTIONS).map(key => [key, row[key]])); try { const result = await api(`/api/camera-control/${state.split}/rows/${encodeURIComponent(row.id)}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ annotation }) }); state.data.splits[state.split].rows[state.rowIndex] = result.row; const meta = state.data.splits[state.split]; meta.complete = meta.rows.filter(x => x.status === "complete").length; meta.pending = meta.rows.length - meta.complete; meta.source = meta.output; if (goNext) nextPending(); render(); status("Saved", "success"); } catch (e) { error(e.message); status("Save failed", "error"); } }
function nextPending() { const rows = state.data.splits[state.split].rows; const start = state.rowIndex; for (let i = 1; i <= rows.length; i += 1) { const index = (start + i) % rows.length; if (rows[index].status !== "complete") { state.rowIndex = index; loadContext().then(render).catch(e => error(e.message)); return; } } }
$("refresh").onclick = () => load().catch(e => { error(e.message); status("Load failed", "error"); });
$("save").onclick = () => save(); $("next").onclick = () => save(true); $("next-pending").onclick = nextPending;
load().catch(e => { error(e.message); status("Load failed", "error"); });

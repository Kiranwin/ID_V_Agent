# VLA Data Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local browser workbench that creates `per_frame_actions.csv`, creates editable candidate `intent_segments.csv`, and presents session diagnostics under the configured raw-session root.

**Architecture:** A Python standard-library HTTP server bound to `127.0.0.1` serves a compact native HTML/CSS/JS workbench. Focused backend modules handle safe session discovery, summaries, atomic intent writes, and orchestration of the existing validation/extraction/initialization functions. The browser edits intent rows only; action rows remain derived and read-only.

**Tech Stack:** Python 3, `http.server`, `json`, `csv`, `pathlib`, existing pandas-based label extraction, native HTML/CSS/JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-data-workbench-design.md`

## Global Constraints

- Only operate on directories under `data/vla_raw_sessions` or the CLI `--root` supplied by the user.
- Do not add runtime dependencies.
- Do not edit `per_frame_actions.csv` manually in the UI.
- Do not connect to the game or send input.
- Preserve existing user files; candidate initialization must refuse overwrite by default.
- Keep all documentation under `docs/`.
- Run focused tests after each task and finish with `python -m idv_agent.scripts.smoke_test` and `pytest`.

---

### Task 1: Backend session store and summaries

**Files:**
- Create: `idv_agent/data_tools/__init__.py`
- Create: `idv_agent/data_tools/session_store.py`
- Test: `tests/test_data_workbench.py`

**Interfaces:**
- Consumes: a configured `Path` raw-session root and existing session files.
- Produces: `SessionStore(root)`, `list_sessions()`, `resolve_session(name)`, `summary(name)`, and `atomic_write_intents(name, rows)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_session_store_lists_and_summarizes_session(tmp_path):
    session = make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    store = SessionStore(tmp_path / "sessions")

    assert [item["name"] for item in store.list_sessions()] == ["s1"]
    summary = store.summary("s1")
    assert summary["frame_count"] == 3
    assert summary["frame_start"] == 0
    assert summary["frame_end"] == 2
    assert summary["raw_errors"] == []

def test_session_store_rejects_path_escape(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    with pytest.raises(ValueError, match="session"):
        store.resolve_session("..\\outside")

def test_atomic_intent_write_preserves_existing_file_on_validation_error(tmp_path):
    session = make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    target = session / "intent_segments.csv"
    target.write_text("original", encoding="utf-8")
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(ValueError):
        store.atomic_write_intents("s1", [{"start_frame": 0, "end_frame": 2, "intent": "bad"}])
    assert target.read_text(encoding="utf-8") == "original"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workbench.py -q`
Expected: FAIL because `idv_agent.data_tools.session_store` and `SessionStore` do not exist.

- [ ] **Step 3: Implement the minimal store**

Implement root resolution and containment with `Path.resolve()` and `Path.is_relative_to()`; enumerate only child directories containing `meta.json` or `frames`; load metadata and CSVs with the standard library; delegate raw and intent validation to existing validators; compute counts and bounded frame previews; validate full coverage against the actual frame range; write a UTF-8 BOM CSV to a sibling temporary path and replace the target only after validation succeeds.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data_workbench.py -q`
Expected: PASS.

- [ ] **Step 5: Refactor only after green**

Keep path checks and CSV serialization in small private helpers if the implementation duplicates them. Re-run the focused test after refactoring.

### Task 2: Workbench orchestration

**Files:**
- Create: `idv_agent/data_tools/workbench.py`
- Modify: `tests/test_data_workbench.py`

**Interfaces:**
- Consumes: `SessionStore`; existing `extract_session`, `init_intent_segments.init`, `validate_vla_raw.validate`.
- Produces: `DataWorkbench(store)`, `sessions()`, `session_summary(name)`, `extract_actions(name)`, and `init_intents(name, overwrite=False)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_actions_refuses_invalid_raw_session(tmp_path):
    session = tmp_path / "sessions" / "bad"
    session.mkdir(parents=True)
    (session / "meta.json").write_text("{}", encoding="utf-8")
    workbench = DataWorkbench(SessionStore(tmp_path / "sessions"))

    with pytest.raises(ValueError, match="raw"):
        workbench.extract_actions("bad")

def test_init_intents_refuses_existing_file_without_overwrite(tmp_path):
    session = make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    (session / "per_frame_actions.csv").write_text(
        "frame_id,category_name,text_action\n0,NOOP,\n1,NOOP,\n2,NOOP,\n", encoding="utf-8"
    )
    (session / "intent_segments.csv").write_text("sentinel", encoding="utf-8")
    workbench = DataWorkbench(SessionStore(tmp_path / "sessions"))

    with pytest.raises(FileExistsError):
        workbench.init_intents("s1")
    assert (session / "intent_segments.csv").read_text(encoding="utf-8") == "sentinel"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workbench.py::test_extract_actions_refuses_invalid_raw_session tests/test_data_workbench.py::test_init_intents_refuses_existing_file_without_overwrite -q`
Expected: FAIL because `DataWorkbench` does not exist.

- [ ] **Step 3: Implement orchestration**

Resolve the session through the store, run raw validation before extraction, translate validator errors into `ValueError`, call `extract_session`, and return the resulting row count. For intent initialization, call the existing initializer with its default candidate boundary behavior and pass through `overwrite`; convert the existing overwrite error to `FileExistsError` for the API layer.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_data_workbench.py -q`
Expected: PASS.

### Task 3: HTTP server and CLI

**Files:**
- Create: `idv_agent/scripts/data_workbench.py`
- Modify: `tests/test_data_workbench.py`

**Interfaces:**
- Consumes: `DataWorkbench` and static assets.
- Produces: `WorkbenchHandler`, `create_server(root, host, port)`, and CLI `python -m idv_agent.scripts.data_workbench`.

- [ ] **Step 1: Write the failing request tests**

```python
def test_http_api_lists_sessions_and_returns_summary(tmp_path):
    make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    server = start_test_server(tmp_path / "sessions")
    try:
        assert get_json(server, "/api/sessions")["sessions"][0]["name"] == "s1"
        summary = get_json(server, "/api/sessions/s1/summary")
        assert summary["frame_count"] == 3
    finally:
        server.shutdown()

def test_http_api_rejects_unknown_session(tmp_path):
    server = start_test_server(tmp_path / "sessions")
    try:
        response = request(server, "/api/sessions/unknown/summary")
        assert response.status == 404
        assert "error" in json.loads(response.read())
    finally:
        server.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_workbench.py -k http -q`
Expected: FAIL because the server module and routes do not exist.

- [ ] **Step 3: Implement the server**

Use `ThreadingHTTPServer` with a handler carrying one `DataWorkbench` instance. Serve `/` and files below the static asset directory. Parse URL paths with `urllib.parse`, decode JSON request bodies with a bounded content length, and route the five API operations from the spec. Return JSON with explicit status codes 200, 400, 404, 409, and 500. Never interpolate arbitrary paths into file reads; session names go through `SessionStore.resolve_session`.

- [ ] **Step 4: Run request tests**

Run: `pytest tests/test_data_workbench.py -k http -q`
Expected: PASS.

- [ ] **Step 5: Add CLI behavior and test it**

Add `--root`, `--host` defaulting to `127.0.0.1`, and `--port` defaulting to `8765`. Print the local URL and serve until interrupted. Test `create_server` directly so the suite does not leave a process running.

### Task 4: Browser workbench UI

**Files:**
- Create: `idv_agent/data_tools/static/index.html`
- Create: `idv_agent/data_tools/static/app.js`
- Create: `idv_agent/data_tools/static/styles.css`
- Modify: `tests/test_data_workbench.py`

**Interfaces:**
- Consumes: the HTTP contract from Task 3.
- Produces: a usable single-page workbench with session selection, generation controls, diagnostics, frame previews, and intent editing.

- [ ] **Step 1: Write the failing static asset test**

```python
def test_http_root_serves_workbench_markup(tmp_path):
    server = start_test_server(tmp_path / "sessions")
    try:
        response = request(server, "/")
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert "session-list" in body
        assert "intent-table" in body
        assert "per-frame-actions" in body
    finally:
        server.shutdown()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_data_workbench.py::test_http_root_serves_workbench_markup -q`
Expected: FAIL because the static asset does not exist.

- [ ] **Step 3: Implement the static UI**

Build a responsive, dense tool layout with a dark neutral shell, a warm accent for active/dirty state, and clearly separated sidebar/overview/review regions. Use native controls and no external assets or libraries. `app.js` loads sessions, selects a session, renders summary values and diagnostics, invokes extraction/initialization, renders image previews using same-origin `/api` preview responses or safe static paths, serializes edited intent rows, and displays server errors inline. Keep the action summary read-only and make the intent table inputs explicit.

- [ ] **Step 4: Run static and API tests**

Run: `pytest tests/test_data_workbench.py -q`
Expected: PASS.

### Task 5: Documentation and integration verification

**Files:**
- Modify: `docs/05-操作手册.md`
- Modify: `docs/00-当前状态.md`
- Modify: `tests/test_data_workbench.py` only if a discovered regression needs coverage.

**Interfaces:**
- Consumes: the final CLI command and UI behavior from Tasks 1-4.
- Produces: documented startup/usage instructions and an accurate current-state entry.

- [ ] **Step 1: Add the workbench command to the operator manual**

Document `python -m idv_agent.scripts.data_workbench --root data/vla_raw_sessions`, the local URL, the action extraction flow, the intent editing/save flow, and the fact that the tool is offline/read-only with respect to game input.

- [ ] **Step 2: Update current status**

Add a dated entry under current code/data status describing the data workbench and its focused test coverage. Update the next-step list only where this task changes it; do not rewrite historical metrics.

- [ ] **Step 3: Run focused and full verification**

Run:

```powershell
pytest tests/test_data_workbench.py -q
python -m idv_agent.scripts.smoke_test
pytest
```

Expected: all commands exit 0; the focused suite includes the new store, orchestration, HTTP, and static UI checks.

- [ ] **Step 4: Run a live local smoke check**

Start `python -m idv_agent.scripts.data_workbench --root data/vla_raw_sessions --port 8765`, request `/` and `/api/sessions` locally, then stop the server. Confirm no files are changed by simply opening the UI; use a temporary test session for generation/save behavior.

- [ ] **Step 5: Inspect final worktree**

Run `git status --short` and verify there are no temporary files, generated test outputs, or changes outside the scoped files and the required docs.


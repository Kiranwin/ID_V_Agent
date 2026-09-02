# VLA Data Workbench Design

**Goal:** Provide a local browser workbench that turns a raw VLA session into `per_frame_actions.csv` and an editable, validated `intent_segments.csv`, while showing enough session information for human review.

**Scope:** This feature covers offline data preparation for directories under `data/vla_raw_sessions`. It does not edit per-frame actions, generate VLA chunks, perform visual annotation, inject input, connect to the game, or replace existing command-line tools.

## User Flow

1. Start the local workbench with the raw-session root.
2. Select a session from the session list.
3. Review raw integrity status and session metadata.
4. Run deterministic action extraction. The operation creates or replaces `per_frame_actions.csv` only after extraction succeeds.
5. Generate candidate intent segments. Existing intent files are never overwritten unless the user explicitly requests a rebuild.
6. Review frame previews, action distribution, movement/camera statistics, and candidate segment boundaries.
7. Edit only `start_frame`, `end_frame`, `intent`, and `notes` in the intent table.
8. Save the intent CSV. The workbench atomically replaces the file only after the submitted rows pass schema and full-coverage validation.

## Architecture

The implementation uses a Python standard-library HTTP server bound to `127.0.0.1` and a static browser UI served by the same process. No dependency is added. The server receives a configured raw-session root and rejects any requested session path that is not contained by that root.

The backend is split into focused modules:

- `idv_agent/data_tools/session_store.py`: safe root/session discovery, metadata loading, CSV summaries, frame preview lookup, and atomic intent writes.
- `idv_agent/data_tools/workbench.py`: application operations that compose existing raw validation, extraction, candidate initialization, and intent validation functions.
- `idv_agent/scripts/data_workbench.py`: CLI entry point and HTTP routes.
- `idv_agent/data_tools/static/index.html`, `app.js`, and `styles.css`: browser UI.

Existing domain logic remains authoritative:

- `idv_agent.labels.extract.extract_session` produces `per_frame_actions.csv`.
- `idv_agent.scripts.init_intent_segments.init` produces candidate segment CSVs.
- `idv_agent.scripts.validate_vla_raw.validate` validates raw sessions.
- `idv_agent.scripts.validate_intent_segments.validate` validates intent segments.

## HTTP Contract

All responses are JSON except `GET /`, which serves the UI.

- `GET /api/sessions`: returns discovered sessions with name, frame count, duration, FPS, and booleans for raw validity, actions existence, and intents existence.
- `GET /api/sessions/{session}/summary`: returns metadata, validation errors, action counts/statistics, intent rows, frame range, and a bounded list of preview frames.
- `POST /api/sessions/{session}/actions`: runs raw validation and deterministic extraction. It returns the new action row count and summary. It refuses extraction when raw validation has errors.
- `POST /api/sessions/{session}/intents/init`: creates candidate segments. It returns a conflict when the intent file already exists unless `overwrite=true` is explicitly supplied.
- `PUT /api/sessions/{session}/intents`: accepts a JSON array of rows, validates full frame coverage and allowed intents, then atomically writes `intent_segments.csv`.

The API uses plain JSON error objects with HTTP 400 for invalid input, 404 for unknown sessions, 409 for overwrite conflicts, and 500 for unexpected processing failures. Error messages are human-readable and do not expose paths outside the configured root.

## UI Layout

The page is a compact workbench rather than a marketing page:

- Header: tool name, selected session, and server status.
- Sidebar: session list with raw/actions/intents status indicators.
- Overview: frame count, duration, FPS, validation state, and processing actions.
- Diagnostics: action category distribution, movement/camera coverage, and intent coverage.
- Review area: bounded thumbnail strip with frame IDs and an editable intent-segment table.
- Status area: operation progress/result and validation errors.

The action table is not editable. The intent table uses native number inputs, a select containing the frozen eight intents (`decipher`, `kite`, `rescue`, `rotate`, `travel`, `search`, `gate`, `idle`), and a text input for notes. The UI never displays a control for sending input or interacting with the game.

## Data Safety

- Session names are treated as directory names, not arbitrary paths.
- The server resolves every session path and checks that it remains below the configured root.
- Intent saves use a sibling temporary file followed by replacement, so a failed validation or write does not destroy the previous file.
- Candidate initialization defaults to non-overwrite.
- Extraction and initialization are synchronous, local, and offline; no network access is made by the application.

## Testing and Acceptance

Tests will cover:

- session discovery and summary values from a minimal raw session;
- rejection of traversal/absolute paths outside the configured root;
- action extraction refusal on invalid raw data and successful output creation on valid data;
- candidate intent initialization and refusal to overwrite an existing intent file by default;
- intent save validation for invalid intents, overlaps, gaps, and complete coverage;
- preservation of the original intent file when atomic save validation or replacement fails;
- HTTP status and JSON shape for the core routes;
- static UI response and presence of the session/summary/editing workflow.

Acceptance requires the workbench to run with `python -m idv_agent.scripts.data_workbench --root data/vla_raw_sessions`, show existing sessions, create the two requested outputs for a valid raw session, edit and save intent segments, surface validation errors without corrupting existing data, pass focused tests, pass `python -m idv_agent.scripts.smoke_test`, and pass the full test suite.


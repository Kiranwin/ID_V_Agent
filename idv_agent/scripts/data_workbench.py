"""Serve the local browser workbench for offline VLA session preparation."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

from idv_agent.data_tools.session_store import SessionStore
from idv_agent.data_tools.review_store import ReviewConflict
from idv_agent.data_tools.workbench import DataWorkbench


MAX_BODY_BYTES = 8 * 1024 * 1024


class WorkbenchHandler(BaseHTTPRequestHandler):
    """HTTP routes for one configured ``DataWorkbench`` instance."""

    workbench: DataWorkbench
    static_dir: Path
    protocol_version = "HTTP/1.0"

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, value: object, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _read_json(self) -> object:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("请求必须为 JSON 对象")
        return value

    def _path_parts(self) -> list[str]:
        return [unquote(part) for part in urlsplit(self.path).path.split("/") if part]

    def do_GET(self) -> None:
        try:
            parts = self._path_parts()
            if not parts:
                return self._send_file(self.static_dir / "index.html")
            if parts == ["api", "sessions"]:
                return self._send_json({"schema": "idv.workbench.v6", "sessions": self.workbench.sessions()})
            if parts == ["api", "mvp-v6", "readiness"]:
                return self._send_json(self.workbench.annotation_readiness())
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "summary":
                return self._send_json(self.workbench.session_summary(parts[2]))
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "mvp-v6":
                return self._send_json(self.workbench.mvp_v6_summary(parts[2]))
            if len(parts) == 5 and parts[:2] == ["api", "sessions"] and parts[3:] == ["mvp-v6", "rows"]:
                workspace = parse_qs(urlsplit(self.path).query).get("workspace", [None])[0]
                return self._send_json(self.workbench.review_rows(parts[2], workspace=workspace))
            if len(parts) == 5 and parts[:2] == ["api", "sessions"] and parts[3] == "frames":
                session = self.workbench.store.resolve_session(parts[2])
                if not parts[4].isdigit(): raise ValueError("frame must be an integer")
                path = (session / "frames" / f"{int(parts[4]):08d}.jpg").resolve()
                if path.parent != (session / "frames").resolve() or not path.is_file():
                    raise FileNotFoundError("frame not found")
                return self._send_file(path)
            if parts[0] == "static" and len(parts) == 2 and parts[1] in {"app.js", "styles.css"}:
                path = (self.static_dir / parts[1]).resolve()
                if path.parent != self.static_dir.resolve() or not path.is_file(): raise FileNotFoundError("asset not found")
                return self._send_file(path)
            self._send_error(HTTPStatus.NOT_FOUND, "route not found")
        except FileNotFoundError as exc: self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc: self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc: self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            parts = self._path_parts()
            if len(parts) != 5 or parts[:3] != ["api", "sessions", parts[2]] or parts[3] != "mvp-v6":
                return self._send_error(HTTPStatus.NOT_FOUND, "V6 route not found")
            name, operation, payload = parts[2], parts[4], self._read_json()
            if operation == "save-row":
                result = self.workbench.update_review_row(name, workspace=payload.get("workspace"),
                    row_id=payload["row_id"], patch=payload["patch"], revision=payload["revision"], reviewer=payload["reviewer"])
            elif operation == "check-review":
                result = self.workbench.check_review(name, workspace=payload.get("workspace"), reviewer=payload["reviewer"])
            elif operation == "finish-review":
                result = self.workbench.finish_review(name, workspace=payload.get("workspace"), reviewer=payload["reviewer"])
            elif operation == "export-q-all":
                result = self.workbench.export_q_all(name, workspace=payload.get("workspace"))
            elif operation == "prepare": result = self.workbench.prepare_mvp_v6(name, output=payload.get("output"))
            elif operation == "import": result = self.workbench.import_mvp_v6(name, workspace=payload.get("workspace"), output=payload.get("output"), annotator=str(payload.get("annotator", "")), completion_note=str(payload.get("completion_note", "")), split=str(payload.get("split", "")), scenario_group=str(payload.get("scenario_group", "")))
            elif operation == "validate": result = self.workbench.validate_mvp_v6(name, workspace=payload.get("workspace"), prompt_only=bool(payload.get("prompt_only", False)), require_reviewed=bool(payload.get("require_reviewed", False)))
            elif operation == "review": result = self.workbench.review_mvp_v6(name, workspace=payload.get("workspace"), csv_path=str(payload["csv_path"]), output=payload.get("output"), reviewer=str(payload.get("reviewer", "")))
            elif operation == "export-q": result = self.workbench.export_q_mvp_v6(name, workspace=payload.get("workspace"), output=str(payload["output"]))
            elif operation == "export": result = self.workbench.export_mvp_v6(name, workspace=payload.get("workspace"), output=str(payload["output"]))
            else: return self._send_error(HTTPStatus.NOT_FOUND, "unknown V6 operation")
            self._send_json(result)
        except ReviewConflict as exc: self._send_error(HTTPStatus.CONFLICT, str(exc))
        except FileExistsError as exc: self._send_error(HTTPStatus.CONFLICT, str(exc))
        except FileNotFoundError as exc: self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except (KeyError, ValueError) as exc: self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc: self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_PUT(self) -> None:
        self._send_error(HTTPStatus.METHOD_NOT_ALLOWED,
                         "V6 workbench is versioned; use POST /mvp-v6/<operation>")

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            return self._send_error(HTTPStatus.NOT_FOUND, "file not found")
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

def create_server(root: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    store = SessionStore(root)
    workbench = DataWorkbench(store)
    static_dir = Path(__file__).resolve().parents[1] / "data_tools" / "static"
    handler = type("ConfiguredWorkbenchHandler", (WorkbenchHandler,),
                   {"workbench": workbench, "static_dir": static_dir})
    return ThreadingHTTPServer((host, port), handler)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动本地 VLA 数据处理工作台")
    parser.add_argument("--root", type=Path, default=Path("data/raw_sessions"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = create_server(args.root, args.host, args.port)
    print(f"[workbench-v6] http://{args.host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

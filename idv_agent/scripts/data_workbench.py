"""Serve the local browser workbench for offline VLA session preparation."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from idv_agent.data_tools.session_store import SessionStore
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
        return value

    def _path_parts(self) -> list[str]:
        return [unquote(part) for part in urlsplit(self.path).path.split("/") if part]

    def do_GET(self) -> None:
        try:
            parts = self._path_parts()
            if not parts:
                return self._send_file(self.static_dir / "index.html")
            if parts[0] == "api" and parts[1:] == ["sessions"]:
                return self._send_json({"sessions": self.workbench.sessions()})
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "summary":
                return self._send_json(self.workbench.session_summary(parts[2]))
            if len(parts) == 5 and parts[:2] == ["api", "sessions"] and parts[3] == "frames":
                return self._send_frame(parts[2], parts[4])
            if parts[0] == "static" and len(parts) == 2:
                return self._send_static_asset(parts[1])
            self._send_error(HTTPStatus.NOT_FOUND, "route not found")
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            parts = self._path_parts()
            if len(parts) == 4 and parts[:3] == ["api", "sessions", parts[2]] and parts[3] == "actions":
                count = self.workbench.extract_actions(parts[2])
                return self._send_json({"action_count": count})
            if len(parts) == 5 and parts[:3] == ["api", "sessions", parts[2]] and parts[3:] == ["intents", "init"]:
                payload = self._read_json()
                overwrite = bool(payload.get("overwrite", False)) if isinstance(payload, dict) else False
                count = self.workbench.init_intents(parts[2], overwrite=overwrite)
                return self._send_json({"segment_count": count})
            if len(parts) == 5 and parts[:3] == ["api", "sessions", parts[2]] and parts[3:] == ["chunks", "build"]:
                result = self.workbench.build_vla_chunks_v4(parts[2])
                return self._send_json(result)
            self._send_error(HTTPStatus.NOT_FOUND, "route not found")
        except FileExistsError as exc:
            self._send_error(HTTPStatus.CONFLICT, str(exc))
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_PUT(self) -> None:
        try:
            parts = self._path_parts()
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "intents":
                payload = self._read_json()
                if not isinstance(payload, list):
                    raise ValueError("intent payload must be an array")
                count = self.workbench.store.atomic_write_intents(parts[2], payload)
                return self._send_json({"segment_count": count})
            self._send_error(HTTPStatus.NOT_FOUND, "route not found")
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            return self._send_error(HTTPStatus.NOT_FOUND, "file not found")
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_static_asset(self, name: str) -> None:
        path = (self.static_dir / name).resolve()
        if path.parent != self.static_dir.resolve() or not path.is_file():
            return self._send_error(HTTPStatus.NOT_FOUND, "asset not found")
        self._send_file(path)

    def _send_frame(self, session_name: str, frame_name: str) -> None:
        session = self.workbench.store.resolve_session(session_name)
        if not frame_name.isdigit():
            raise ValueError("frame must be an integer")
        path = (session / "frames" / f"{int(frame_name):08d}.jpg").resolve()
        if path.parent != (session / "frames").resolve() or not path.is_file():
            raise FileNotFoundError("frame not found")
        self._send_file(path)


def create_server(root: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    store = SessionStore(root)
    workbench = DataWorkbench(store)
    static_dir = Path(__file__).resolve().parents[1] / "data_tools" / "static"
    handler = type(
        "ConfiguredWorkbenchHandler",
        (WorkbenchHandler,),
        {"workbench": workbench, "static_dir": static_dir},
    )
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动本地 VLA 数据处理工作台")
    parser.add_argument("--root", type=Path, default=Path("data/vla_raw_sessions"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = create_server(args.root, args.host, args.port)
    print(f"[workbench] http://{args.host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Browser edits backed by an atomic, versioned V6 review CSV."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from idv_agent.scripts.import_mvp_anylabeling import REVIEW_FIELDS, apply_review_row, ensure_session_review_csv, normalize_review_edit

EDITABLE = (
    "scope", "phase", "target_candidate", "target_track_id", "steering", "path",
    "action_source", "move", "camera_dx", "camera_dy", "reason", "reviewed",
    "decoding", "decoding_evidence_frames",
)
CHOICES = {
    "scope": ["", "in_scope", "out_of_scope", "uncertain"],
    "phase": ["", "search", "approach", "align", "interact", "maintain_decode"],
    "steering": ["", "target_center", "path_follow", "search_sweep", "hold"],
    "path": ["", "direct", "detour_left", "detour_right", "blocked", "unknown"],
    "action_source": ["", "accept_replay", "correction", "exclude"],
    "reviewed": ["pending", "reviewed"],
    "decoding": ["", "unknown", "true", "false"],
    "move": ["", *map(str, range(9))],
    "camera_dx": ["", "-110", "-25", "0", "25", "110"],
    "camera_dy": ["", "-110", "-25", "0", "25", "110"],
}


class ReviewConflict(ValueError):
    pass


class ReviewStore:
    def __init__(self):
        self.lock = RLock()

    def _read(self, workspace):
        workspace = Path(workspace)
        path = ensure_session_review_csv(workspace)
        content = path.read_bytes()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        rows = list(reader)
        fields = list(reader.fieldnames or ())
        for field in REVIEW_FIELDS:
            if field not in fields:
                fields.append(field)
        decisions = {row["id"]: row for row in map(json.loads,
                     (workspace / "decisions.jsonl").read_text(encoding="utf-8").splitlines())}
        if len(rows) != len(decisions) or {row.get("id") for row in rows} != set(decisions):
            raise ValueError("审核表端点与 decisions.jsonl 不一致")
        for row in rows:
            for field in fields:
                row.setdefault(field, "")
            row["reviewed"] = row["reviewed"] or "pending"
            source = decisions[row["id"]]
            if str(row["frame"]) != str(source["frame"]) or row["q_target"] != str(int(source["labels"]["facts"]["interact_prompt"])):
                raise ValueError("审核表只读帧号/Q 与当前工程不一致")
        return path, content, rows, fields, decisions

    def load(self, workspace):
        with self.lock:
            path, content, rows, fields, decisions = self._read(workspace)
            enriched = []
            for row in rows:
                source = decisions[row["id"]]
                detections = source.get("visual_import", {}).get("detections", [])
                cipher_candidates = [shape for shape in detections if shape.get("label") != "interact_prompt"]
                visual_issues = []
                if source["labels"]["facts"].get("interact_prompt") is not True:
                    candidate_mode = "not_prompt_frame"
                else:
                    candidate_mode = "prompt_no_navigation"
                enriched.append({**row, "candidates": source.get("visual_import", {}).get("detections", []),
                                 "history_frames": source["history_frames"], "replay": source["replay"],
                                 "issues": source["eligibility_issues"] + visual_issues,
                                 "suggested_target_candidate": (cipher_candidates[0]["candidate"]
                                                                if source["labels"]["facts"].get("interact_prompt") is not True
                                                                and len(cipher_candidates) == 1 else None),
                                 "candidate_mode": candidate_mode})
            return {"schema": "idv.workbench.v6.review", "workspace": str(workspace),
                    "path": str(path), "revision": hashlib.sha256(content).hexdigest(),
                    "rows": enriched, "choices": CHOICES, "editable": EDITABLE,
                    "progress": {"total": len(rows), "reviewed": sum(row["reviewed"] == "reviewed" for row in rows)}}

    def _validate_edit(self, edit, decision, reviewer):
        edit = normalize_review_edit(decision, edit)
        for field, choices in CHOICES.items():
            if edit.get(field, "") not in choices:
                raise ValueError(f"{field} 的值不在可选范围内")
        target = edit.get("target_candidate", "")
        candidates = {shape["candidate"] for shape in decision.get("visual_import", {}).get("detections", [])
                      if shape["label"] != "interact_prompt"}
        if target not in {"", "none", "unknown", "occluded", *candidates}:
            raise ValueError("目标必须来自当前帧密码机候选框")
        evidence = json.loads(edit.get("decoding_evidence_frames") or "[]")
        if not isinstance(evidence, list) or any(type(frame) is not int or not 0 <= frame <= decision["frame"] for frame in evidence):
            raise ValueError("破译证据只能引用当前或过去的帧")
        if edit["reviewed"] == "reviewed":
            apply_review_row(decision, edit, reviewer=reviewer)

    def save(self, workspace, *, row_id, patch, revision, reviewer):
        if not isinstance(patch, dict) or set(patch) - set(EDITABLE):
            raise ValueError("请求包含只读字段或未知字段")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("请填写标注人")
        with self.lock:
            path, before, rows, fields, decisions = self._read(workspace)
            current_revision = hashlib.sha256(before).hexdigest()
            if revision != current_revision:
                raise ReviewConflict("文件已有新修改，请重新加载后保存，避免覆盖其他标注")
            if row_id not in decisions:
                raise ValueError("未知端点 ID")
            edited = next(row for row in rows if row["id"] == row_id)
            for key, value in patch.items():
                if key == "decoding_evidence_frames" and isinstance(value, list):
                    value = json.dumps(value)
                if value is not None and not isinstance(value, (str, int)):
                    raise ValueError(f"{key} 必须是文本或整数")
                text = "" if value is None else str(value)
                if len(text) > 4000 or "\x00" in text:
                    raise ValueError(f"{key} 内容过长或无效")
                edited[key] = text
            normalized = normalize_review_edit(decisions[row_id], edited)
            edited.update(normalized)
            self._validate_edit(edited, decisions[row_id], reviewer)
            buf = io.StringIO(newline="")
            writer = csv.DictWriter(buf, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            after = buf.getvalue().encode("utf-8-sig")
            backups = path.parent / "review_history"
            backups.mkdir(exist_ok=True)
            archive = backups / f"{current_revision}.csv"
            if not archive.exists():
                archive.write_bytes(before)
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as handle:
                    handle.write(after); handle.flush(); os.fsync(handle.fileno())
                # Detect edits made externally while validating this request.
                if path.read_bytes() != before:
                    raise ReviewConflict("审核文件已变化，请重新加载")
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            with (backups / "audit.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"row_id": row_id, "reviewer": reviewer, "before": current_revision,
                                         "after": hashlib.sha256(after).hexdigest(), "patch": patch}, ensure_ascii=False)+"\n")
            return self.load(workspace)

    def check_complete(self, workspace, reviewer):
        with self.lock:
            _, _, rows, _, decisions = self._read(workspace)
            errors = []
            for row in rows:
                try:
                    if row["reviewed"] != "reviewed":
                        raise ValueError("端点尚未标注完成")
                    self._validate_edit(row, decisions[row["id"]], reviewer)
                except ValueError as exc:
                    errors.append({"id": row["id"], "frame": int(row["frame"]), "error": str(exc)})
            return {"pass": not errors, "errors": errors, "total": len(rows),
                    "reviewed": sum(row["reviewed"] == "reviewed" for row in rows)}

"""V6-only offline data workbench operations for the M29 pipeline."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from typing import Any
from idv_agent.scripts.import_mvp_anylabeling import apply_navigation, import_completed
from idv_agent.scripts.prepare_mvp_v6 import export, export_q, prepare, validate_workspace
from idv_agent.scripts.validate_vla_raw import validate as validate_raw
from .session_store import SessionStore
from .review_store import ReviewStore


class DataWorkbench:
    """Single V6 façade; no V5 action/chunk/intent operations."""
    def __init__(self, store: SessionStore):
        self.store = store
        self.reviews = ReviewStore()
        self.annotation_root = self.store.root.parent / "annotations" / "mvp_v6"

    def sessions(self) -> list[dict[str, Any]]:
        return self.store.list_sessions()

    def session_summary(self, name: str) -> dict[str, Any]:
        return self.store.summary(name)

    def annotation_readiness(self) -> dict[str, Any]:
        """Live cross-session label coverage shown while browser review proceeds."""
        phases, decoding = Counter(), Counter()
        steering, paths, action_sources = Counter(), Counter(), Counter()
        sessions, splits, groups = [], set(), set()
        total = reviewed = q_positive = q_negative = 0
        for summary in self.sessions():
            name = summary["name"]
            workspace = self._workspace(name, "xany_import_v2")
            if not workspace.is_dir():
                continue
            manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
            project = self.reviews.load(workspace)
            rows = project["rows"]
            completed = [row for row in rows if row["reviewed"] == "reviewed"]
            total += len(rows); reviewed += len(completed)
            q_positive += sum(row["q_target"] == "1" for row in rows)
            q_negative += sum(row["q_target"] == "0" for row in rows)
            phases.update(row["phase"] for row in completed if row["phase"])
            decoding.update(row["decoding"] or "unknown" for row in completed)
            steering.update(row["steering"] for row in completed if row["steering"])
            paths.update(row["path"] for row in completed if row["path"] not in {"", "unknown"})
            action_sources.update(row["action_source"] for row in completed if row["action_source"])
            splits.add(manifest.get("split")); groups.add(manifest.get("scenario_group"))
            sessions.append({"session": name, "reviewed": len(completed), "total": len(rows)})
        missing = [name for name in ("search", "approach", "align", "interact", "maintain_decode") if phases[name] == 0]
        blockers = []
        if reviewed != total: blockers.append(f"端点审核未完成：{reviewed}/{total}")
        if missing: blockers.append("缺少阶段：" + ", ".join(missing))
        if decoding["true"] == 0 or decoding["false"] == 0: blockers.append("decoding 必须同时有 true / false")
        if action_sources["accept_replay"] + action_sources["correction"] == 0: blockers.append("没有可训练的导航动作")
        if len([key for key, value in steering.items() if value]) < 2: blockers.append("steering 至少需要两类真实覆盖")
        if len([key for key, value in paths.items() if value]) < 2: blockers.append("path 至少需要两类真实覆盖")
        valid_splits = {value for value in splits if value}; valid_groups = {value for value in groups if value}
        if "val" not in valid_splits or len(valid_groups) < 2: blockers.append("缺少独立 scenario_group 的 val session")
        return {"schema": "idv.workbench.v6.readiness", "pass": total > 0 and not blockers,
                "reviewed": reviewed, "total": total, "pending": total-reviewed, "sessions": sessions,
                "q": {"positive_endpoints": q_positive, "negative_endpoints": q_negative},
                "phases": dict(phases), "missing_phases": missing, "decoding": dict(decoding),
                "steering": dict(steering), "paths": dict(paths), "action_sources": dict(action_sources),
                "splits": sorted(valid_splits), "scenario_groups": sorted(valid_groups), "blockers": blockers}

    def _workspace(self, name: str, suffix: str) -> Path:
        self.store.resolve_session(name)
        return self.annotation_root / f"{name}_{suffix}"

    def mvp_v6_summary(self, name: str) -> dict[str, Any]:
        session = self.store.resolve_session(name)
        imported = self._workspace(name, "xany_import_v2")
        if not imported.is_dir():
            imported = self._workspace(name, "xany_import_v1")
        prompt = self._workspace(name, "prompt_q")
        reviewed = self._workspace(name, "nav_review_v1")
        result: dict[str, Any] = {**self.store.summary(name), "schema": "idv.workbench.v6", "session": name,
            "raw_root": str(session), "raw_errors": validate_raw(session),
            "workspaces": {"prompt_q": {"path": str(prompt), "exists": prompt.is_dir()},
                           "xany_import": {"path": str(imported), "exists": imported.is_dir()},
                           "nav_review": {"path": str(reviewed), "exists": reviewed.is_dir()}}}
        if imported.is_dir():
            result["xany_import_validation"] = validate_workspace(imported, prompt_only=True)
            report_path = imported / "visual_import_report.json"
            if report_path.is_file():
                result["xany_import_report"] = json.loads(report_path.read_text(encoding="utf-8"))
        if reviewed.is_dir():
            result["nav_review_validation"] = validate_workspace(reviewed)
        projects = []
        if self.annotation_root.is_dir():
            for candidate in self.annotation_root.glob(f"{name}_*"):
                try:
                    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
                    if manifest.get("session_id") == name and (candidate / "decisions.jsonl").is_file() and (
                            (candidate / "navigation_review_with_state.csv").is_file() or (candidate / "navigation_review.csv").is_file()):
                        projects.append({"path": str(candidate.resolve()), "name": candidate.name,
                                         "split": manifest.get("split"), "scenario_group": manifest.get("scenario_group")})
                except (OSError, ValueError):
                    continue
        result["projects"] = sorted(projects, key=lambda item: item["name"], reverse=True)
        return result

    def prepare_mvp_v6(self, name: str, *, output: str | None = None) -> dict[str, Any]:
        target = Path(output) if output else self._workspace(name, "prompt_q")
        if target.exists():
            raise FileExistsError(str(target))
        return prepare(self.store.resolve_session(name), target)

    def import_mvp_v6(self, name: str, *, workspace: str | None = None, output: str | None = None, annotator: str,
                      completion_note: str, split: str, scenario_group: str) -> dict[str, Any]:
        source = self.resolve_workspace(name, workspace or str(self._workspace(name, "prompt_q")))
        return import_completed(source, Path(output) if output else self._workspace(name, "xany_import_v2"),
                                annotator=annotator, completion_note=completion_note, split=split, scenario_group=scenario_group)

    def validate_mvp_v6(self, name: str, *, workspace: str | None = None,
                        prompt_only: bool = False, require_reviewed: bool = False) -> dict[str, Any]:
        return validate_workspace(self.resolve_workspace(name, workspace),
                                  prompt_only=prompt_only, require_reviewed=require_reviewed)

    def review_mvp_v6(self, name: str, *, workspace: str | None = None, csv_path: str,
                      output: str | None = None, reviewer: str) -> dict[str, Any]:
        return apply_navigation(self.resolve_workspace(name, workspace),
                                Path(csv_path), Path(output) if output else self._workspace(name, "nav_review_v1"), reviewer=reviewer)

    def export_q_mvp_v6(self, name: str, *, workspace: str | None = None, output: str) -> dict[str, Any]:
        return export_q(self.resolve_workspace(name, workspace), Path(output))

    def export_mvp_v6(self, name: str, *, workspace: str | None = None, output: str) -> dict[str, Any]:
        return export(self.resolve_workspace(name, workspace or str(self._workspace(name, "nav_review_v1"))), Path(output))

    def resolve_workspace(self, name: str, workspace: str | None = None) -> Path:
        session = self.store.resolve_session(name)
        if workspace:
            target = Path(workspace)
        else:
            target = self._workspace(name, "xany_import_v2")
            if not target.is_dir():
                target = self._workspace(name, "xany_import_v1")
        target = target.resolve()
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "idv.mvp_workspace.v6" or manifest.get("session_id") != name or Path(manifest["raw_root"]).resolve() != session:
            raise ValueError("标注工程与所选 session 不一致")
        if target == session or session in target.parents:
            raise ValueError("标注工程不能写入 raw session")
        return target

    def review_rows(self, name: str, *, workspace: str | None = None) -> dict:
        return self.reviews.load(self.resolve_workspace(name, workspace))

    def update_review_row(self, name: str, *, workspace: str | None = None,
                          row_id: str, patch: dict, revision: str, reviewer: str) -> dict:
        return self.reviews.save(self.resolve_workspace(name, workspace), row_id=row_id,
                                 patch=patch, revision=revision, reviewer=reviewer)

    def check_review(self, name: str, *, workspace: str | None = None, reviewer: str) -> dict:
        target = self.resolve_workspace(name, workspace)
        validate_workspace(target, prompt_only=True)
        return self.reviews.check_complete(target, reviewer)

    def finish_review(self, name: str, *, workspace: str | None = None, reviewer: str) -> dict:
        from datetime import datetime
        target = self.resolve_workspace(name, workspace)
        with self.reviews.lock:
            check = self.check_review(name, workspace=str(target), reviewer=reviewer)
            if not check["pass"]:
                return {"exported": False, **check}
            csv_path = self.reviews.load(target)["path"]
            output = target.parent / f"{name}_nav_review_{datetime.now():%Y%m%d_%H%M%S_%f}"
            result = apply_navigation(target, Path(csv_path), output, reviewer=reviewer)
            return {"exported": True, "workspace": str(output),
                    "training_file": str(output / "training_v6.jsonl"), **result}

    def export_q_all(self, name: str, *, workspace: str | None = None) -> dict:
        from datetime import datetime
        target = self.resolve_workspace(name, workspace)
        validate_workspace(target, prompt_only=True)
        source = target / "prompt_q_all_frames.jsonl"
        if not source.is_file():
            raise ValueError("请打开 X-AnyLabeling 导入工程导出全部 Q 帧")
        content = source.read_bytes()
        records = [json.loads(line) for line in content.decode("utf-8").splitlines() if line]
        from idv_agent.training.m29_dataset import M29Dataset
        M29Dataset(source)  # validate schema, Q contract and image hashes
        directory = self.store.root.parent / "derived" / "prompt_q"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl"
        with output.open("xb") as handle:
            handle.write(content)
        return {"output": str(output), "frames": len(records),
                "q_positive": sum(row["targets"]["q"] for row in records),
                "q_negative": sum(not row["targets"]["q"] for row in records)}

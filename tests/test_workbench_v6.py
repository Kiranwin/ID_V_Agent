import json
from pathlib import Path

from idv_agent.data_tools.session_store import SessionStore
from idv_agent.data_tools.workbench import DataWorkbench
from test_mvp_v6_annotations import _raw


def test_workbench_v6_summary_reports_raw_and_annotation_stages(tmp_path):
    root = tmp_path / "raw_sessions"
    session = _raw(root)
    workbench = DataWorkbench(SessionStore(root))
    summary = workbench.mvp_v6_summary("ep")
    assert summary["raw_errors"] == []
    assert summary["workspaces"]["xany_import"]["exists"] is False
    assert summary["workspaces"]["nav_review"]["exists"] is False


def test_workbench_v6_import_delegates_and_is_fail_closed(tmp_path):
    root = tmp_path / "raw_sessions"
    session = _raw(root)
    annotations = tmp_path / "annotations" / "mvp_v6" / "ep_prompt_q"
    from idv_agent.scripts.prepare_mvp_v6 import prepare
    from PIL import Image
    for image in (session / "frames").glob("*.jpg"):
        Image.new("RGB", (1, 1)).save(image)
        image.with_suffix(".json").write_text(json.dumps({
            "imagePath": image.name, "imageWidth": 1, "imageHeight": 1, "shapes": []
        }))
    prepare(session, annotations)
    workbench = DataWorkbench(SessionStore(root))
    out = tmp_path / "imported"
    result = workbench.import_mvp_v6("ep", output=str(out), annotator="user",
                                    completion_note="done", split="train",
                                    scenario_group="scene")
    assert result["frames"] == 50
    assert (out / "prompt_q_endpoints.jsonl").is_file()
    assert workbench.mvp_v6_summary("ep")["workspaces"]["xany_import"]["exists"] is False

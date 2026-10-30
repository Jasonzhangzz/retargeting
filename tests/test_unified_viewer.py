from pathlib import Path
import json
from retarget.viewer.runtime import scan, filter_rows


def test_manifest_scanning_and_filters(tmp_path):
    for method in ("cem", "mppi"):
        folder = tmp_path / method
        folder.mkdir()
        cases = [
            dict(task="grasp", name="1_type", template="1_type", hand=h, method=method)
            for h in ("wuji1", "wuji2")
        ]
        (folder / "manifest.json").write_text(json.dumps(dict(cases=cases)))
    rows = scan(tmp_path)
    assert len(rows) == 4
    selected = filter_rows(rows, method="mppi", hand="wuji2", template="1_type")
    assert (
        len(selected) == 1
        and selected[0]["method"] == "mppi"
        and selected[0]["hand"] == "wuji2"
    )
    assert not filter_rows(rows, task="humanoid")

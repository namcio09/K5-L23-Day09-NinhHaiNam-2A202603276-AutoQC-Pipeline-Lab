"""Regression test for the optional extreme-box-aspect-ratio check."""

from PIL import Image

from pipeline.checks.label_format import run
from pipeline.common import Dataset, load_config


def test_extreme_box_aspect_ratio_is_reported(tmp_path):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (640, 480), "white").save(root / "images" / "sample.jpg")
    (root / "labels" / "sample.txt").write_text(
        "0 0.5 0.3 0.5 0.02\n1 0.5 0.7 0.02 0.8\n",
        encoding="utf-8",
    )
    ds = Dataset(root, ["car", "person", "truck", "bus"])

    issues = run(ds, load_config())

    aspect_issues = [issue for issue in issues if issue.code == "BOX_EXTREME_ASPECT"]
    assert [(issue.image, issue.line) for issue in aspect_issues] == [("sample", 1), ("sample", 2)]
    assert all(issue.severity == "warn" and issue.extra["aspect_ratio"] > 10 for issue in aspect_issues)

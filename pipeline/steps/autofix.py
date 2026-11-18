"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    fixes = []
    errors = [issue for issue in issues if issue.severity == ERROR]

    # Xóa ảnh trước để các lỗi nhãn khác của cùng ảnh không còn tác động.
    removed = set()
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    for issue in errors:
        if issue.code.startswith("DUP_IMAGE_") or issue.code in remove_codes:
            if issue.image in removed:
                continue
            image_path = ds.image_path(issue.image)
            label_path = ds.label_path(issue.image)
            if image_path is not None:
                image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            removed.add(issue.image)
            fixes.append({"image": issue.image, "line": "", "action": "remove_image",
                          "before": "ảnh và nhãn trong dataset", "after": "đã xóa"})

    # Ảnh thiếu nhãn phải được chuyển cho người gán; tuyệt đối không tạo file .txt rỗng.
    for issue in errors:
        if issue.code == "MISSING_LABEL_FILE" and issue.image not in removed:
            fixes.append({"image": issue.image, "line": "", "action": "needs_label",
                          "before": "(không có file)", "after": "chuyển người gán"})

    # Sửa tên lớp trong data.yaml một lần, kể cả khi có nhiều issue BAD_CLASS_NAME.
    class_issues = [issue for issue in errors if issue.code == "BAD_CLASS_NAME"]
    if class_issues:
        path = ds.yaml_path
        before_text = path.read_text(encoding="utf-8") if path.exists() else ""
        data = yaml.safe_load(before_text) if before_text.strip() else {}
        if not isinstance(data, dict):
            data = {}
        old_names = data.get("names")
        data["names"] = {i: name for i, name in enumerate(cfg["classes"])}
        after_text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        if before_text != after_text:
            path.write_text(after_text, encoding="utf-8")
            fixes.append({"image": "data.yaml", "line": "", "action": "rename_classes",
                          "before": str(old_names), "after": str(cfg["classes"])})

    # Gom lỗi theo ảnh và số dòng gốc, để việc xóa nhiều dòng không làm lệch số dòng.
    by_image = defaultdict(lambda: defaultdict(list))
    for issue in errors:
        if (issue.image not in removed and issue.code in {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX", "BOX_OUT_OF_IMAGE"}
                and issue.line is not None):
            by_image[issue.image][issue.line].append(issue)

    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    for stem, line_issues in by_image.items():
        label_path = ds.label_path(stem)
        if not label_path.exists():
            continue
        original_text = label_path.read_text(encoding="utf-8", errors="replace")
        original_lines = original_text.splitlines()
        new_lines = list(original_lines)
        changed = False

        for line_no, line_errors in sorted(line_issues.items()):
            index = line_no - 1
            if index < 0 or index >= len(original_lines):
                continue
            before = original_lines[index]
            codes = {issue.code for issue in line_errors}

            if codes & drop_codes:
                new_lines[index] = None
                fixes.append({"image": stem, "line": line_no, "action": "drop_line",
                              "before": before, "after": "(đã xóa dòng lỗi)"})
                changed = True
                continue

            if "BOX_OUT_OF_IMAGE" in codes:
                try:
                    class_id, cx, cy, box_w, box_h = parse_yolo_line(before)
                except ValueError:
                    new_lines[index] = None
                    fixes.append({"image": stem, "line": line_no, "action": "drop_line",
                                  "before": before, "after": "(dòng không phân tích được)"})
                    changed = True
                    continue

                left = max(0.0, cx - box_w / 2)
                right = min(1.0, cx + box_w / 2)
                top = max(0.0, cy - box_h / 2)
                bottom = min(1.0, cy + box_h / 2)
                clipped_w, clipped_h = right - left, bottom - top
                if clipped_w <= 0 or clipped_h <= 0:
                    new_lines[index] = None
                    fixes.append({"image": stem, "line": line_no, "action": "drop_line",
                                  "before": before, "after": "(box nằm ngoài ảnh)"})
                else:
                    clipped_cx, clipped_cy = (left + right) / 2, (top + bottom) / 2
                    after = f"{class_id} {clipped_cx:.6g} {clipped_cy:.6g} {clipped_w:.6g} {clipped_h:.6g}"
                    new_lines[index] = after
                    fixes.append({"image": stem, "line": line_no, "action": "clip_box",
                                  "before": before, "after": after})
                changed = True

        if changed:
            kept_lines = [line for line in new_lines if line is not None]
            ending = "\n" if original_text.endswith(("\n", "\r")) else ""
            label_path.write_text("\n".join(kept_lines) + ending, encoding="utf-8")

    return fixes

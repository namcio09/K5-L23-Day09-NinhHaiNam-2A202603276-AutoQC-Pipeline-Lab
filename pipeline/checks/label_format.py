"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    """Kiểm tra cú pháp và hình học của các box trong dataset."""
    issues = []
    rules = cfg["label_format"]
    overflow_tol = rules["overflow_tol"]
    dup_box_iou = rules["dup_box_iou"]
    min_box_px = rules["min_box_px"]
    max_box_frac = rules["max_box_frac"]

    for image_path in ds.image_files():
        stem = image_path.stem
        lines = ds.read_label_lines(stem)
        if lines is None:
            continue
        size = ds.image_size(stem)
        width, height = size if size is not None else (None, None)
        previous_boxes = []

        for line_no, text in lines:
            try:
                class_id, cx, cy, box_w, box_h = parse_yolo_line(text)
            except ValueError as exc:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem,
                                    f"Dòng nhãn không đúng định dạng YOLO: {exc}", line_no))
                continue

            if not 0 <= class_id < len(ds.classes):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem,
                                    f"ID lớp {class_id} nằm ngoài phạm vi hợp lệ.", line_no))
                continue
            if box_w <= 0 or box_h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem,
                                    "Chiều rộng và chiều cao của box phải lớn hơn 0.", line_no))
                continue

            left, right = cx - box_w / 2, cx + box_w / 2
            top, bottom = cy - box_h / 2, cy + box_h / 2
            overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
            if overflow > overflow_tol:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                                    f"Box vượt mép ảnh {overflow:.6g} (tọa độ chuẩn hóa).", line_no,
                                    {"overflow": overflow}))

            box = (cx, cy, box_w, box_h)
            if any(class_id == old_class and iou_xywh(box, old_box) > dup_box_iou
                   for old_class, old_box in previous_boxes):
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem,
                                    "Box gần như trùng với box cùng lớp ở dòng trước.", line_no))
            previous_boxes.append((class_id, box))

            if width is not None and height is not None:
                short_side_px = min(box_w * width, box_h * height)
                if short_side_px < min_box_px:
                    issues.append(Issue("label_format", WARN, "BOX_TINY", stem,
                                        f"Cạnh ngắn của box là {short_side_px:.2f}px, nhỏ hơn {min_box_px}px.", line_no))

            if box_w * box_h > max_box_frac:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem,
                                    f"Diện tích box chiếm {box_w * box_h:.4g} ảnh, vượt {max_box_frac}.", line_no))

            # Bonus: cảnh báo box có tỷ lệ khung hình cực đoan (rộng/cao hoặc cao/rộng > 10).
            if width is not None and height is not None:
                pixel_w, pixel_h = box_w * width, box_h * height
                if pixel_w > 0 and pixel_h > 0:
                    aspect_ratio = max(pixel_w / pixel_h, pixel_h / pixel_w)
                    if aspect_ratio > 10:
                        issues.append(Issue("label_format", WARN, "BOX_EXTREME_ASPECT", stem,
                                            f"Tỷ lệ khung hình box bất thường ({aspect_ratio:.2f}:1, ngưỡng 10:1).",
                                            line_no, {"aspect_ratio": round(aspect_ratio, 4)}))

    return issues

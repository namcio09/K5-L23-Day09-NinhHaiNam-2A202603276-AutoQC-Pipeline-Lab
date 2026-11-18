"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    rules = cfg["second_opinion"]
    conf_threshold = rules["conf"]
    min_area_px = rules["min_area_px"]
    fp_conf = rules["fp_conf"]
    iou_missing = rules["iou_missing"]
    iou_match = rules["iou_match"]
    iou_loose = rules["iou_loose"]
    issues = []

    for image_path in ds.image_files():
        stem = image_path.stem
        if stem not in preds:
            continue
        label_lines = ds.read_label_lines(stem)
        if label_lines is None:
            continue

        # Giữ lại số dòng gốc để cảnh báo box nhãn thừa trỏ đúng vị trí trong file.
        labels = []
        for line_no, text in label_lines:
            try:
                class_id, cx, cy, box_w, box_h = parse_yolo_line(text)
            except ValueError:
                continue
            if not 0 <= class_id < len(ds.classes):
                continue
            labels.append((line_no, class_id, (cx, cy, box_w, box_h)))

        # Các dự đoán dùng để so thiếu/sai/lỏng phải đạt cả ngưỡng tin cậy và kích thước.
        size = ds.image_size(stem)
        main_preds = []
        all_fp_preds = []
        for row in preds[stem]:
            try:
                class_id, confidence, cx, cy, box_w, box_h = row
                class_id = int(class_id)
                confidence = float(confidence)
                box = (float(cx), float(cy), float(box_w), float(box_h))
            except (TypeError, ValueError):
                continue
            if box[2] <= 0 or box[3] <= 0:
                continue
            if confidence >= fp_conf:
                all_fp_preds.append((class_id, confidence, box))
            area_px = box[2] * box[3] * size[0] * size[1] if size else 0
            if confidence >= conf_threshold and size is not None and area_px >= min_area_px:
                main_preds.append((class_id, confidence, box))

        # Ý kiến model theo đúng thứ tự dự đoán trong cache.
        for pred_class, confidence, pred_box in main_preds:
            candidates = [
                (label_class, iou_xywh(pred_box, label_box))
                for _, label_class, label_box in labels
                if iou_xywh(pred_box, label_box) >= iou_missing
            ]
            if not candidates:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_MISSING", stem,
                                    "Model thấy vật thể nhưng không tìm thấy box nhãn khớp.",
                                    extra={"conf": confidence}))
                continue

            same_class = [iou for label_class, iou in candidates if label_class == pred_class]
            if same_class:
                best_iou = max(same_class)
                if best_iou < iou_loose:
                    issues.append(Issue("second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem,
                                        "Box nhãn cùng lớp nhưng lệch hoặc rộng hơn dự đoán model.",
                                        extra={"conf": confidence, "iou": best_iou}))
                continue

            best_other_iou = max(iou for _, iou in candidates)
            if best_other_iou >= iou_match:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem,
                                    "Model và nhãn có vẻ chỉ cùng vật thể nhưng khác lớp.",
                                    extra={"conf": confidence, "iou": best_other_iou}))

        # Box nhãn không khớp bất kỳ dự đoán đủ tin cậy nào là ứng viên false positive.
        for line_no, _, label_box in labels:
            best_iou = max((iou_xywh(label_box, pred_box) for _, _, pred_box in all_fp_preds), default=0.0)
            if best_iou < iou_missing:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                                    "Box nhãn không khớp với dự đoán nào của model.", line_no,
                                    {"iou": best_iou}))

    return issues

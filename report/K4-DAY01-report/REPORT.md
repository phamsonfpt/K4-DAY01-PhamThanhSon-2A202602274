# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh - prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.
- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  - `class_id`: 468
  - `class_name`: `cab`
  - `rank`: 1
  - `score`: 0.510915
  - `taxonomy_name`: `ImageNet-1K`

- Record này mô tả toàn ảnh như thế nào?
  Record hạng 1 cho thấy mô hình classification dự đoán `cab` (taxi) là lớp phù hợp nhất với toàn bộ ảnh `traffic`. Classification hoạt động ở cấp độ ảnh, vì vậy nhãn này mô tả nội dung tổng thể của ảnh chứ không chỉ ra vị trí cụ thể của chiếc taxi.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Class list được xác định trước bởi taxonomy của dataset/dự án. Trong bài lab này, mô hình sử dụng taxonomy `ImageNet-1K`. Checkpoint chỉ dự đoán trong danh sách các lớp đã được định nghĩa, không tự tạo thêm class mới.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  Cần giữ `class_id`, `class_name` và `taxonomy_name` để xác định chính xác lớp, tránh nhầm lẫn giữa các lớp có tên tương tự và đảm bảo tính nhất quán khi annotation, training và kiểm tra dữ liệu.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Guideline cần quy định rõ cách xác định chủ thể chính và cách chọn nhãn khi ảnh có nhiều chủ thể. Đồng thời cần quy định cách xử lý trường hợp có nhiều lớp, chủ thể không rõ ràng hoặc ảnh không phù hợp với bất kỳ lớp nào.

-Vì sao model score không phải ground truth?
  `score` chỉ thể hiện mức độ tin tưởng của mô hình đối với dự đoán `cab`. Giá trị `0.510915` không có nghĩa là ground truth đúng 51.0915%. Ground truth phải do con người tạo hoặc xác nhận dựa trên taxonomy và guideline của dự án.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):class_name cho biết loại vật thể, score là độ tin cậy của model, bbox_xyxy xác định vị trí vật thể bằng tọa độ pixel, bbox_width/height là kích thước box.
- Diễn giải vị trí box bằng lời:[x_min, y_min, x_max, y_max], trong đó (x_min,y_min) là góc trên trái và (x_max,y_max) là góc dưới phải.
- So sánh số prediction ở hai threshold:Threshold thấp → nhiều prediction hơn, tăng độ bao phủ nhưng reviewer phải kiểm tra nhiều hơn. Threshold cao → ít prediction hơn nhưng có thể bỏ sót vật thể.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
- Đề xuất một quy tắc box chặt:Box phải bao phủ đầy đủ vật thể và sát ranh giới vật thể, hạn chế background.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?Guideline cần quy định cách vẽ box; nếu không xác định rõ ranh giới thì cần đưa vào review/escalation.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):instance_id xác định từng object, class_name là tên lớp, score là độ tin cậy, polygon gồm các điểm tạo thành đường bao object.
- Polygon bổ sung chi tiết gì so với box?:Polygon mô tả biên dạng chính xác của object, chi tiết hơn bounding box.
- `instance_id` dùng để làm gì và không phải loại ID nào?: Dùng để phân biệt từng object riêng biệt, không phải class ID.
- Đề xuất một quy tắc biên mask:Mask nên bám sát đường biên thực tế của object, không ăn quá nhiều background.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?: Guideline cần quy định cách vẽ biên; nếu không xác định rõ thì đưa vào review/escalation thay vì tự đoán.

## 4. Vòng đời và kiểm tra chất lượng

`Ảnh thô → Guideline → Ground Truth → Huấn luyện → Prediction → QC/Rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|
| Phân loại ảnh | Gán class cho toàn ảnh | Nhiều chủ thể, khó chọn class | Kiểm tra nhãn |
| Phát hiện vật thể | Class + bounding box | Box lệch, che khuất/cắt mép | Kiểm tra box |
| Instance segmentation | Class + polygon/mask | Biên mờ, che khuất, tiếp xúc | Kiểm tra mask |
Mục đích: Đảm bảo dữ liệu được gán nhãn đúng và nhất quán. Các trường hợp không rõ ràng cần được review/rework thay vì tự đoán.

## 5. An toàn dữ liệu

- Chỉ sử dụng dữ liệu được dự án cho phép và không chia sẻ ra ngoài phạm vi.
- Nếu phát hiện ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý và báo cho người phụ trách dự án.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.





# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** [CẦN ĐIỀN: CPU hoặc GPU]

**Python / PyTorch / Ultralytics:** Python [CẦN ĐIỀN] / PyTorch [CẦN ĐIỀN] / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không thay đổi (cần xác nhận nếu đã chỉnh sửa code).

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K.

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Trả lời: Record này phân loại cả bức ảnh vì tác vụ Image Classification dự đoán nhãn cho toàn bộ ảnh, không xác định vị trí từng vật thể. Trong ảnh traffic có nhiều phương tiện nhưng model dự đoán lớp cab có điểm cao nhất.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Trả lời: Danh sách tên lớp được xác định sẵn trong taxonomy ImageNet-1K đi kèm checkpoint đã huấn luyện. Model không tự tạo ra danh sách lớp mới.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   Trả lời: class_id là mã định danh của lớp, class_name giúp con người hiểu tên lớp, taxonomy_name xác định bộ nhãn được sử dụng. Ghi đủ ba thông tin giúp tránh nhầm lẫn giữa các bộ dữ liệu.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Trả lời: Guideline cần quy định cách xác định chủ thể chính của ảnh, tiêu chí ưu tiên khi có nhiều vật thể và cách xử lý nếu không xác định được một lớp phù hợp.

6. Vì sao không được lấy `score` của model làm ground truth?

   Trả lời: Score chỉ thể hiện mức điểm dự đoán của model, không chứng minh dự đoán chắc chắn đúng. Ground truth phải được xác định theo quy tắc gán nhãn và kiểm tra chất lượng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=[CẦN ĐIỀN TỪ ẢNH KITCHEN] ; `score`=[CẦN ĐIỀN] ; `bbox_xyxy`=[CẦN ĐIỀN] ; `bbox_width`=[CẦN ĐIỀN] ; `bbox_height`=[CẦN ĐIỀN].

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Trả lời: Bounding box là hình chữ nhật bao quanh vật thể. Tọa độ được viết theo dạng [x_min, y_min, x_max, y_max], trong đó (x_min, y_min) là góc trên bên trái, (x_max, y_max) là góc dưới bên phải.

   Chiều rộng = x_max - x_min.
   
   Chiều cao = y_max - y_min.

   Với record đã chọn: [CẦN ĐIỀN TỌA ĐỘ VÀ KÍCH THƯỚC THỰC TẾ] pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Trả lời: Theo kết quả chạy Colab của tôi:
   
   - Ngưỡng 0.20: 17 prediction.
   - Ngưỡng 0.60: 6 prediction.
   
   Ngoài ra, ngưỡng 0.35 có 11 prediction.

   Nhận xét: Khi tăng ngưỡng confidence, số lượng prediction được giữ lại giảm. Khi giảm ngưỡng, model giữ thêm các dự đoán có điểm thấp hơn.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Trả lời: Reviewer phải kiểm tra thêm các hộp dự đoán có độ tin cậy thấp, phát hiện hộp sai, hộp trùng, nhãn sai và kiểm tra vị trí của bounding box.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Trả lời: Bounding box phải bao sát vật thể theo quy tắc thống nhất, hạn chế chứa nền không liên quan và phải xác định đúng lớp của vật thể.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Trả lời: Trong ảnh kitchen, người ở phía sau quầy/bàn bếp bị che một phần cơ thể bởi các vật dụng phía trước. Tôi đề xuất guideline quy định rõ cách gán nhãn vật thể bị che khuất. Nếu chưa có quy tắc xác định phạm vi bounding box thì cần hỏi Lab Coach để tránh mỗi người gán nhãn một cách.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=[CẦN ĐIỀN TỪ ẢNH KITCHEN] ; `class_name`=[CẦN ĐIỀN] ; `score`=[CẦN ĐIỀN] ; số điểm polygon=`polygon_point_count`=[CẦN ĐIỀN] ; 3–5 điểm đầu của `polygon_xy`=[CẦN ĐIỀN].

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Trả lời: Polygon thể hiện đường viền chi tiết của vật thể, giúp xác định chính xác hơn phần diện tích thuộc vật thể. Bounding box chỉ thể hiện hình chữ nhật bao quanh vật thể.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   Trả lời: instance_id dùng để phân biệt từng đối tượng riêng biệt trong kết quả phân đoạn. Nhiều vật thể có thể cùng class_id nhưng mỗi vật thể có instance_id khác nhau. instance_id không dùng để theo dõi vật thể qua nhiều khung hình như tracking ID.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Trả lời: Mask phải bám sát đường viền phần vật thể nhìn thấy, không tô lấn sang nền hoặc vật thể khác. Các vật thể riêng biệt cần được phân đoạn thành các instance riêng.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Trả lời: Trong ảnh kitchen, các bát và đồ vật trên bàn nằm sát hoặc che khuất nhau, khiến việc xác định ranh giới trở nên khó khăn. Tôi sử dụng guideline để quy định cách phân biệt đường biên và chỉ vẽ phần vật thể nhìn thấy. Nếu đường biên không rõ đến mức không thể xác định chính xác thì cần hỏi Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn bộ ảnh | Ảnh traffic có nhiều loại phương tiện, model dự đoán cab với score 0.510915 nhưng chưa chắc là nhãn chuẩn | Chọn nhãn phù hợp dựa trên guideline | Kiểm tra nhãn có đúng chủ đề và taxonomy không |
| Phát hiện vật thể | Nhãn lớp và bounding box của từng vật thể | Ảnh kitchen có nhiều vật thể gần nhau, một số bị che khuất hoặc hộp chồng lấn | Vẽ bounding box và gán class cho từng vật thể | Kiểm tra hộp sai, thiếu, trùng và nhãn không chính xác |
| Instance segmentation | Nhãn lớp và mask/polygon riêng cho từng vật thể | Ảnh kitchen có các đồ vật chạm nhau và đường biên khó phân biệt | Vẽ mask theo biên vật thể và tách từng instance | Kiểm tra đường viền, vùng tô sai và các instance bị gộp |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Trả lời: Chỉ sử dụng dữ liệu được phép trong bài lab, không đưa thông tin cá nhân hoặc dữ liệu nhạy cảm lên Colab và GitHub công khai.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Trả lời: Tôi sẽ dừng xử lý dữ liệu đó và báo cho giảng viên hoặc Lab Coach để được hướng dẫn, không tự ý sử dụng hay chia sẻ.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS

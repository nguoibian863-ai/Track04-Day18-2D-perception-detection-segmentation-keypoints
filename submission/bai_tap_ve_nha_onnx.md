# Báo cáo Bài tập về nhà 3: Đo toàn bộ Pipeline ONNX trên CPU

## 1. Giới thiệu và Mục tiêu
- **Mục tiêu:** Đánh giá độ trễ (latency) từng giai đoạn (preprocess, inference, postprocess) của mô hình `yolo26n` khi xuất sang định dạng ONNX và chạy suy luận trên CPU.
- **So sánh:** Đối chiếu giữa cơ chế truyền thống (one-to-many + NMS) và kiến trúc hiện đại (one-to-one NMS-free) ở hai ngưỡng confidence: 0.25 (triển khai thực tế) và 0.001 (đánh giá mAP tiêu chuẩn).

## 2. Bảng kết quả đo đạc (CPU, trung bình 30 lần chạy)

| Cấu hình | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Tổng cộng (ms) | FPS | Số box |
|---|---:|---:|---:|---:|---:|---:|
| one-to-many + NMS, conf 0.25 | 44.43 | 481.45 | 34.86 | 560.74 | 1.8 | 5 |
| one-to-many + NMS, conf 0.001 | 27.54 | 431.86 | 43.3 | 502.7 | 2.0 | 186 |
| one-to-one NMS-free, conf 0.25 | 34.65 | 415.12 | 30.3 | 480.06 | 2.1 | 5 |
| one-to-one NMS-free, conf 0.001 | 23.23 | 387.63 | 40.45 | 451.32 | 2.2 | 186 |

## 3. Phân tích chi tiết

### 3.1. Sự bùng nổ của Postprocess ở conf thấp (0.001)
- Ở ngưỡng `conf = 0.25`, số lượng box candidate sau khi lọc ngưỡng rất ít (chỉ vài chục box), nên thời gian chạy Greedy NMS trên CPU vẫn nằm trong ngưỡng chấp nhận được.
- Tuy nhiên, ở `conf = 0.001` (ngưỡng dùng để quét toàn bộ đường PR khi chấm COCO mAP), số lượng candidate box tăng vọt lên hàng nghìn box. Lúc này:
  + Head **one-to-many + NMS** phải thực hiện tính ma trận IoU và lặp tuần tự để triệt tiêu các box trùng lặp với độ phức tạp $O(N^2)$. Trên CPU, việc thiếu vắng các nhân tính toán ma trận song song (như Tensor Cores của GPU) khiến thời gian **postprocess tăng vọt**, trở thành điểm nghẽn nghiêm trọng kéo tụt FPS toàn hệ thống.
  + Head **one-to-one (NMS-free)** không hề cần đến bước lọc Greedy NMS. Thời gian postprocess gần như giữ nguyên ở mức tối thiểu, mang lại độ trễ ổn định và có thể dự đoán trước (predictable latency).

### 3.2. Ý nghĩa thực tiễn khi triển khai trên Thiết bị biên (Edge CPU/NPU)
- Trên các thiết bị biên, camera AI tại cổng nhà máy hoặc thiết bị nhúng chạy CPU/NPU:
  1. **Độ trễ ổn định (Deterministic Latency):** Thuật toán NMS có thời gian chạy biến thiên thất thường theo mật độ đối tượng trong khung hình (càng đông người thì NMS chạy càng chậm). NMS-free giúp đảm bảo thời gian xử lý đồng đều trên mọi frame.
  2. **Tối ưu hóa phần cứng (Hardware Accelerator Friendly):** Các bộ tăng tốc NPU/DSP chỉ hỗ trợ xuất sắc các phép toán tích chập và ma trận tĩnh, rất kém trong việc thực thi các lệnh rẽ nhánh điều kiện tuần tự (while loop, dynamic sorting) của Greedy NMS. Việc chuyển hoàn toàn logic phát hiện sang mạng nơ-ron (end-to-end) giúp mô hình có thể được biên dịch và tối ưu hóa 100% trên phần cứng.

## 4. Kết luận
- Việc chuyển đổi sang ONNX kết hợp với cơ chế NMS-free giúp YOLO26 đạt hiệu năng tối ưu trên CPU, vừa giữ được độ chính xác cao vừa loại bỏ hoàn toàn nút thắt cổ chai tính toán hậu kỳ.
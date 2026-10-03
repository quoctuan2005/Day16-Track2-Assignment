# Báo cáo Thực hành LAB 16: Cloud AI Environment Setup (GCP)

## 1. Thông tin cấu hình thử nghiệm
- **Nền tảng:** Google Cloud Platform (GCP)
- **Tài nguyên tính toán:** Compute Engine VM (`e2-medium`: 2 vCPU, 4GB RAM)
- **Hệ điều hành:** Debian 12 (Linux)
- **Mô hình & Thuật toán:** LightGBM (Gradient Boosting Decision Tree)
- **Bộ dữ liệu:** Kaggle Credit Card Fraud Detection (284,807 dòng, 31 đặc trưng, dung lượng ~150MB)

---

## 2. Bảng kết quả Benchmark (`benchmark_result.json`)

| Metric | Kết quả | Đơn vị / Ý nghĩa |
|---|---|---|
| **Thời gian load data** | **3.3436** | giây (đọc và phân tích file CSV 150MB) |
| **Thời gian training** | **2.1462** | giây (huấn luyện trên 80% dữ liệu) |
| **Best iteration** | **1** | vòng lặp tối ưu |
| **AUC-ROC** | **0.9367** | Khả năng phân biệt gian lận rất cao |
| **Accuracy** | **0.9990** | Độ chính xác tổng thể (99.9%) |
| **F1-Score** | **0.7421** | Trung hòa giữa Precision và Recall |
| **Precision** | **0.6667** | Tỷ lệ dự đoán đúng trong các ca cảnh báo gian lận |
| **Recall** | **0.8367** | Tỷ lệ phát hiện được các vụ gian lận thực tế |
| **Inference Latency (1 dòng)** | **0.751** | ms / sample (< 1 phần nghìn giây) |
| **Inference Throughput (1000 dòng)** | **795,581.18** | dòng / giây |

---

## 3. Báo cáo nhận xét (5 - 10 dòng)

1. **Hiệu năng nạp dữ liệu và huấn luyện:** Quá trình huấn luyện LightGBM trên máy ảo CPU `e2-medium` (2 vCPU, 4GB RAM) diễn ra cực kỳ nhanh chóng: thời gian đọc và tiền xử lý toàn bộ tập dữ liệu 284,807 dòng chỉ mất **3.34 giây**, và thời gian hoàn tất huấn luyện mô hình chỉ mất **2.15 giây**.
2. **Độ chính xác và phát hiện gian lận:** Mặc dù tập dữ liệu mất cân bằng nghiêm trọng (tỷ lệ gian lận rất nhỏ ~0.17%), mô hình đạt được chỉ số **AUC-ROC = 0.9367**, **Accuracy = 99.9%** và **Recall = 83.67%**, chứng minh thuật toán phân loại và bắt trúng hầu hết các giao dịch gian lận mà không cần cân chỉnh trọng số phức tạp.
3. **Tốc độ suy luận (Inference Speed):** Tốc độ phản hồi đạt mức siêu nhanh phục vụ thời gian thực với **độ trễ (latency) chỉ 0.751 ms/dòng** (dưới 1 mili-giây) và **throughput đạt ~795,580 dòng/giây**, hoàn toàn đáp ứng các hệ thống thanh toán trực tuyến yêu cầu độ trễ cực thấp.
4. **Hiệu quả tài nguyên và chi phí:** Kết quả thử nghiệm khẳng định rằng với các bài toán dữ liệu dạng bảng (Tabular Data), việc sử dụng máy ảo CPU cấu hình nhỏ (`e2-medium`) kết hợp thuật toán tối ưu như LightGBM mang lại hiệu quả vượt trội, tiết kiệm tối đa ngân sách vận hành trên Cloud (~$0.09/giờ) mà không bắt buộc phải đầu tư chi phí lớn cho phần cứng GPU.

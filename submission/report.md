# Báo cáo Thực hành LAB 16: Cloud AI Environment Setup (GCP)

1. Tôi dùng Google Cloud Platform (GCP), region us-central1 (zone us-central1-a), instance e2-medium (2 vCPU, 4GB RAM), source commit b89c83b.
2. Dataset có 284,807 dòng (Credit Card Fraud Detection), chia train/test theo tỷ lệ 80/20 với stratify=y, seed 42.
3. Load dữ liệu mất 3.3436 giây; training mất 2.1462 giây; best iteration là 1.
4. AUC 0.9367, Accuracy 0.9990, F1 0.7421, Precision 0.6667, Recall 0.8367 trên tập test.
5. Latency 1 dòng 0.751 ms; throughput batch 1.000 dòng 795,581.18 dòng/giây; đo lường trung bình qua vòng lặp inference với LightGBM.
6. CPU/RAM/Network tôi quan sát lúc chạy benchmark: CPU ~100% trong lúc nạp và train, RAM dùng ~1.4GB / 3.8GB, Network nhận data ổn định; ảnh đính kèm trong thư mục screenshots/.
7. Billing tại thời điểm kiểm tra ghi nhận $0.00 / chưa cập nhật (do độ trễ batch 2-24h của GCP); ước tính chi phí thực tế ~$0.09/giờ cho e2-medium + Cloud NAT + Load Balancer.
8. Tôi đã tải kết quả và xóa tài nguyên lúc hoàn tất benchmark; bằng chứng dọn dẹp xác nhận qua `terraform destroy complete` và `gcloud compute instances list` (trả về 0 items).

import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb

print("=== BẮT ĐẦU BENCHMARK LIGHTGBM TRÊN GCP CPU ===")

# 1. Đo thời gian nạp dữ liệu
t0 = time.time()
df = pd.read_csv("creditcard.csv")
load_time = time.time() - t0
print(f"1. Thời gian load data: {load_time:.3f} giây (Shape: {df.shape})")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Huấn luyện mô hình
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbosity": -1,
    "n_jobs": -1
}

t_train_start = time.time()
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)
train_time = time.time() - t_train_start
best_iteration = model.best_iteration
print(f"2. Thời gian training: {train_time:.3f} giây (Best iteration: {best_iteration})")

# 3. Đánh giá chất lượng mô hình trên Test Set
y_pred_proba = model.predict(X_test, num_iteration=best_iteration)
y_pred_binary = (y_pred_proba >= 0.5).astype(int)

auc = float(roc_auc_score(y_test, y_pred_proba))
acc = float(accuracy_score(y_test, y_pred_binary))
f1 = float(f1_score(y_test, y_pred_binary))
precision = float(precision_score(y_test, y_pred_binary))
recall = float(recall_score(y_test, y_pred_binary))

# 4. Đo Inference Latency (1 dòng)
single_sample = X_test.iloc[[0]]
latency_runs = 500
t_lat_start = time.time()
for _ in range(latency_runs):
    _ = model.predict(single_sample, num_iteration=best_iteration)
single_latency_ms = ((time.time() - t_lat_start) / latency_runs) * 1000

# 5. Đo Inference Throughput (1000 dòng)
batch_samples = X_test.iloc[:1000]
t_batch_start = time.time()
_ = model.predict(batch_samples, num_iteration=best_iteration)
batch_time = time.time() - t_batch_start
throughput_qps = 1000.0 / batch_time

results = {
    "data_load_time_sec": round(load_time, 4),
    "training_time_sec": round(train_time, 4),
    "best_iteration": int(best_iteration),
    "auc_roc": round(auc, 4),
    "accuracy": round(acc, 4),
    "f1_score": round(f1, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "inference_latency_single_row_ms": round(single_latency_ms, 3),
    "inference_throughput_1000_rows_qps": round(throughput_qps, 2)
}

with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n" + "="*50)
print(f"{'Metric':<35} | {'Kết quả':<15}")
print("="*50)
print(f"{'Thời gian load data (s)':<35} | {results['data_load_time_sec']}")
print(f"{'Thời gian training (s)':<35} | {results['training_time_sec']}")
print(f"{'Best iteration':<35} | {results['best_iteration']}")
print(f"{'AUC-ROC':<35} | {results['auc_roc']}")
print(f"{'Accuracy':<35} | {results['accuracy']}")
print(f"{'F1-Score':<35} | {results['f1_score']}")
print(f"{'Precision':<35} | {results['precision']}")
print(f"{'Recall':<35} | {results['recall']}")
print(f"{'Inference latency (1 row)':<35} | {results['inference_latency_single_row_ms']} ms")
print(f"{'Inference throughput (1000 rows)':<35} | {results['inference_throughput_1000_rows_qps']} rows/s")
print("="*50)
print("Đã lưu kết quả vào benchmark_result.json")

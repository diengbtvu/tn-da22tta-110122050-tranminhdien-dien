# Mô hình dự đoán tốc độ tạo hydro trong hệ thống kỵ khí

**Khóa luận tốt nghiệp – Trường Đại học Trà Vinh**

| | |
|---|---|
| **Sinh viên** | Trần Minh Điền |
| **MSSV** | 110122050 |
| **Lớp** | DA22TTA |
| **Khóa** | 2022 |
| **Giảng viên hướng dẫn** | TS. Nguyễn Bảo Ân |

---

## 1. Mục tiêu

Xây dựng mô hình học máy dự đoán **tốc độ tạo hydro (Hydrogen Production Rate – HPR)**
của hệ thống lên men kỵ khí dựa trên các thông số vận hành của **bể phản ứng sinh học**
(pH, VSS, nồng độ các axit béo dễ bay hơi, ORP, COD, hiệu suất phân hủy sucrose…).

Đề tài so sánh 5 mô hình hồi quy, kết hợp **phân tích quan hệ xám (Grey Relational
Analysis – GRA)** để xếp hạng đặc trưng, **thêm đặc trưng tuần tự** để chọn tập đặc trưng
tốt nhất, và **SHAP** để diễn giải mô hình thắng cuộc.

Phương pháp tham chiếu: *Wang et al. (2024), International Journal of Hydrogen Energy 68, 388–397.*

---

## 2. Dữ liệu

`src/data/Dataset.new3.csv` — 11 đặc trưng đầu vào → 1 biến mục tiêu `HPR`:

```
pH, VSS, Ethanol, Acetate, Propionate, Butyrate,
Sucrose_Degradation, ORP_Mid, ORP_Low, VFA, COD_O   →   HPR
```

(Kèm `src/data/pilot-performance.xls` — dữ liệu vận hành thí điểm.)

---

## 3. Kiến trúc & nội dung thư mục

```
tn-da22tta-110122050-tranminhdien-dien/
├── docs/
│   └── report.pdf              # Quyển khóa luận tốt nghiệp (bản đầy đủ)
└── src/
    ├── run_pipeline.py         # Điểm chạy pipeline ZenML (MLOps)
    ├── requirements.txt        # Thư viện Python cần cài
    ├── data/                   # Bộ dữ liệu huấn luyện
    ├── ml_train/               # PIPELINE 1 — paper-aligned
    │   └── train_pipeline.py   #   (GRA → 5 mô hình → thêm đặc trưng → SHAP)
    ├── pipelines/              # PIPELINE 2 — ZenML
    │   └── training_pipeline.py
    ├── steps/                  # Các bước ZenML: ingest → clean → train → evaluate
    └── src/                    # Mã mô hình dùng cho pipeline ZenML
```

Dự án có **hai pipeline huấn luyện**:

- **`ml_train/` (paper-aligned)** — bám sát phương pháp trong khóa luận: tiền xử lý,
  chuẩn hóa MinMax (X, y), GRA (ξ = 0.5), huấn luyện 5 mô hình, thêm đặc trưng tuần tự
  theo thứ hạng GRA, chọn mô hình tốt nhất (R² cao nhất → MSE thấp nhất), lưu mô hình +
  tính SHAP. Đây là pipeline mà web demo sử dụng.
- **`pipelines/` + `steps/` + `src/` (ZenML)** — pipeline MLOps chuẩn hóa
  `ingest_df → clean_df → train_model → evaluate_model`, log số liệu (R², RMSE, MSE, MAE)
  vào **MLflow**.

Các mô hình hỗ trợ: **Random Forest, SVM, KNN, Decision Tree, XGBoost** (và ANN).

---

## 4. Phần mềm cần thiết

- Python 3.10–3.12
- Các thư viện chính: scikit-learn, xgboost, tensorflow/keras, shap, pandas, numpy,
  matplotlib, **mlflow**, **zenml** (xem đầy đủ trong `src/requirements.txt`).

---

## 5. Cách chạy

### Chuẩn bị môi trường

```bash
cd src
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Cách 1 — Pipeline paper-aligned (khuyến nghị, khớp nội dung khóa luận)

```bash
python ml_train/train_pipeline.py --file data/Dataset.new3.csv
# Excel nhiều sheet:
# python ml_train/train_pipeline.py --file data/pilot-performance.xls --sheet-name <ten_sheet>
```

Kết quả: in JSON tóm tắt mô hình tốt nhất ra màn hình, đồng thời lưu mô hình, bộ chuẩn hóa,
biểu đồ và giá trị SHAP.

### Cách 2 — Pipeline ZenML + MLflow

```bash
# Khởi tạo ZenML (lần đầu)
zenml init

python run_pipeline.py --data data/Dataset.new3.csv --model RandomForest
# --model chọn 1 trong: RandomForest | SVM | KNN | DecisionTree | XGBoost

# Xem kết quả theo dõi thí nghiệm:
mlflow ui
```

---

## 6. Tài liệu

- Quyển khóa luận đầy đủ: [`docs/report.pdf`](docs/report.pdf)

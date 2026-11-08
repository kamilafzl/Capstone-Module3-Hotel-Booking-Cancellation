# Hotel Booking Cancellation Prediction

**Capstone Project Module 3 Machine Learning (Klasifikasi)**
Created by: **Kamila Fazilatunisa**

---

## Project Introduction

Sebuah hotel di Portugal (data dari sebuah *resort hotel* di Algarve dan sebuah *city hotel* di Lisbon, periode kedatangan Juli 2015 – Agustus 2017) menghadapi masalah **pembatalan booking**: **36,83%** dari 83.573 booking dalam data dibatalkan. Angka ini sejalan dengan rata-rata industri global yang mendekati 40% (D-EDGE, 2018). Kamar hotel adalah *perishable inventory*. Kamar yang kosong karena pembatalan mendadak berarti pendapatan hilang (±€349 per booking), sedangkan *overbooking* tanpa dasar data berisiko memaksa hotel memindahkan tamu ke hotel lain (*walking guest*, ±€204 per kasus).

Project ini membangun **model machine learning klasifikasi** untuk memprediksi apakah sebuah booking akan dibatalkan. Model ini membantu **Revenue Management** dan **Reservation team** mengantisipasi pembatalan, misalnya dengan menjual ulang kamar secara terkendali, melakukan *reconfirmation*, atau menyesuaikan kebijakan deposit.

### Business Problem
- **Stakeholder:** Revenue Management (pengguna utama), Reservation & Front Office, Sales & Marketing.
- **Problem:** hotel tidak tahu sejak awal booking mana yang akan batal sehingga kehilangan pendapatan dari kamar kosong, sementara antisipasi tanpa data berisiko *walking guest*.
- **Goals:** (1) model dengan F2-Score ≥ 0,75, (2) mengetahui faktor utama pembatalan, (3) menurunkan total biaya akibat pembatalan ≥ 30% dibanding tanpa model.

### Dataset
- Sumber: *Hotel Booking Demand* (Antonio, de Almeida & Nunes, 2019, *Data in Brief* 22: 41–49).
- 83.573 baris × 11 kolom (10 fitur + target `is_canceled`). Setiap baris adalah satu booking.
- Fitur: `country`, `market_segment`, `previous_cancellations`, `booking_changes`, `deposit_type`, `days_in_waiting_list`, `customer_type`, `reserved_room_type`, `required_car_parking_spaces`, `total_of_special_requests`.

### Metric Evaluation
- **FN** (booking batal tapi diprediksi datang) → kamar kosong ≈ ADR €101,83 × 3,43 malam ≈ **€349,28**.
- **FP** (tamu datang tapi diprediksi batal) → *cost of walk* ≈ 2 × ADR ≈ **€203,66**.
- Karena FN lebih mahal, metric utama adalah **F2-Score** (recall lebih diprioritaskan), ditambah ROC-AUC, PR-AUC, dan metric bisnis **Total Cancellation Cost** untuk memilih *threshold*.

### Workflow
1. Business Problem Understanding
2. Data Understanding
3. Data Cleaning: *missing value* `country` → `Unknown`, analisis duplikat (87,8% baris identik dipertahankan karena merupakan booking berbeda), kategori invalid, outlier
4. Exploratory Data Analysis + Chi-Square test & Cramér's V
5. Data Preparation: One Hot Encoding, Binary Encoding (`country`), RobustScaler, dan eksperimen *feature engineering*
6. Modeling: benchmark 8 algoritma (5-Fold Stratified CV), 5 metode penanganan *imbalance*, *hyperparameter tuning* (RandomizedSearchCV), optimasi *threshold* berbasis biaya
7. Interpretasi (Feature Importance & SHAP), analisis kapan model dapat dipercaya, *error analysis*, *learning curve*, dan uji *data leakage*
8. Business Impact (cost analysis + sensitivitas)
9. Conclusion & Recommendation

### Result
| Item | Hasil |
| --- | --- |
| Model final | **LightGBM + RandomUnderSampler (tuned), threshold 0,51** |
| F2-Score (test) | **0,770** |
| Recall / Precision (test) | 79,4% / 68,7% |
| ROC-AUC / PR-AUC (test) | 0,891 / 0,848 |
| Penghematan biaya (data test) | **€1,25 juta (58,3%)** dibanding tanpa model |

Faktor utama pembatalan: `deposit_type` (Non Refund), `country` (tamu domestik PRT), `market_segment` (Groups & Online TA), dan riwayat pembatalan. Permintaan parkir, *special request*, dan perubahan booking menurunkan risiko.

---

## Repository Structure
| File | Keterangan |
| --- | --- |
| `Capstone_Module3_Hotel_Booking_Cancellation.ipynb` | Dokumentasi lengkap pengerjaan project (Jupyter Notebook) |
| `Model_Hotel_Booking_Cancellation.sav` | Model final (Pipeline preprocessing + LightGBM) dan threshold, disimpan dengan Pickle |
| `data_hotel_booking_demand.csv` | Dataset |
| `README.md` | Project introduction |

## Cara Menggunakan Model
```python
import pickle
import pandas as pd

model = pickle.load(open('Model_Hotel_Booking_Cancellation.sav', 'rb'))
booking = pd.DataFrame([{
    'country': 'PRT', 'market_segment': 'Online TA', 'previous_cancellations': 0, 'booking_changes': 0,
    'deposit_type': 'No Deposit', 'days_in_waiting_list': 0, 'customer_type': 'Transient',
    'reserved_room_type': 'A', 'required_car_parking_spaces': 0, 'total_of_special_requests': 0
}])
proba = model['pipeline'].predict_proba(booking[model['features']])[:, 1]
prediksi = (proba >= model['threshold']).astype(int)   # 1 = berisiko cancel
```

## Environment
Notebook dijalankan dengan Python 3.13, pandas 3.0, scikit-learn 1.9, imbalanced-learn 0.14, category_encoders 2.11, LightGBM 4.7, XGBoost 3.4, SHAP 0.54. File `.sav` sebaiknya di-*load* dengan versi library yang sama. Jika versi berbeda, jalankan ulang notebook untuk menghasilkan file model baru.

## Video Presentation
*(link video ditambahkan di sini)*

## References
1. Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.
2. Antonio, N., de Almeida, A., & Nunes, L. (2017). *Predicting hotel booking cancellations to decrease uncertainty and increase revenue*. Tourism & Management Studies, 13(2).
3. D-EDGE Hospitality Solutions via Hospitality Technology (2019). *Global Cancellation Rate of Hotel Reservations Reaches 40% on Average*.
4. eCornell. *The Cheapest and Best Approach to Overbooking*.

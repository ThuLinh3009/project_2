# Xây dựng hệ thống dự đoán kết quả của sinh viên

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=ThuLinh3009/project_2&branch=main&mainModule=demo.py)

> **Họ và tên:** Đoàn Thị Thu Linh
> **Lớp:** 12423TN 
> **MSSV:** 12423020  
> **Học phần:** Đồ án 2

---

## 1. Gioi thieu de tai

### Bai toan

Tai cac truong dai hoc, mot so sinh vien co nguy co truot mon hoc nhung khong duoc canh bao som de co ke hoach ho tro kip thoi. Bai toan dat ra la: **dua vao ket qua hoc tap HK1 – HK3, du doan sinh vien nao co kha nang truot mon o HK4** de giang vien va nha truong can thiep som.

### Muc tieu

- Xay dung mo hinh phan loai **Pass / Fail** cho **6 mon hoc HK4** cua sinh vien nganh **Ky thuat Phan mem (KTPM)**.
- Uu tien toi da hoa **Recall Fail** (bat dung cang nhieu sinh vien truot cang tot), dong thoi dam bao **F1 macro** o muc chap nhan duoc.
- Trien khai thanh ung dung web tuong tac (Streamlit) de demo truc tiep.

### 6 mon hoc du doan

| Mon hoc | Ki hieu trong dataset |
|---|---|
| Do an 1 | `do_an_1_test` |
| Kiem thu Phan mem | `kiem_thu_phan_mem_test` |
| Thiet ke Web co ban | `thiet_ke_web_co_ban_test` |
| Cong nghe Phan mem | `cong_nghe_phan_mem_test` |
| Tieng Anh 1 | `tieng_anh_1_test` |
| Xac suat Thong ke | `xac_suat_thong_ke_test` |

---

## 2. Dataset

### Nguon du lieu

Du lieu diem hoc tap cua **473 sinh vien** nganh KTPM (khoa K19, K20, K21) tai truong Dai hoc Cong nghe Thong tin — DHQG TP.HCM, thu thap qua 3 hoc ki (HK1 – HK3).

### Mo ta cac cot (features dau vao — 12 cot)

| Ten cot | Mo ta |
|---|---|
| `dai_so_tuyen_tinh` | Diem mon Dai so tuyen tinh (HK1) |
| `giai_tich` | Diem mon Giai tich (HK1) |
| `giai_tich_so` | Diem mon Giai tich so (HK2) |
| `kien_truc_may_tinh` | Diem mon Kien truc May tinh (HK2) |
| `lap_trinh_python_co_ban` | Diem mon Lap trinh Python (HK1) |
| `cau_truc_du_lieu_va_giai_thuat` | Diem mon Cau truc du lieu & Giai thuat (HK2) |
| `co_so_ky_thuat_lap_trinh` | Diem mon Co so ky thuat lap trinh (HK1) |
| `co_so_du_lieu` | Diem mon Co so du lieu (HK2) |
| `lap_trinh_huong_doi_tuong` | Diem mon Lap trinh huong doi tuong (HK2) |
| `lap_trinh_ung_dung_windows_form` | Diem mon Lap trinh ung dung Windows Form (HK3) |
| `phap_luat_dai_cuong` | Diem mon Phap luat dai cuong (HK2) |
| `tieng_anh_tang_cuong` | Diem mon Tieng Anh tang cuong (HK1) |

### Nhan du doan (6 cot target)

Gia tri: `pass` = 1, `fail` = 0

| Mon | Pass | Fail | Ti le Fail |
|---|---|---|---|
| Do an 1 | 392 | 81 | 17.1% |
| Kiem thu PM | 448 | 25 | 5.3% |
| Thiet ke Web | 415 | 58 | 12.3% |
| Cong nghe PM | 431 | 42 | 8.9% |
| Tieng Anh 1 | 441 | 32 | 6.8% |
| Xac suat TK | 434 | 39 | 8.2% |

> **File chinh:** `data/4.1_Dataset_final/clean_hk4-ktpm-with-tests.csv`

---

## 3. Pipeline

```
[1] Du lieu thu
    K19.xlsx / K20.xlsx / K21.xlsx
          |
          v
[2] Loc & chuan hoa
    - Giu lai HK1-HK3 (12 features)
    - Loai cac cot khong lien quan (the duc, ky nang mem, ...)
    - Tao nhan Pass/Fail cho 6 mon HK4
          |
          v
[3] Tien xu ly (trong Pipeline sklearn)
    SimpleImputer(strategy="median")   <- xu ly gia tri thieu
          |
          v
[4] Can bang lop
    BorderlineSMOTE(sampling_strategy=1.0)  <- oversampling lop Fail
          |
          v
[5] Chon dac trung — MultiMetricSelector (tu xay dung)
    Tinh toan 3 chi so:
      - F-score  (SelectKBest / f_classif)
      - Mutual Information  (mutual_info_classif)
      - Random Forest Feature Importance
    → Xep hang trung binh (Avg Rank) → giu top features
    → Loai bo features tuong quan cao (>0.8) giu cai tot hon
          |
          v
[6] Chuan hoa
    StandardScaler (tru trung binh, chia do lech chuan)
          |
          v
[7] Huan luyen — GridSearchCV (5-fold StratifiedKFold, scoring=f1_macro)
    4 mo hinh: LR | KNN | RF | SVM
          |
          v
[8] Chon best model tung mon
    Tieu chi: F1 macro cao nhat → uu tien Recall Fail cao nhat khi bang nhau
          |
          v
[9] Inference
    Dua vao diem HK1-HK3 cua sinh vien → du doan Pass/Fail 6 mon
    → Xuat danh sach sinh vien nguy co → Download Excel
```

---

## 4. Mo hinh su dung

| Mo hinh | Ly do chon |
|---|---|
| **Logistic Regression** | Nen tang, de giai thich, hieu qua voi du lieu nho, it overfit |
| **KNN** | Khong tham so, bat duoc bien gioi phi tuyen don gian |
| **Random Forest** | Ensemble manh, chiu duoc nhieu va outlier, cho feature importance |
| **SVM** | Hieu qua voi khong gian chieu cao, tot khi du lieu mat can bang |

Tat ca 4 mo hinh deu duoc boc trong **sklearn Pipeline** cung cau truc: `Imputer → SMOTE → FeatureSelector → Scaler → Model`, dam bao khong ro ri du lieu tu tap test sang tap train.

**Toi uu sieu tham so:** `GridSearchCV` voi 5-fold `StratifiedKFold` (giu nguyen ti le Pass/Fail moi fold).

---

## 5. Ket qua

### Best model tung mon (tren tap Test — 20% du lieu)

| Mon hoc | Best Model | So features | F1 Macro | Recall Fail |
|---|---|---|---|---|
| Do an 1 | **Logistic Regression** | 10 | **0.786** | **0.875** |
| Kiem thu PM | **Random Forest** | 8 | **0.632** | 0.600 |
| Thiet ke Web | **Random Forest** | 11 | **0.654** | 0.500 |
| Cong nghe PM | **KNN** | 8 | **0.615** | **0.750** |
| Tieng Anh 1 | **KNN** | 9 | **0.789** | 0.500 |
| Xac suat TK | **Logistic Regression** | 8 | **0.728** | **0.750** |

### So sanh 4 mo hinh tung mon (Recall Fail / F1 Macro)

| Mon | LR | KNN | RF | SVM |
|---|---|---|---|---|
| Do an 1 | 0.875 / 0.786 | 0.812 / 0.656 | 0.812 / 0.713 | 0.750 / 0.704 |
| Kiem thu PM | 0.600 / 0.598 | 0.600 / 0.598 | 0.600 / **0.632** | 0.600 / 0.620 |
| Thiet ke Web | 0.667 / 0.638 | 0.500 / 0.534 | 0.500 / **0.654** | 0.250 / 0.547 |
| Cong nghe PM | 0.625 / 0.604 | **0.750** / 0.615 | 0.625 / 0.595 | 0.625 / 0.578 |
| Tieng Anh 1 | 0.333 / 0.708 | 0.500 / **0.789** | 0.333 / 0.683 | 0.333 / 0.739 |
| Xac suat TK | **0.750** / **0.728** | 0.625 / 0.644 | 0.375 / 0.632 | 0.625 / 0.692 |

### Classification Report — Do an 1 (mon co ket qua tot nhat)

```
                precision    recall  f1-score   support

Khong dat (0)       0.54      0.88      0.67        16
      Dat (1)       0.97      0.85      0.91        79

     accuracy                           0.85        95
    macro avg       0.75      0.86      0.79        95
 weighted avg       0.90      0.85      0.87        95
```

> **Nhan xet:** Mo hinh uu tien bat dung sinh vien co nguy co truot (Recall Fail cao). Do du lieu mat can bang nghiem trong (ti le Fail 5–17%), Precision cua lop Fail con thap — day la danh doi chap nhan duoc trong bai toan canh bao som.

---

## 6. Huong dan chay

### Cach 1 — Chay tren Streamlit Cloud (khong can cai dat)

1. Truy cap link deploy:  
   **[>> Mo ung dung tren Streamlit Cloud <<](https://share.streamlit.io/deploy?repository=ThuLinh3009/project_2&branch=main&mainModule=demo.py)**
2. Sidebar → chon trang → thao tac truc tiep tren trinh duyet.

---

### Cach 2 — Chay local

#### Cai moi truong

```bash
# 1. Clone repo
git clone https://github.com/ThuLinh3009/project_2.git
cd project_2

# 2. Tao virtual environment (khuyen nghi)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Cai thu vien
pip install -r requirements.txt
```

#### Chay demo / inference

```bash
streamlit run demo.py
```

Trinh duyet tu dong mo tai `http://localhost:8501`

**Thu tu su dung:**
1. **Tong quan** — xem tong the du lieu
2. **EDA** — bieu do phan phoi, tuong quan
3. **Huan luyen** — nhan "Bat dau huan luyen tat ca 6 mon" (cho ~5–10 phut)
4. **Phan tich mon hoc** — Confusion Matrix, Feature Importance
5. **Sinh vien nguy co** — xem danh sach + tai Excel

> **Upload du lieu rieng:** Sidebar co nut "Tai len file CSV" — co the dung file CSV khac co cung dinh dang voi `clean_hk4-ktpm-with-tests.csv`.

#### Chay notebook (train / thi nghiem)

```bash
pip install jupyter
jupyter notebook app/Test_KTPM_fixed_n6.ipynb
```

---

## 7. Cau truc thu muc

<<<<<<< HEAD
- **Python 3.10+**
- **Streamlit** - giao dien web
- **scikit-learn** - mo hinh ML
- **imbalanced-learn** - BorderlineSMOTE
- **pandas / numpy** - xu ly du lieu
- **matplotlib / seaborn** - truc quan hoa
=======
```
project_2/
├── demo.py                               # Ung dung Streamlit chinh
├── requirements.txt                      # Thu vien Python can thiet
│
├── app/
│   ├── Test_KTPM_fixed_n2.ipynb         # Thi nghiem voi SMOTE k=2
│   └── Test_KTPM_fixed_n6.ipynb         # Thi nghiem voi SMOTE k=6 (ket qua chinh)
│
├── data/
│   ├── 01_DL_goc/                        # Du lieu diem goc (K19, K20, K21 — .xlsx)
│   ├── 2.1_CNTT/                         # Tach theo nganh CNTT (.csv)
│   ├── 2.2_KHMT/                         # Tach theo nganh KHMT (.csv)
│   ├── 2.3_KTPM/                         # Tach theo nganh KTPM (.csv)
│   ├── 3.1_Loc_DL_3HK/
│   │   ├── CNTT/                         # Du lieu CNTT sau loc 3 HK
│   │   ├── KHMT/                         # Du lieu KHMT sau loc 3 HK
│   │   └── KTPM/                         # Du lieu KTPM sau loc 3 HK
│   ├── 3.2_Dataset_HK4/                  # Dataset diem tung mon HK4
│   └── 4.1_Dataset_final/
│       ├── clean_hk4-ktpm-with-tests.csv # Dataset cuoi (dung huan luyen)
│       └── clean_hk4-ktpm.csv            # Dataset khong co nhan test
│
├── report/
│   ├── paper_fixed_n1.docx               # Bao cao khoa hoc (phien ban n=1)
│   └── paper_fixed_n6.docx               # Bao cao khoa hoc (phien ban n=6)
│
└── slide/
    ├── [DA2]_DoanThiThuLinh_final.pdf    # Slide bao cao (PDF)
    ├── project_2.pptx                    # Slide (ban goc)
    └── project_2_new.pptx               # Slide (ban cap nhat)
```

---

## Cong nghe su dung

| Thanh phan | Thu vien |
|---|---|
| Giao dien web | `streamlit` |
| Mo hinh ML | `scikit-learn` |
| Can bang lop | `imbalanced-learn` (BorderlineSMOTE) |
| Xu ly du lieu | `pandas`, `numpy` |
| Truc quan hoa | `matplotlib`, `seaborn` |
| Xuat Excel | `openpyxl` |
>>>>>>> b505fcb (Update README: add full sections (intro, dataset, pipeline, models, results, guide))

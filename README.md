# Du Doan Sinh Vien Co Nguy Co Truot Mon - Do An 2

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=ThuLinh3009/project_2&branch=main&mainModule=demo.py)

> **Sinh vien:** Doan Thi Thu Linh  
> **Do an:** Do An 2 - Ky thuat phan mem  
> **Chu de:** Xay dung mo hinh du doan sinh vien co nguy co truot mon hoc ky 4

---

## Gioi thieu

He thong su dung du lieu diem hoc tap **HK1 - HK3** cua sinh vien nganh **Ky thuat Phan mem (KTPM)** de du doan kha nang **Pass / Fail** cho 6 mon hoc tai **Hoc ky 4**.

**6 mon du doan:**
| Mon hoc | Nhan viet tat |
|---|---|
| Do an 1 | do_an_1_test |
| Kiem thu Phan mem | kiem_thu_phan_mem_test |
| Thiet ke Web co ban | thiet_ke_web_co_ban_test |
| Cong nghe Phan mem | cong_nghe_phan_mem_test |
| Tieng Anh 1 | tieng_anh_1_test |
| Xac suat Thong ke | xac_suat_thong_ke_test |

---

## Chay Demo tren Streamlit Cloud

1. Truy cap link sau de chay ung dung truc tiep tren trinh duyet (khong can cai dat):

   **[>> Nhan vao day de mo ung dung <<](https://share.streamlit.io/deploy?repository=ThuLinh3009/project_2&branch=main&mainModule=demo.py)**

2. Hoac tu deploy:
   - Dang nhap [share.streamlit.io](https://share.streamlit.io)
   - Chon **New app**
   - Repository: `ThuLinh3009/project_2`
   - Branch: `main`
   - Main file: `demo.py`
   - Nhan **Deploy**

---

## Cau truc thu muc

```
project_2/
├── demo.py                          # Ung dung Streamlit chinh
├── requirements.txt                 # Cac thu vien can thiet
├── app/
│   ├── Test_KTPM_fixed_n2.ipynb    # Notebook thi nghiem (n=2)
│   └── Test_KTPM_fixed_n6.ipynb    # Notebook thi nghiem (n=6)
├── data/
│   ├── 01_DL_goc/                  # Du lieu diem goc K19, K20, K21
│   ├── 2.1_CNTT/                   # Du lieu nganh CNTT
│   ├── 2.2_KHMT/                   # Du lieu nganh KHMT
│   ├── 2.3_KTPM/                   # Du lieu nganh KTPM
│   ├── 3.1_Loc_DL_3HK/             # Du lieu sau loc 3 hoc ki (CNTT/KHMT/KTPM)
│   ├── 3.2_Dataset_HK4/            # Dataset tung mon HK4
│   └── 4.1_Dataset_final/          # Dataset cuoi cung dung huan luyen
│       └── clean_hk4-ktpm-with-tests.csv
├── report/
│   ├── paper_fixed_n1.docx         # Bao cao khoa hoc
│   └── paper_fixed_n6.docx
└── slide/
    ├── [DA2]_DoanThiThuLinh_final.pdf
    ├── project_2.pptx
    └── project_2_new.pptx
```

---

## Pipeline Mo Hinh

```
Du lieu dau vao (HK1-HK3)
        |
        v
  SimpleImputer (median)
        |
        v
  BorderlineSMOTE (xu ly mat can bang lop)
        |
        v
  MultiMetricSelector (F-score + Mutual Info + RF Importance)
        |
        v
  StandardScaler
        |
        v
  Mo hinh phan loai (GridSearchCV - 5-fold StratifiedKFold)
  - Logistic Regression
  - KNN
  - Random Forest
  - SVM
```

**Chon best model:** F1 Macro cao nhat, uu tien Recall Fail cao nhat khi bang nhau.

---

## Cac Trang trong Ung Dung

| Trang | Noi dung |
|---|---|
| **Tong quan** | Thong ke so luong sinh vien, features, ti le Pass/Fail |
| **EDA** | Phan phoi diem, ma tran tuong quan, boxplot |
| **Huan luyen** | Chay GridSearchCV, hien thi ket qua tung mon |
| **Phan tich mon hoc** | Confusion matrix, feature importance, so sanh mo hinh |
| **Sinh vien nguy co** | Danh sach sinh vien co nguy co truot, xuat Excel |

---

## Chay Local (Tuy chon)

```bash
git clone https://github.com/ThuLinh3009/project_2.git
cd project_2
pip install -r requirements.txt
streamlit run demo.py
```

---

## Du Lieu

- **473 sinh vien** nganh KTPM (K19, K20, K21)
- **12 features:** diem cac mon hoc HK1 - HK3
- **6 nhan du doan:** Pass/Fail cho 6 mon HK4
- File chinh: `data/4.1_Dataset_final/clean_hk4-ktpm-with-tests.csv`
- Co the tai len file CSV khac tu sidebar trong ung dung

---

## Cong Nghe Su Dung

- **Python 3.10+**
- **Streamlit** - giao dien web
- **scikit-learn** - mo hinh ML
- **imbalanced-learn** - BorderlineSMOTE
- **pandas / numpy** - xu ly du lieu
- **matplotlib / seaborn** - truc quan hoa
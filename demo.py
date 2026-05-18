import streamlit as st
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from collections import Counter
import io

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss,
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE

# ─── Cau hinh trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Du doan sinh vien nguy co truot",
    page_icon="🎓",
    layout="wide",
)

# ─── Hang so ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGET_COLS = [
    "do_an_1_test", "kiem_thu_phan_mem_test", "thiet_ke_web_co_ban_test",
    "cong_nghe_phan_mem_test", "tieng_anh_1_test", "xac_suat_thong_ke_test",
]
TARGET_VI = {
    "do_an_1_test":              "Do an 1",
    "kiem_thu_phan_mem_test":    "Kiem thu PM",
    "thiet_ke_web_co_ban_test":  "Thiet ke Web",
    "cong_nghe_phan_mem_test":   "Cong nghe PM",
    "tieng_anh_1_test":          "Tieng Anh 1",
    "xac_suat_thong_ke_test":    "Xac suat TK",
}
DROP_COLS = [
    "student_id", "ky_nang_mem",
    "giao_duc_the_chat_1", "giao_duc_the_chat_2", "giao_duc_the_chat_3",
]
FEAT_VI = {
    "dai_so_tuyen_tinh":              "Dai so tuyen tinh",
    "giai_tich":                      "Giai tich",
    "giai_tich_so":                   "Giai tich so",
    "kien_truc_may_tinh":             "Kien truc may tinh",
    "lap_trinh_python_co_ban":        "Lap trinh Python",
    "cau_truc_du_lieu_va_giai_thuat": "Cau truc du lieu",
    "co_so_ky_thuat_lap_trinh":       "Co so ky thuat LT",
    "co_so_du_lieu":                  "Co so du lieu",
    "lap_trinh_huong_doi_tuong":      "LT huong doi tuong",
    "lap_trinh_ung_dung_windows_form":"LT ung dung Windows",
    "phap_luat_dai_cuong":            "Phap luat dai cuong",
    "tieng_anh_tang_cuong":           "Tieng Anh tang cuong",
}
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
PARAM_GRIDS = {
    "Logistic Regression": {
        "feat__avg_rank_threshold": [8, 9],
        "model__C": [0.01, 0.1, 1, 10],
    },
    "KNN": {
        "feat__avg_rank_threshold": [8, 9],
        "model__n_neighbors": [9, 11, 13, 15, 21],
        "model__weights": ["uniform"],
    },
    "Random Forest": {
        "feat__avg_rank_threshold": [8, 9],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [8, 12, 20],
        "model__n_estimators": [100, 200],
    },
    "SVM": [
        {
            "feat__avg_rank_threshold": [8, 9],
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1, 5, 10, 50, 100],
            "model__gamma": ["scale", "auto", 0.01, 0.1],
        },
        {
            "feat__avg_rank_threshold": [8, 9],
            "model__kernel": ["linear"],
            "model__C": [0.1, 1, 5, 10, 50],
        },
    ],
}

# ─── ML helpers ──────────────────────────────────────────────────────────────
class MultiMetricSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, avg_rank_threshold=8,
                 corr_threshold=0.8, random_state=42):
        self.feature_names = feature_names
        self.avg_rank_threshold = avg_rank_threshold
        self.corr_threshold = corr_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        X_arr = np.array(X)
        feat = list(self.feature_names)
        sel_f = SelectKBest(f_classif, k="all").fit(X_arr, y)
        mi = mutual_info_classif(X_arr, y, random_state=self.random_state)
        rf_fi = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state).fit(X_arr, y)
        rdf = pd.DataFrame({
            "Dac trung": feat,
            "F-score": sel_f.scores_,
            "Mut.Info": mi,
            "RF Imp.": rf_fi.feature_importances_,
        })
        rdf["Avg Rank"] = (
            rdf["F-score"].rank(ascending=False) +
            rdf["Mut.Info"].rank(ascending=False) +
            rdf["RF Imp."].rank(ascending=False)) / 3
        candidates = rdf[rdf["Avg Rank"] <= self.avg_rank_threshold]["Dac trung"].tolist()
        X_cand = pd.DataFrame(X_arr, columns=feat)[candidates]
        corr_mat = X_cand.corr().abs()
        selected = candidates.copy()
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                f1c, f2c = candidates[i], candidates[j]
                if f1c in selected and f2c in selected:
                    if corr_mat.loc[f1c, f2c] > self.corr_threshold:
                        r1 = rdf[rdf["Dac trung"] == f1c]["Avg Rank"].values[0]
                        r2 = rdf[rdf["Dac trung"] == f2c]["Avg Rank"].values[0]
                        selected.remove(f2c if r1 <= r2 else f1c)
        self.selected_features_ = selected
        self.selected_indices_ = [feat.index(f) for f in selected]
        return self

    def transform(self, X):
        return np.array(X)[:, self.selected_indices_]

    def get_feature_names_out(self):
        return np.array(self.selected_features_)


def new_smote(k):
    return BorderlineSMOTE(
        sampling_strategy=1.0, k_neighbors=max(1, k), random_state=RANDOM_STATE)


def build_models(k_smote, feat_cols):
    return {
        "Logistic Regression": Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("smote", new_smote(k_smote)),
            ("feat",  MultiMetricSelector(feature_names=feat_cols)),
            ("scl",   StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("smote", new_smote(k_smote)),
            ("feat",  MultiMetricSelector(feature_names=feat_cols)),
            ("scl",   StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=15, weights="uniform")),
        ]),
        "Random Forest": Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("smote", new_smote(k_smote)),
            ("feat",  MultiMetricSelector(feature_names=feat_cols)),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=3, min_samples_leaf=10,
                min_samples_split=20, max_features="sqrt",
                random_state=RANDOM_STATE)),
        ]),
        "SVM": Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("smote", new_smote(k_smote)),
            ("feat",  MultiMetricSelector(feature_names=feat_cols)),
            ("scl",   StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced",
                          probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def prepare(df, feat_cols, target):
    ids = df["student_id"]
    X = df.drop(columns=TARGET_COLS + DROP_COLS, errors="ignore")
    y = df[target].dropna().astype(int)
    X, y, ids = X.loc[y.index], y, ids.loc[y.index]
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, ids, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    return (X.reset_index(drop=True), y.reset_index(drop=True),
            Xtr.reset_index(drop=True), Xte.reset_index(drop=True),
            ytr.reset_index(drop=True), yte.reset_index(drop=True),
            itr.reset_index(drop=True), ite.reset_index(drop=True))


# ─── Load du lieu ────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file_content=None, file_name=None):
    if file_content is not None:
        df_raw = pd.read_csv(io.BytesIO(file_content))
    else:
        df_raw = pd.read_csv("data/4.1_Dataset_final/clean_hk4-ktpm-with-tests.csv")
    df = df_raw[df_raw["student_id"].astype(str).str.strip() != "so_tin_chi"].copy()
    df.reset_index(drop=True, inplace=True)
    df["student_id"] = df["student_id"].astype(str).str.strip()
    feat_cols = [c for c in df.columns if c not in TARGET_COLS and c not in DROP_COLS]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in TARGET_COLS:
        df[c] = df[c].astype(str).str.strip().map({"pass": 1, "fail": 0})
    return df, feat_cols


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("Du doan sinh vien")
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Tai len file CSV (tuy chon)", type="csv")

data_ok = True
try:
    if uploaded is not None:
        df, FEAT_COLS = load_data(
            file_content=uploaded.read(), file_name=uploaded.name)
    else:
        df, FEAT_COLS = load_data()
except FileNotFoundError:
    st.sidebar.error("Khong tim thay file CSV mac dinh.\nVui long tai len file.")
    data_ok = False
    df, FEAT_COLS = None, []

page = st.sidebar.radio(
    "Chon trang",
    ["Tong quan", "EDA", "Huan luyen", "Phan tich mon hoc", "Sinh vien nguy co"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Pipeline: Imputer -> SMOTE -> FeatureSelector -> Scaler -> Model")

if not data_ok:
    st.warning("Vui long tai len file du lieu de bat dau.")
    st.stop()

# =============================================================================
# TRANG 1 — TONG QUAN
# =============================================================================
if page == "Tong quan":
    st.title("Tong quan du lieu")
    st.markdown("**Du lieu:** Diem hoc tap HK1-HK3 cua 473 sinh vien nganh KTPM. "
                "**Muc tieu:** Du doan Pass/Fail cho 6 mon hoc HK4.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tong sinh vien", len(df))
    col2.metric("So features", len(FEAT_COLS))
    col3.metric("So mon du doan", len(TARGET_COLS))
    total_fail = sum((df[t] == 0).sum() for t in TARGET_COLS)
    col4.metric("Tong luot truot (6 mon)", int(total_fail))

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("5 dong dau du lieu")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("Gia tri NaN & Zero")
        nan_s = df[FEAT_COLS].isnull().sum()
        zero_s = (df[FEAT_COLS] == 0).sum()
        info_df = pd.DataFrame({"NaN": nan_s, "Zero": zero_s})
        info_df = info_df[info_df.any(axis=1)]
        if not info_df.empty:
            info_df.index = [FEAT_VI.get(i, i) for i in info_df.index]
            st.dataframe(info_df, use_container_width=True)
        else:
            st.success("Khong co NaN hoac Zero dang chu y.")

    with col_r:
        st.subheader("Phan phoi Pass/Fail theo mon")
        pf_rows = []
        for c in TARGET_COLS:
            vc = df[c].value_counts()
            p, f = int(vc.get(1, 0)), int(vc.get(0, 0))
            pf_rows.append({
                "Mon hoc": TARGET_VI[c],
                "Pass": p, "Fail": f,
                "Tong": p + f,
                "Fail %": f"{f / (p + f) * 100:.1f}%",
                "Ti le P:F": f"{p / max(f, 1):.1f}:1",
            })
        st.dataframe(pd.DataFrame(pf_rows), use_container_width=True, hide_index=True)

        st.subheader("Thong ke mo ta features")
        desc = df[FEAT_COLS].describe().round(2).T[["mean", "std", "min", "max"]]
        desc.index = [FEAT_VI.get(i, i) for i in desc.index]
        st.dataframe(desc, use_container_width=True)

# =============================================================================
# TRANG 2 — EDA
# =============================================================================
elif page == "EDA":
    st.title("Kham pha du lieu (EDA)")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Phan phoi diem", "Ma tran tuong quan", "Pass/Fail theo mon", "Boxplot"]
    )

    with tab1:
        st.subheader("Phan phoi diem tung mon hoc (12 features)")
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for ax, c in zip(axes.flatten(), FEAT_COLS):
            ax.hist(df[c].dropna(), bins=14, color="#4C72B0",
                    edgecolor="white", alpha=0.85)
            ax.set_title(FEAT_VI.get(c, c), fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes.flatten()[len(FEAT_COLS):]:
            ax.set_visible(False)
        plt.suptitle("Phan phoi diem 12 mon hoc (HK1-3)", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("Ma tran tuong quan giua cac features")
        corr = df[FEAT_COLS].corr()
        labels = [FEAT_VI.get(c, c) for c in corr.columns]
        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_plot = corr.copy()
        corr_plot.index = labels
        corr_plot.columns = labels
        sns.heatmap(corr_plot, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", linewidths=0.4, ax=ax,
                    annot_kws={"size": 7})
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.info("Tuong quan cao (>0.8) giua 2 features: MultiMetricSelector se loai feature kem hon.")

    with tab3:
        st.subheader("So luong Pass / Fail tung mon")
        pf = {}
        for c in TARGET_COLS:
            vc = df[c].value_counts()
            pf[TARGET_VI[c]] = {"Pass": int(vc.get(1, 0)), "Fail": int(vc.get(0, 0))}
        pf_plot = pd.DataFrame(pf).T
        fig, ax = plt.subplots(figsize=(10, 5))
        pf_plot.plot(kind="bar", ax=ax,
                     color=["#2ca02c", "#d62728"], edgecolor="white", alpha=0.85)
        ax.set_title("Phan phoi Dat / Khong dat — 6 mon HK4")
        plt.xticks(rotation=20, ha="right")
        for p in ax.patches:
            ax.annotate(str(int(p.get_height())),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Nhan xet mat can bang lop:**")
        cols_pf = st.columns(3)
        for i, c in enumerate(TARGET_COLS):
            vc = df[c].value_counts()
            f = int(vc.get(0, 0))
            total = len(df[c].dropna())
            cols_pf[i % 3].metric(
                TARGET_VI[c], f"Fail: {f}/{total}",
                delta=f"{f / total * 100:.1f}%",
                delta_color="inverse",
            )

    with tab4:
        st.subheader("Boxplot diem theo Pass/Fail")
        tgt_sel = st.selectbox(
            "Chon mon hoc:",
            TARGET_COLS,
            format_func=lambda x: TARGET_VI[x],
            key="eda_box_sel",
        )
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for ax, fc in zip(axes.flatten(), FEAT_COLS):
            tmp = df[[fc, tgt_sel]].dropna()
            tmp.boxplot(column=fc, by=tgt_sel, ax=ax,
                        boxprops=dict(color="#4C72B0"),
                        medianprops=dict(color="red"))
            ax.set_title(FEAT_VI.get(fc, fc), fontsize=8)
            ax.set_xlabel("0=Fail  1=Pass", fontsize=7)
            ax.tick_params(labelsize=7)
        for ax in axes.flatten()[len(FEAT_COLS):]:
            ax.set_visible(False)
        plt.suptitle(f"Phan phoi diem theo Pass/Fail — {TARGET_VI[tgt_sel]}", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =============================================================================
# TRANG 3 — HUAN LUYEN
# =============================================================================
elif page == "Huan luyen":
    st.title("Huan luyen mo hinh")
    st.markdown("""
    **Pipeline:** `Imputer → BorderlineSMOTE → MultiMetricSelector → StandardScaler → Model`

    **GridSearchCV:** 5-fold StratifiedKFold, scoring = F1 macro

    **Chon best model:** F1 macro cao nhat, sau do Recall Fail (phan thang khi bang nhau)
    """)

    if st.button("Bat dau huan luyen tat ca 6 mon", type="primary"):
        st.session_state.pop("train_results", None)
        st.session_state.pop("all_fail", None)

        results = {}
        all_fail_list = []
        summary_rows = []

        progress = st.progress(0, text="Dang chuan bi...")
        status_box = st.empty()
        total_steps = len(TARGET_COLS) * len(PARAM_GRIDS)
        step = 0

        for tgt in TARGET_COLS:
            tgt_name = TARGET_VI[tgt]
            status_box.info(f"Dang huan luyen: **{tgt_name}** ...")

            Xf_, yf_, Xtr_, Xte_, ytr_, yte_, _, _ = prepare(df, FEAT_COLS, tgt)
            k_ = min(2, int(ytr_.value_counts().min()) - 1)
            models = build_models(k_, FEAT_COLS)

            fitted_pipes = {}
            sel_rows = []

            for nm_, pipe_ in models.items():
                progress.progress(
                    step / total_steps,
                    text=f"{tgt_name} — {nm_}...",
                )
                grid_ = GridSearchCV(pipe_, PARAM_GRIDS[nm_], cv=CV5,
                                     scoring="f1_macro", n_jobs=-1, verbose=0)
                grid_.fit(Xtr_, ytr_)
                best_ = grid_.best_estimator_
                fitted_pipes[nm_] = best_

                yp_ = best_.predict(Xte_)
                pp_ = best_.predict_proba(Xte_)
                rec_ = recall_score(yte_, yp_, labels=[0], average=None, zero_division=0)[0]
                f1_ = f1_score(yte_, yp_, average="macro", zero_division=0)
                n_sel = len(best_.named_steps["feat"].selected_features_)

                sel_rows.append({
                    "Model": nm_,
                    "Recall_fail": rec_,
                    "F1_macro": f1_,
                    "N_features": n_sel,
                    "Log_loss": round(log_loss(yte_, pp_), 4),
                    "CV_score": round(grid_.best_score_, 4),
                    "Best_params": {
                        k.replace("model__", "").replace("feat__", "feat:"): v
                        for k, v in grid_.best_params_.items()
                    },
                    "pipe": best_,
                    "yp": yp_,
                    "yte": yte_,
                    "pp": pp_,
                })
                step += 1

            sel_df_ = pd.DataFrame(sel_rows).sort_values(
                ["F1_macro", "Recall_fail"], ascending=[False, False]
            ).reset_index(drop=True)
            best_nm_ = sel_df_.loc[0, "Model"]
            best_pipe_ = fitted_pipes[best_nm_]

            yp_full = best_pipe_.predict(Xf_)
            fail_mask = (yp_full == 0)
            ids_full = df.loc[yf_.index, "student_id"].reset_index(drop=True)

            fail_df_ = pd.DataFrame({
                "student_id": ids_full[fail_mask].values,
                "mon_hoc": tgt,
                "mon_hoc_vi": tgt_name,
                "du_doan": "fail",
                "thuc_te": yf_.values[fail_mask],
            })
            all_fail_list.append(fail_df_)

            results[tgt] = {
                "sel_rows": sel_rows,
                "sel_df": sel_df_,
                "best_nm": best_nm_,
                "Xte": Xte_, "yte": yte_,
                "fitted_pipes": fitted_pipes,
            }
            summary_rows.append({
                "Mon hoc": tgt_name,
                "Best Model": best_nm_,
                "F1 macro": round(float(sel_df_.loc[0, "F1_macro"]), 4),
                "Recall Fail": round(float(sel_df_.loc[0, "Recall_fail"]), 4),
                "So features": int(sel_df_.loc[0, "N_features"]),
                "Du doan Fail": int(fail_mask.sum()),
                "Thuc te Fail": int((yf_ == 0).sum()),
            })

        progress.progress(1.0, text="Hoan thanh!")
        status_box.success("Huan luyen xong tat ca 6 mon!")

        st.session_state["train_results"] = results
        st.session_state["all_fail"] = pd.concat(all_fail_list, ignore_index=True)
        st.session_state["summary_df"] = pd.DataFrame(summary_rows)

    if "train_results" in st.session_state:
        st.markdown("---")
        st.subheader("Ket qua tong hop — Best model tung mon")
        sum_df = st.session_state["summary_df"]
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

        # Bar chart tong hop
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len(TARGET_COLS))
        names = [TARGET_VI[t] for t in TARGET_COLS]
        f1_vals = sum_df["F1 macro"].values
        rf_vals = sum_df["Recall Fail"].values

        axes[0].bar(x, f1_vals, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        axes[0].set_ylim(0, 1); axes[0].set_title("F1 macro (best model) theo mon")
        axes[0].set_ylabel("F1 macro")
        for i, v in enumerate(f1_vals):
            axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

        axes[1].bar(x, rf_vals, color="#d62728", edgecolor="white", alpha=0.85)
        axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        axes[1].set_ylim(0, 1); axes[1].set_title("Recall Fail (best model) theo mon")
        axes[1].set_ylabel("Recall Fail")
        for i, v in enumerate(rf_vals):
            axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.subheader("Chi tiet 4 model — tung mon")
        tgt_sel = st.selectbox(
            "Chon mon hoc:",
            TARGET_COLS,
            format_func=lambda x: TARGET_VI[x],
            key="train_detail_sel",
        )
        res = st.session_state["train_results"][tgt_sel]
        best_nm = res["best_nm"]
        rows_disp = []
        for r in res["sel_rows"]:
            rows_disp.append({
                "Model": r["Model"],
                "CV F1 macro": r["CV_score"],
                "Test F1 macro": round(r["F1_macro"], 4),
                "Recall Fail": round(r["Recall_fail"], 4),
                "So features": r["N_features"],
                "Log Loss": r["Log_loss"],
                "Best params": str(r["Best_params"]),
            })
        detail_df = pd.DataFrame(rows_disp).sort_values(
            ["Test F1 macro", "Recall Fail"], ascending=[False, False]
        )

        def highlight_best(row):
            return ["background-color: #fffacd"] * len(row) \
                if row["Model"] == best_nm else [""] * len(row)

        st.dataframe(
            detail_df.style.apply(highlight_best, axis=1),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"Best model duoc chon: **{best_nm}** (highlight vang)")
    else:
        st.info("Nhan nut **Bat dau huan luyen** de chay mo hinh. Qua trinh mat khoang 5-10 phut.")

# =============================================================================
# TRANG 4 — PHAN TICH TUNG MON HOC
# =============================================================================
elif page == "Phan tich mon hoc":
    st.title("Phan tich chi tiet tung mon hoc")

    if "train_results" not in st.session_state:
        st.warning("Vui long chay **Huan luyen** truoc.")
        st.stop()

    tgt_sel = st.selectbox(
        "Chon mon hoc:",
        TARGET_COLS,
        format_func=lambda x: TARGET_VI[x],
    )
    res = st.session_state["train_results"][tgt_sel]
    yte = res["yte"]
    best_nm = res["best_nm"]

    st.markdown(
        f"**Best model:** `{best_nm}` &nbsp;|&nbsp; "
        f"**F1 macro:** `{res['sel_df'].loc[0,'F1_macro']:.4f}` &nbsp;|&nbsp; "
        f"**Recall Fail:** `{res['sel_df'].loc[0,'Recall_fail']:.4f}`"
    )
    st.markdown("---")

    # Confusion matrix 4 models
    st.subheader("Confusion Matrix — 4 Models")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, r in zip(axes, res["sel_rows"]):
        cm = confusion_matrix(r["yte"], r["yp"])
        ConfusionMatrixDisplay(cm, display_labels=["Khong dat", "Dat"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        is_best = (r["Model"] == best_nm)
        ax.set_title(
            f"{'[BEST] ' if is_best else ''}{r['Model']}\n"
            f"F1m={r['F1_macro']:.3f} | RecFail={r['Recall_fail']:.3f}",
            fontsize=9,
            color="green" if is_best else "black",
        )
    plt.suptitle(f"Confusion Matrix — {TARGET_VI[tgt_sel]}", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # So luong Pass/Fail thuc te vs du doan
    st.markdown("---")
    st.subheader("So luong Pass/Fail: Thuc te vs Du doan")
    cols_pf = st.columns(4)
    for col, r in zip(cols_pf, res["sel_rows"]):
        with col:
            is_best = (r["Model"] == best_nm)
            label = f"{'⭐ ' if is_best else ''}{r['Model']}"
            st.markdown(f"**{label}**")
            act_f = int((r["yte"] == 0).sum())
            act_p = int((r["yte"] == 1).sum())
            pred_f = int((r["yp"] == 0).sum())
            pred_p = int((r["yp"] == 1).sum())
            cmp_data = pd.DataFrame({
                "": ["Pass", "Fail"],
                "Thuc te": [act_p, act_f],
                "Du doan": [pred_p, pred_f],
            })
            st.dataframe(cmp_data, hide_index=True, use_container_width=True)

            fig_b, ax = plt.subplots(figsize=(3, 2.5))
            x_pos = np.arange(2)
            ax.bar(x_pos - 0.2, [act_p, act_f], 0.35,
                   label="Thuc te", color=["#2ca02c", "#d62728"], alpha=0.8)
            ax.bar(x_pos + 0.2, [pred_p, pred_f], 0.35,
                   label="Du doan", color=["#98df8a", "#ff9896"], alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(["Pass", "Fail"], fontsize=8)
            ax.legend(fontsize=7)
            ax.set_title(r["Model"], fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close(fig_b)

    # Classification report
    st.markdown("---")
    st.subheader(f"Classification Report — {best_nm}")
    best_r = next(r for r in res["sel_rows"] if r["Model"] == best_nm)
    report = classification_report(
        best_r["yte"], best_r["yp"],
        target_names=["Khong dat (0)", "Dat (1)"],
        zero_division=0,
    )
    st.code(report)

    # Features duoc chon
    st.markdown("---")
    st.subheader(f"Features duoc chon boi best model ({best_nm})")
    best_pipe_obj = res["fitted_pipes"][best_nm]
    sel_feats = best_pipe_obj.named_steps["feat"].selected_features_
    feat_df = pd.DataFrame({
        "Feature": sel_feats,
        "Ten tieng Viet": [FEAT_VI.get(f, f) for f in sel_feats],
    })
    st.dataframe(feat_df, hide_index=True, use_container_width=True)

# =============================================================================
# TRANG 5 — SINH VIEN NGUY CO
# =============================================================================
elif page == "Sinh vien nguy co":
    st.title("Sinh vien co nguy co truot mon")

    if "all_fail" not in st.session_state:
        st.warning("Vui long chay **Huan luyen** truoc.")
        st.stop()

    all_fail = st.session_state["all_fail"]
    sum_df = st.session_state["summary_df"]

    # Tong hop theo sinh vien
    student_grp = (
        all_fail.groupby("student_id")["mon_hoc_vi"]
        .apply(list)
        .reset_index()
        .rename(columns={"mon_hoc_vi": "Mon nguy co truot"})
    )
    student_grp["So mon"] = student_grp["Mon nguy co truot"].apply(len)
    student_grp = student_grp.sort_values(
        ["So mon", "student_id"], ascending=[False, True]
    ).reset_index(drop=True)
    student_grp["Mon nguy co truot"] = student_grp["Mon nguy co truot"].apply(
        lambda x: ", ".join(x)
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tong sinh vien nguy co", len(student_grp))
    col2.metric("Truot >= 3 mon", int((student_grp["So mon"] >= 3).sum()))
    col3.metric("Truot >= 5 mon", int((student_grp["So mon"] >= 5).sum()))
    col4.metric("Truot ca 6 mon", int((student_grp["So mon"] == 6).sum()))

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Bo loc")
        min_fail = st.slider("So mon truot toi thieu", 1, 6, 1)
        mon_filter = st.multiselect(
            "Loc theo mon hoc cu the:",
            options=list(TARGET_VI.values()),
        )

    with col_r:
        st.subheader("Phan phoi so mon truot")
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = student_grp["So mon"].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values,
               color="#d62728", edgecolor="white", alpha=0.85)
        for i, (xi, v) in enumerate(zip(counts.index, counts.values)):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=9)
        ax.set_xlabel("So mon truot")
        ax.set_ylabel("So sinh vien")
        ax.set_title("Phan phoi so mon truot du doan")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Ap dung bo loc
    filtered = student_grp[student_grp["So mon"] >= min_fail].copy()
    if mon_filter:
        filtered = filtered[
            filtered["Mon nguy co truot"].apply(
                lambda x: any(m in x for m in mon_filter)
            )
        ]

    st.markdown("---")
    st.subheader(f"Danh sach sinh vien ({len(filtered)} sinh vien)")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Download
    st.markdown("---")
    st.subheader("Xuat ket qua Excel")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        all_fail[["student_id", "mon_hoc_vi", "du_doan", "thuc_te"]].rename(
            columns={
                "mon_hoc_vi": "mon_hoc",
                "thuc_te": "thuc_te (1=pass,0=fail)",
            }
        ).to_excel(writer, sheet_name="fail_by_subject", index=False)
        student_grp.to_excel(writer, sheet_name="fail_by_student", index=False)
        sum_df.to_excel(writer, sheet_name="best_models", index=False)
    output.seek(0)

    st.download_button(
        label="Tai xuong Excel",
        data=output,
        file_name="fail_students_demo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    st.subheader("Tom tat best model tung mon")
    st.dataframe(sum_df, use_container_width=True, hide_index=True)

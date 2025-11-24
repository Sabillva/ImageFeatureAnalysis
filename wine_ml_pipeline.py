"""
Wine dataset ile EDA, preprocessing, PCA, LDA, 12 model, validation karsilastirma,
test degerlendirmesi ve SHAP analizleri. Grafikleri ./outputs/ altina kaydediyorum.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import label_binarize

import shap
import joblib

# ---------- Ayarlar ----------
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 1. Veri Setinin Yüklenmesi ----------
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# İlk 5 satır
print("=== İlk 5 satir ===")
print(X.head())
X.head().to_csv(os.path.join(OUTPUT_DIR, "df_head5.csv"), index=False)

# ---------- 2. Veri Kalite Kontrolleri ----------
# 2.1 Eksik değer analizi
missing = X.isna().sum()
print("\n=== Eksik degerler (sutun bazinda) ===")
print(missing)
missing.to_csv(os.path.join(OUTPUT_DIR, "missing_per_column.csv"))

# 2.2 Aykırı Değer (IQR yöntemi örneği)
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
is_outlier = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))
outlier_counts = is_outlier.sum()
print("\n=== Aykirg değer sayilari (IQR yontemi) ===")
print(outlier_counts)
outlier_counts.to_csv(os.path.join(OUTPUT_DIR, "outlier_counts.csv"))

# Basit etki kontrolü: logistic regression validation accuracy ile outliers çıkarma karşılaştırması
# Not: Bu sadece kısa etkisini göstermek için küçük bir test olucak
scaler_tmp = StandardScaler()
X_scaled_tmp = pd.DataFrame(scaler_tmp.fit_transform(X), columns=X.columns)
# çıkarılmış veri şöyledir
rows_with_any_outlier = is_outlier.any(axis=1)
X_no_out = X_scaled_tmp.loc[~rows_with_any_outlier]
y_no_out = y.loc[~rows_with_any_outlier]

X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(
    X_scaled_tmp, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train_no_t, X_val_no_t, y_train_no_t, y_val_no_t = train_test_split(
    X_no_out, y_no_out, test_size=0.2, random_state=RANDOM_STATE, stratify=y_no_out
)

clf_tmp = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
clf_tmp.fit(X_train_t, y_train_t)
acc_with = accuracy_score(y_val_t, clf_tmp.predict(X_val_t))

clf_tmp2 = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
clf_tmp2.fit(X_train_no_t, y_train_no_t)
acc_without = accuracy_score(y_val_no_t, clf_tmp2.predict(X_val_no_t))

with open(os.path.join(OUTPUT_DIR, "outlier_effect.txt"), "w") as f:
    f.write(f"Validation accuracy WITH outliers: {acc_with:.4f}\n")
    f.write(f"Validation accuracy WITHOUT outliers: {acc_without:.4f}\n")

print("\n=== Aykiri degerlerin kisa etkisi (LogisticRegression ornegi) ===")
print(f"Acc with outliers: {acc_with:.4f}, without outliers: {acc_without:.4f}")

# 2.3 Veri tipi ve dağılım
dtypes = X.dtypes
print("\n=== Sutun dtype'lari ===")
print(dtypes)
dtypes.to_csv(os.path.join(OUTPUT_DIR, "dtypes.csv"))

num_cols = X.select_dtypes(include=[np.number]).shape[1]
cat_cols = X.select_dtypes(exclude=[np.number]).shape[1]
with open(os.path.join(OUTPUT_DIR, "num_cat_counts.txt"), "w") as f:
    f.write(f"Numeric columns: {num_cols}\nCategorical columns: {cat_cols}\n")
print(f"\nNumeric columns: {num_cols}, Categorical columns: {cat_cols}")

# ---------- 3. Keşifsel Veri Analizi (EDA) ----------
desc = X.describe().T
desc[['q1','q3']] = X.quantile([0.25,0.75]).T
desc = desc.rename(columns={'25%':'q1','75%':'q3'})
desc.to_csv(os.path.join(OUTPUT_DIR, "statistics_per_feature.csv"))
print("\n=== İstatistikler kaydedilmiştir ===")

# 3.2 Korelasyon matrisi ve heatmap
corr = X.corr(method='pearson')
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pearson_correlation_heatmap.png"))
plt.close()
print("Korelasyon heatmap kaydedilmiştir")

# En yüksek korelasyonlu 3 çift özellik
corr_abs = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
top_pairs = corr_abs.unstack().sort_values(ascending=False).dropna().head(3)
print("\n=== En yuksek korelasyonlu 3 cift ===")
print(top_pairs)
top_pairs.to_csv(os.path.join(OUTPUT_DIR, "top_3_corr_pairs.csv"))

# 3.3 Boxplot (tüm özellikler için küçük figürler halinde)
for col in X.columns:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=X[col])
    plt.title(f"Boxplot - {col}")
    safe_name = col.replace("/", "_").replace(" ", "_")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{safe_name}.png"))
    plt.close()
print("Boxplotlar kaydedilmiştir")

# ---------- 4. Veri Ölçeklendirme ----------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
X_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_scaled.csv"), index=False)
print("Veri olceklendirme tamamlanmiştir ve kaydedilmiştir (X_scaled.csv).")

# ---------- 5. Veri Setinin Bölünmesi ----------
# Önce train+temp (train %70, temp %30) sonra temp'i validation %10 test %20 olarak bölüyoruz
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
# temp içinden validation=1/3 (=> 0.30 * 1/3 = 0.10 overall) ve test=2/3 (=> 0.20 overall)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"\nSplit sizes -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

# ---------- 6. Özellik Seçimi ve Boyut İndirgeme ----------
# 6.1 PCA
pca = PCA(random_state=RANDOM_STATE)
pca.fit(X_train)
explained = pca.explained_variance_ratio_

# choose components where explained variance ratio > mean(explained)
mean_ev = explained.mean()
chosen_idx = np.where(explained > mean_ev)[0]
if len(chosen_idx) < 2:
    # en az 2 component seçiyoruz (grafik için)
    n_comp_pca = max(2, np.argmax(np.cumsum(explained) >= 0.90) + 1)
else:
    n_comp_pca = len(chosen_idx)

pca = PCA(n_components=n_comp_pca, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# explained variance grafiği
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA - cumulative explained variance")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_explained_variance.png"))
plt.close()

# 2D scatter of first two PCA components
if pca.n_components_ >= 2:
    plt.figure(figsize=(7,5))
    df_temp = pd.DataFrame(X_train_pca[:, :2], columns=["PC1","PC2"])
    df_temp['target'] = y_train.values
    sns.scatterplot(x="PC1", y="PC2", hue="target", palette="Set1", data=df_temp)
    plt.title("PCA: First two components (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_scatter_2d.png"))
    plt.close()

# 6.2 LDA
n_classes = y.nunique()
lda_n_comp = min(3, n_classes - 1)  # LDA can have at most n_classes-1 components
lda = LDA(n_components=lda_n_comp)
X_train_lda = lda.fit_transform(X_train, y_train)
X_val_lda = lda.transform(X_val)
X_test_lda = lda.transform(X_test)

# LDA scatter (ilk ikisi)
if lda_n_comp >= 2:
    plt.figure(figsize=(7,5))
    df_temp2 = pd.DataFrame(X_train_lda[:, :2], columns=["LD1","LD2"])
    df_temp2['target'] = y_train.values
    sns.scatterplot(x="LD1", y="LD2", hue="target", palette="Set1", data=df_temp2)
    plt.title("LDA: First two components (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lda_scatter_2d.png"))
    plt.close()

# ---------- 7. Modellerin Kurulması ----------
models = {
    "LogisticRegression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "GaussianNB": GaussianNB()
}

representations = {
    "raw": (X_train, X_val, X_test),
    "pca": (pd.DataFrame(X_train_pca), pd.DataFrame(X_val_pca), pd.DataFrame(X_test_pca)),
    "lda": (pd.DataFrame(X_train_lda), pd.DataFrame(X_val_lda), pd.DataFrame(X_test_lda))
}

results = []

def evaluate_on_validation(clf, Xv, yv):
    y_pred = clf.predict(Xv)
    # multi-class metrics -> use macro averaging
    acc = accuracy_score(yv, y_pred)
    prec = precision_score(yv, y_pred, average='macro', zero_division=0)
    rec = recall_score(yv, y_pred, average='macro', zero_division=0)
    f1 = f1_score(yv, y_pred, average='macro', zero_division=0)
    # roc_auc for multiclass (ovr)
    try:
        yv_bin = label_binarize(yv, classes=np.unique(y))
        proba = clf.predict_proba(Xv)
        roc = roc_auc_score(yv_bin, proba, average='macro', multi_class='ovr')
    except Exception:
        roc = np.nan
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

# Train all models on each representation
for rep_name, (Xtr, Xval, Xte) in representations.items():
    # ensure DataFrame/ndarray shapes are consistent
    Xtr_arr = np.array(Xtr)
    Xval_arr = np.array(Xval)
    for model_name, model in models.items():
        clf = model
        clf.fit(Xtr_arr, y_train)
        metrics = evaluate_on_validation(clf, Xval_arr, y_val)
        results.append({
            "representation": rep_name,
            "model": model_name,
            **metrics
        })
        # modeli kaydediyoruz
        model_fname = f"{rep_name}_{model_name}.joblib"
        joblib.dump(clf, os.path.join(OUTPUT_DIR, model_fname))

results_df = pd.DataFrame(results)
results_df.to_excel(os.path.join(OUTPUT_DIR, "validation_results.xlsx"), index=False)
print("\nValidation sonuclari kaydedilmiştir: validation_results.xlsx")
print(results_df)

# ---------- 8. Validation Karşılaştırma Tablosu (kaydedilenler) ----------
# Tablo zaten kaydedildi yukarıda. Fakat biraz daha okunaklı çıktı için:
print("\n=== Top validation results sorted by roc_auc (desc, nan last) ===")
print(results_df.sort_values(by=["roc_auc","f1"], ascending=[False, False]).head(10))

# ---------- 9. En İyi Modelin Test Üzerinde Değerlendirilmesi ----------
# Seçim ölçütü: validation roc_auc (en yüksek), eşitliktir -> f1 yüksek olan seçilir.
valid_sorted = results_df.copy()
# replace nan with -inf for sorting to put them at bottom
valid_sorted['roc_auc_sort'] = valid_sorted['roc_auc'].fillna(-999)
best_row = valid_sorted.sort_values(by=['roc_auc_sort', 'f1'], ascending=[False, False]).iloc[0]
print("\n=== Secilen En İyi Model (on validation) ===")
print(best_row)

best_repr = best_row['representation']
best_model_name = best_row['model']
best_model_path = os.path.join(OUTPUT_DIR, f"{best_repr}_{best_model_name}.joblib")
best_clf = joblib.load(best_model_path)

# Test'te değerlendirme
X_test_final = np.array(representations[best_repr][2])
y_pred_test = best_clf.predict(X_test_final)
metrics_test = {
    "accuracy": accuracy_score(y_test, y_pred_test),
    "precision": precision_score(y_test, y_pred_test, average='macro', zero_division=0),
    "recall": recall_score(y_test, y_pred_test, average='macro', zero_division=0),
    "f1": f1_score(y_test, y_pred_test, average='macro', zero_division=0)
}
# ROC-AUC test
try:
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    proba_test = best_clf.predict_proba(X_test_final)
    metrics_test["roc_auc"] = roc_auc_score(y_test_bin, proba_test, average='macro', multi_class='ovr')
except Exception:
    metrics_test["roc_auc"] = np.nan

print("\n=== Test metrics for best model ===")
print(metrics_test)
pd.Series(metrics_test).to_csv(os.path.join(OUTPUT_DIR, "best_model_test_metrics.csv"))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix: {best_repr} + {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.png"))
plt.close()

# ROC curve(s) for multiclass: one-vs-rest plotting
try:
    # her sınıf için  sklearn's RocCurveDisplay kullanıyoruz
    from sklearn.metrics import roc_curve, auc
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    proba_test = best_clf.predict_proba(X_test_final)
    n_classes_ = y_test_bin.shape[1]
    plt.figure(figsize=(8,6))
    for i in range(n_classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba_test[:, i])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_val:.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (test) - {best_repr} + {best_model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_model_roc_curves.png"))
    plt.close()
except Exception as e:
    print("ROC plot not available:", e)

# ---------- 10. XAI — SHAP Analizi ----------
# 10.1 En iyi validation model için SHAP
print("\n=== SHAP analysis for best model ===")
# prepare a small background set for KernelExplainer if needed
background_sample = 100
if isinstance(representations[best_repr][0], pd.DataFrame):
    background = representations[best_repr][0].sample(n=min(background_sample, len(representations[best_repr][0])), random_state=RANDOM_STATE).values
else:
    background = np.array(representations[best_repr][0])
    if background.shape[0] > background_sample:
        background = background[np.random.choice(background.shape[0], background_sample, replace=False)]

# choose explainer according to model type
tree_models = (DecisionTreeClassifier, RandomForestClassifier)
linear_models = (LogisticRegression,)

if isinstance(best_clf, RandomForestClassifier) or isinstance(best_clf, DecisionTreeClassifier):
    explainer = shap.TreeExplainer(best_clf)
    shap_values = explainer.shap_values(background)
    # For plotting summary, need shap values on validation set
    Xv_for_shap = representations[best_repr][1]
    if isinstance(Xv_for_shap, pd.DataFrame):
        Xv_vals = Xv_for_shap.values
        Xv_cols = Xv_for_shap.columns.tolist()
    else:
        Xv_vals = np.array(Xv_for_shap)
        # create generic column names if not available
        Xv_cols = [f"f{i}" for i in range(Xv_vals.shape[1])]
    # compute shap values for validation
    shap_values_val = explainer.shap_values(Xv_vals)
    # summary plot (class-wise if multiclass)
    for i, class_shap in enumerate(shap_values_val):
        plt.figure()
        shap.summary_plot(class_shap, Xv_vals, feature_names=Xv_cols, show=False)
        plt.title(f"SHAP summary - class {i}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_summary_{best_repr}_{best_model_name}_class{i}.png"), bbox_inches='tight')
        plt.close()
    # bar plot (mean abs)
    plt.figure()
    # shap.summary_plot with plot_type='bar'
    shap.summary_plot(shap_values_val, Xv_vals, feature_names=Xv_cols, plot_type="bar", show=False)
    plt.title("SHAP feature importance (bar)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_bar_{best_repr}_{best_model_name}.png"), bbox_inches='tight')
    plt.close()
elif isinstance(best_clf, LogisticRegression):
    try:
        explainer = shap.LinearExplainer(best_clf, background)
        Xv_for_shap = representations[best_repr][1]
        Xv_vals = Xv_for_shap.values if isinstance(Xv_for_shap, pd.DataFrame) else np.array(Xv_for_shap)
        shap_values_val = explainer.shap_values(Xv_vals)
        shap.summary_plot(shap_values_val, Xv_vals, feature_names=(Xv_for_shap.columns.tolist() if isinstance(Xv_for_shap, pd.DataFrame) else [f"f{i}" for i in range(Xv_vals.shape[1])]), show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_summary_{best_repr}_{best_model_name}.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("LinearExplainer failed, falling back to KernelExplainer:", e)
        explainer = shap.KernelExplainer(best_clf.predict_proba, background)
        Xv_for_shap = representations[best_repr][1]
        Xv_vals = Xv_for_shap.values if isinstance(Xv_for_shap, pd.DataFrame) else np.array(Xv_for_shap)
        shap_values = explainer.shap_values(Xv_vals[:100])  # limit for speed
        shap.summary_plot(shap_values, Xv_vals[:100], show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_summary_{best_repr}_{best_model_name}_kernel.png"), bbox_inches='tight')
        plt.close()
else:
    # fallback: KernelExplainer
    explainer = shap.KernelExplainer(best_clf.predict_proba, background)
    Xv_for_shap = representations[best_repr][1]
    Xv_vals = Xv_for_shap.values if isinstance(Xv_for_shap, pd.DataFrame) else np.array(Xv_for_shap)
    shap_values = explainer.shap_values(Xv_vals[:100])  # limit for speed
    shap.summary_plot(shap_values, Xv_vals[:100], show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_summary_{best_repr}_{best_model_name}_kernel.png"), bbox_inches='tight')
    plt.close()

print("SHAP gorselleri kaydedilmiştir")

# 10.2 PCA ve LDA temsilleri için SHAP karşılaştırması
# PCA/LDA temsilinde eğitilmiş modeller için SHAP çalıştırıyoruz
for rep in ["pca", "lda"]:
    print(f"\nSHAP comparison for: {rep}")

    model_path = os.path.join(OUTPUT_DIR, f"{rep}_RandomForest.joblib")
    if not os.path.exists(model_path):
        continue

    clf_rep = joblib.load(model_path)

    X_train_rep = representations[rep][0]
    X_val_rep = representations[rep][1]

    # Arka plan seti
    if isinstance(X_train_rep, pd.DataFrame):
        background = X_train_rep.sample(n=min(50, len(X_train_rep)), random_state=42).values
    else:
        background = np.array(X_train_rep)[:50]

    try:
        expl = shap.TreeExplainer(clf_rep)
        shap_vals = expl.shap_values(np.array(X_val_rep)[:100])

        plt.figure()
        shap.summary_plot(shap_vals, np.array(X_val_rep)[:100], plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {rep.upper()}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_bar_{rep}.png"), bbox_inches='tight')
        plt.close()

        print(f"{rep.upper()} SHAP bar plot kaydedilmiştir")

    except Exception as e:
        print(f"{rep} SHAP failed:", e)

print("PCA & LDA icin SHAP karsilastirmalari denenmiştir ve kaydedilmiştir")

# ---------- Kaydet: validation tablosu csv ve kısa rapor txt ----------
results_df.to_csv(os.path.join(OUTPUT_DIR, "validation_results.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "readme_pipeline.txt"), "w", encoding="utf-8") as f:
    f.write("Outputs:\n")
    f.write("- validation_results.xlsx / .csv: Validation metrikleri (tüm modeller)\n")
    f.write("- best_model_test_metrics.csv: En iyi modelin test sonuclari\n")
    f.write("- best_model_confusion_matrix.png, best_model_roc_curves.png\n")
    f.write("- pca_explained_variance.png, pca_scatter_2d.png, lda_scatter_2d.png\n")
    f.write("- shap_* : SHAP gorselleri\n")
    f.write("- boxplot_*.png, pearson_correlation_heatmap.png\n")
    f.write("\nNotlar:\n- LDA n_components otomatik olarak sinif sayisina gore ayarlandi \n- KernelExplainer agirdir; biraz zaman alabiliyor.\n")

print("\nPipeline tamamlanmiştir. Tum ciktilar ./outputs/ klasorundedir.")

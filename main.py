# 📦 Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import joblib
from tqdm import tqdm

# Sklearn tools for model building and evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ML Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 📁 Load Training Data (All 400 samples, but only 150 have labels)
train_df = pd.read_csv("train.csv")
train_labels_df = pd.read_csv("train_labels.csv")

print(f"Training data shape: {train_df.shape}")
print(f"Labeled training data shape: {train_labels_df.shape}")

# Check labeled vs unlabeled in train.csv
labeled_count = train_df['Class'].notna().sum()
unlabeled_count = train_df['Class'].isna().sum()
print(f"\nIn train.csv:")
print(f"Labeled samples: {labeled_count}")
print(f"Unlabeled samples: {unlabeled_count}")

# 📊 Class distribution in labeled subset
print(f"\nClass distribution in labeled data:")
print(train_labels_df['Class'].value_counts().sort_index())

# Merge labels into train_df for consistency (ensures alignment)
train_df = train_df.drop(columns=['Class'], errors='ignore')
train_df = train_df.merge(train_labels_df, on="Id")

# Extract gene columns and target
gene_cols = [col for col in train_df.columns if col.startswith("gene_")]
X = train_df[gene_cols]
y = train_df["Class"]

# 📁 Load test data (401 samples for prediction)
test_df = pd.read_csv('test.csv') 
print(f"Test data shape: {test_df.shape}")
print(f"Columns: Id + {test_df.shape[1]-1} gene features")

# Extract gene features from test
gene_cols = [col for col in train_df.columns if col.startswith('gene_')]
print(f"\nNumber of gene features: {len(gene_cols)}")

# Prepare test data
X_test = test_df[gene_cols]
test_ids = test_df['Id']

print(f"Test features shape: {X_test.shape}")
print(f"Test IDs shape: {test_ids.shape}")

# ⚙️ Data Preprocessing Pipeline
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

selector = SelectKBest(mutual_info_classif, k=50)
X_selected = selector.fit_transform(X_scaled, y)
X_test_selected = selector.transform(X_test_scaled)
selected_features = [gene_cols[i] for i in selector.get_support(indices=True)]

print(f"Selected top {len(selected_features)} features")

# 🔁 Define Models for Training
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "RandomForest": RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
}

results = {}

# 🧪 Train + Evaluate with TQDM
for name, model in models.items():
    print(f"\n🚀 Training {name}")
    fold_f1s = []

    for fold, (train_idx, val_idx) in tqdm(
        enumerate(cv.split(X_selected, y)),
        total=cv.get_n_splits(),
        desc=f"{name} CV"
    ):
        X_tr, X_val = X_selected[train_idx], X_selected[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average='macro')
        fold_f1s.append(f1)

        if fold == cv.get_n_splits() - 1:
            plt.figure(figsize=(5, 4))
            cm = confusion_matrix(y_val, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix: {name} (Fold {fold+1})")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(f"conf_matrix_{name}.png")
            plt.show()

    avg_f1 = np.mean(fold_f1s)
    results[name] = avg_f1
    print(f"✅ {name} Average F1 (macro): {avg_f1:.4f}")

# 📊 Model Comparison Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Macro F1-score by Model (5-fold CV)")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# 🏆 Final Model
best_model_name = max(results, key=results.get)
final_model = models[best_model_name]
final_model.fit(X_selected, y)
print(f"\nBest model: {best_model_name}")

# 💾 Save Artifacts
joblib.dump(final_model, "final_model.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "feature_selector.pkl")
pd.Series(selected_features).to_csv("selected_features.txt", index=False)

# 📤 Test Prediction
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)
X_test_selected = selector.transform(X_test_scaled)
preds = final_model.predict(X_test_selected)

submission = pd.DataFrame({
    'Id': test_ids,
    'Class': preds
})

submission.to_csv('submission.csv', index=False)

print("\n✅ Submission file created!")
print(f"Submission shape: {submission.shape}")
print(submission.head())

# 🧠 SHAP Interpretation
explainer = shap.Explainer(final_model, X_selected)
shap_values = explainer(X_selected)
shap.summary_plot(shap_values, features=X_selected, feature_names=selected_features, show=False)
plt.savefig("shap_summary.png")
plt.show()

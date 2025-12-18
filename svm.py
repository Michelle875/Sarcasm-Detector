import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

# ======================
# Load data
# ======================
train_tfidf = load_npz("train_tfidf.npz")
valid_tfidf = load_npz("valid_tfidf.npz")
test_tfidf  = load_npz("test_tfidf.npz")

train_df = pd.read_csv("train_features.csv")
valid_df = pd.read_csv("valid_features.csv")
test_df  = pd.read_csv("test_features.csv")

y_train = train_df["label"].astype(int).values
y_valid = valid_df["label"].astype(int).values
y_test  = test_df["label"].astype(int).values

numeric_cols = train_df.columns.difference(
    ["text", "tokens", "pos_seq", "label"]
)

X_train_extra = train_df[numeric_cols].astype(np.float64).values
X_valid_extra = valid_df[numeric_cols].astype(np.float64).values
X_test_extra  = test_df[numeric_cols].astype(np.float64).values

# ======================
# Dimensionality reduction
# ======================
svd = TruncatedSVD(n_components=150, random_state=42)

X_train_svd = svd.fit_transform(train_tfidf)
X_valid_svd = svd.transform(valid_tfidf)
X_test_svd  = svd.transform(test_tfidf)

# ======================
# Feature fusion
# ======================
X_train = np.hstack([X_train_svd, X_train_extra])
X_valid = np.hstack([X_valid_svd, X_valid_extra])
X_test  = np.hstack([X_test_svd,  X_test_extra])

# ======================
# Scaling
# ======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)

# ======================
# MODEL 1: Linear SVM with probability calibration
# ======================
print("Training Linear SVM...")
linear_params = {
    "C": [0.01, 0.1, 1, 10]
}

linear_svm_base = GridSearchCV(
    LinearSVC(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ),
    param_grid=linear_params,
    scoring="f1_weighted",
    cv=5,
    n_jobs=-1,
    verbose=1
)

linear_svm_base.fit(X_train, y_train)

# Wrap with CalibratedClassifierCV to enable predict_proba
print("Calibrating Linear SVM for probability estimates...")
linear_best = CalibratedClassifierCV(linear_svm_base.best_estimator_, cv=3)
linear_best.fit(X_train, y_train)

print(f"\nLinear SVM best params: {linear_svm_base.best_params_}")
print(f"Linear SVM best CV score: {linear_svm_base.best_score_:.4f}")

# ======================
# MODEL 2: RBF SVM
# ======================
print("\nTraining RBF SVM...")
rbf_params = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.001]
}

rbf_svm = GridSearchCV(
    SVC(
        kernel="rbf",
        max_iter=5000,
        class_weight="balanced",
        probability=True,  # Enable probability for RBF
        random_state=42
    ),
    param_grid=rbf_params,
    scoring="f1_weighted",
    cv=3,
    n_jobs=-1,
    verbose=1
)

rbf_svm.fit(X_train, y_train)
rbf_best = rbf_svm.best_estimator_

print(f"\nRBF SVM best params: {rbf_svm.best_params_}")
print(f"RBF SVM best CV score: {rbf_svm.best_score_:.4f}")

# ======================
# Validation comparison
# ======================
linear_val_preds = linear_best.predict(X_valid)
rbf_val_preds    = rbf_best.predict(X_valid)

linear_f1 = f1_score(y_valid, linear_val_preds, average="weighted")
rbf_f1    = f1_score(y_valid, rbf_val_preds, average="weighted")

print("\n=== VALIDATION RESULTS ===")
print(f"LinearSVC F1: {linear_f1:.4f}")
print(f"RBF SVC   F1: {rbf_f1:.4f}")

best_model = linear_best if linear_f1 >= rbf_f1 else rbf_best
print(f"\nSelected model: {type(best_model).__name__}")

# ======================
# Final test evaluation
# ======================
test_preds = best_model.predict(X_test)

test_acc = accuracy_score(y_test, test_preds)
test_f1  = f1_score(y_test, test_preds, average="weighted")

print("\n=== FINAL TEST PERFORMANCE ===")
print(f"Accuracy: {test_acc:.4f}")
print(f"F1-score: {test_f1:.4f}")

# ======================
# Save predictions for train/valid/test
# ======================
train_preds = best_model.predict(X_train)
train_probs = best_model.predict_proba(X_train)
train_pred_class_probs = train_probs[np.arange(len(train_preds)), train_preds]
pd.DataFrame({"prediction": train_preds, "probability": train_pred_class_probs, "label": y_train}).to_csv("svm_train_preds.csv", index=False)

valid_preds = best_model.predict(X_valid)
valid_probs = best_model.predict_proba(X_valid)
valid_pred_class_probs = valid_probs[np.arange(len(valid_preds)), valid_preds]
pd.DataFrame({"prediction": valid_preds, "probability": valid_pred_class_probs, "label": y_valid}).to_csv("svm_valid_preds.csv", index=False)

test_preds = best_model.predict(X_test)
test_probs = best_model.predict_proba(X_test)
test_pred_class_probs = test_probs[np.arange(len(test_preds)), test_preds]
pd.DataFrame({"prediction": test_preds, "probability": test_pred_class_probs, "label": y_test}).to_csv("svm_test_preds.csv", index=False)

# ======================
# Save model and preprocessing objects
# ======================
dump(best_model, "final_svm.pkl")
dump(svd, "svd_svm.joblib")
dump(scaler, "scaler_svm.joblib")

print("\n✓ SVM model, SVD, and scaler saved successfully!")
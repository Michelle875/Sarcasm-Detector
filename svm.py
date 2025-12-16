import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

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
# MODEL 1: Linear SVM (baseline)
# ======================
linear_params = {
    "C": [0.01, 0.1, 1, 10]
}

linear_svm = GridSearchCV(
    LinearSVC(
    max_iter=20000,
    class_weight="balanced",
    random_state=42
    ),
    param_grid=linear_params,
    scoring="f1_weighted",
    cv=5,
    n_jobs=-1,
    verbose=1
)

linear_svm.fit(X_train, y_train)
linear_best = linear_svm.best_estimator_

# ======================
# MODEL 2: RBF SVM
# ======================
rbf_params = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.001]
}

rbf_svm = GridSearchCV(
    SVC(
        kernel="rbf",
        max_iter=20000,
        class_weight="balanced"
    ),
    param_grid=rbf_params,
    scoring="f1_weighted",
    cv=3,
    n_jobs=-1,
    verbose=1
)

rbf_svm.fit(X_train, y_train)
rbf_best = rbf_svm.best_estimator_

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

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_tfidf = load_npz("train_tfidf.npz")
valid_tfidf = load_npz("valid_tfidf.npz")
test_tfidf  = load_npz("test_tfidf.npz")

train_df = pd.read_csv("train_features.csv")
valid_df = pd.read_csv("valid_features.csv")
test_df  = pd.read_csv("test_features.csv")

y_train = train_df["label"].astype(int).values
y_valid = valid_df["label"].astype(int).values
y_test  = test_df["label"].astype(int).values


numeric_cols = train_df.columns.difference(["text", "tokens", "pos_seq", "label"])

X_train_extra = train_df[numeric_cols].astype(np.float64).values
X_valid_extra = valid_df[numeric_cols].astype(np.float64).values
X_test_extra  = test_df[numeric_cols].astype(np.float64).values

svd = TruncatedSVD(n_components=200, random_state=42)

X_train_svd = svd.fit_transform(train_tfidf)
X_valid_svd = svd.transform(valid_tfidf)
X_test_svd  = svd.transform(test_tfidf)

X_train = np.hstack([X_train_svd, X_train_extra])
X_valid = np.hstack([X_valid_svd, X_valid_extra])
X_test  = np.hstack([X_test_svd,  X_test_extra])


n_estimators_list = [300, 600, 1000]
max_depth_list = [20, 30, None]
min_samples_split_list = [2, 4,8]

best_acc = -1
best_params = None
best_model = None

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:

            print(f"Trying: n={n_estimators}, depth={max_depth}, split={min_samples_split}")

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            preds_valid = model.predict(X_valid)
            acc_valid = accuracy_score(y_valid, preds_valid)

            print(f" → Validation accuracy: {acc_valid:.4f}\n")

            if acc_valid > best_acc:
                best_acc = acc_valid
                best_params = (n_estimators, max_depth, min_samples_split)
                best_model = model

print("\nBEST MODEL")
print("n_estimators:", best_params[0])
print("max_depth:", best_params[1])
print("min_samples_split:", best_params[2])
print("Validation Accuracy:", best_acc)


test_preds = best_model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)

print("\nFINAL TEST ACCURACY:", test_acc)

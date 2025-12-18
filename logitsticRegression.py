import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump

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


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)



solvers = ["liblinear", "lbfgs", "saga"]
C_values = [0.1, 1.0, 3.0, 10.0]
penalties = ["l2", "l1", "elasticnet"]
l1_ratios = [0.2, 0.5, 0.8]

best_acc = -1
best_model = None
best_params = None

for solver in solvers:
    for C in C_values:
        for penalty in penalties:

            if penalty == "l1" and solver != "liblinear" and solver != "saga":
                continue
            if penalty == "elasticnet" and solver != "saga":
                continue

            if penalty == "elasticnet":
                for l1_ratio in l1_ratios:
                    print(f"Trying: solver={solver}, C={C}, penalty={penalty}, l1_ratio={l1_ratio}")

                    model = LogisticRegression(
                        solver="saga",
                        C=C,
                        penalty="elasticnet",
                        l1_ratio=l1_ratio,
                        max_iter=5000,
                        random_state=42
                    )
                    model.fit(X_train, y_train)

                    acc = accuracy_score(y_valid, model.predict(X_valid))
                    print(f" → Validation accuracy: {acc:.4f}\n")

                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
                        best_params = (solver, C, penalty, l1_ratio)

            else:
                print(f"Trying: solver={solver}, C={C}, penalty={penalty}")

                model = LogisticRegression(
                    solver=solver,
                    C=C,
                    penalty=penalty,
                    max_iter=5000,
                    random_state=42
                )
                model.fit(X_train, y_train)

                acc = accuracy_score(y_valid, model.predict(X_valid))
                print(f" → Validation accuracy: {acc:.4f}\n")

                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = (solver, C, penalty)



print("\n BEST LOGISTIC REGRESSION MODEL")
print("Solver:", best_params[0])
print("C:", best_params[1])
print("Penalty:", best_params[2])
print("Validation Accuracy:", best_acc)


test_preds = best_model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)

print("\nFINAL TEST ACCURACY:", test_acc)

train_preds = best_model.predict(X_train)
train_probs = best_model.predict_proba(X_train)
train_pred_class_probs = train_probs[np.arange(len(train_preds)), train_preds]

pd.DataFrame({"prediction": train_preds, "probability": train_pred_class_probs, "label": y_train}).to_csv("logisticRegression_train_preds.csv", index=False)

valid_preds = best_model.predict(X_valid)
valid_probs = best_model.predict_proba(X_valid)
valid_pred_class_probs = valid_probs[np.arange(len(valid_preds)), valid_preds]
pd.DataFrame({"prediction": valid_preds, "probability": valid_pred_class_probs, "label": y_valid}).to_csv("logisticRegression_valid_preds.csv", index=False)

test_preds = best_model.predict(X_test)
test_probs = best_model.predict_proba(X_test)
test_pred_class_probs = test_probs[np.arange(len(test_preds)), test_preds]
pd.DataFrame({"prediction": test_preds, "probability": test_pred_class_probs, "label": y_test}).to_csv("logisticRegression_test_preds.csv", index=False)

dump(svd, "svd_LR.joblib")
dump(best_model, "final_LR.pkl")
dump(scaler, "scaler_LR")
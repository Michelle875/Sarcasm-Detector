import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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



solvers = ["liblinear", "lbfgs","saga"]
C_values = [0.1, 1.0, 3.0, 10.0]
penalties = ["l2","l1","elasticnet"]  # l1 only works with liblinear

best_acc = -1
best_model = None
best_params = None


for solver in solvers:
    for C in C_values:
        for penalty in penalties:

            # liblinear supports L1 and L2, lbfgs only L2
            if solver != "liblinear" and penalty == "l1":
                continue
            if solver != "saga" and penalty == "elasticnet":
                continue

            print(f"Trying: solver={solver}, C={C}, penalty={penalty}")
            if penalty != "elasticnet":
                model = LogisticRegression(
                    C=C,
                    solver=solver,
                    penalty=penalty,
                    max_iter=200,
                    random_state=42
                )
            else:
                model = LogisticRegression(
                    C=C,
                    solver=solver,
                    penalty=penalty,
                    max_iter=200,
                    random_state=42,
                    l1_ratio=0.5
                )

            model.fit(X_train, y_train)
            preds_valid = model.predict(X_valid)
            acc_valid = accuracy_score(y_valid, preds_valid)

            print(f" → Validation accuracy: {acc_valid:.4f}\n")

            if acc_valid > best_acc:
                best_acc = acc_valid
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

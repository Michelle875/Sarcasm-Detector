import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
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

kernels = ["linear", "rbf", "poly"]
C_values = [0.1, 1.0, 10.0, 100.0]
gamma_values = ["scale", "auto", 0.001, 0.01]
degree_values = [2, 3, 4]

best_acc = -1
best_model = None
best_params = None

for kernel in kernels:
    for C in C_values:
        if kernel == "linear":
            print(f"Trying: kernel={kernel}, C={C}")
            model = SVC(
                kernel=kernel,
                C=C,
                random_state=42,
                max_iter=5000
            )
            model.fit(X_train, y_train)
            acc = accuracy_score(y_valid, model.predict(X_valid))
            print(f" → Validation accuracy: {acc:.4f}\n")
            
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = {"kernel": kernel, "C": C}
                
        elif kernel == "rbf":
            for gamma in gamma_values:
                print(f"Trying: kernel={kernel}, C={C}, gamma={gamma}")
                model = SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    random_state=42,
                    max_iter=5000
                )
                model.fit(X_train, y_train)
                acc = accuracy_score(y_valid, model.predict(X_valid))
                print(f" → Validation accuracy: {acc:.4f}\n")
                
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = {"kernel": kernel, "C": C, "gamma": gamma}
                    
        elif kernel == "poly":
            for gamma in gamma_values:
                for degree in degree_values:
                    print(f"Trying: kernel={kernel}, C={C}, gamma={gamma}, degree={degree}")
                    model = SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        degree=degree,
                        random_state=42,
                        max_iter=5000
                    )
                    model.fit(X_train, y_train)
                    acc = accuracy_score(y_valid, model.predict(X_valid))
                    print(f" → Validation accuracy: {acc:.4f}\n")
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
                        best_params = {"kernel": kernel, "C": C, "gamma": gamma, "degree": degree}

print("\n=== BEST SVM MODEL ===")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"Validation Accuracy: {best_acc:.4f}")

test_preds = best_model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
print(f"\nFINAL TEST ACCURACY: {test_acc:.4f}")
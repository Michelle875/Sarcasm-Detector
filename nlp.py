import pandas as pd
import numpy as np
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.neural_network import MLPClassifier
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

numeric_cols = train_df.columns.difference(['text','tokens','pos_seq','label'])

X_train_extra = train_df[numeric_cols].astype(np.float64).values
X_valid_extra = valid_df[numeric_cols].astype(np.float64).values
X_test_extra  = test_df[numeric_cols].astype(np.float64).values

# Scale numeric features
scaler = StandardScaler()
X_train_extra = scaler.fit_transform(X_train_extra)
X_valid_extra = scaler.transform(X_valid_extra)
X_test_extra  = scaler.transform(X_test_extra)

# Combine sparse TF-IDF + dense numeric features
X_train = hstack([train_tfidf, csr_matrix(X_train_extra)])
X_valid = hstack([valid_tfidf, csr_matrix(X_valid_extra)])
X_test  = hstack([test_tfidf,  csr_matrix(X_test_extra)])

solvers = ['adam']
learning_rates = [0.001, 0.01, 0.1]
hidden_layers = [(256, 128)]
activations = ['relu', 'tanh', 'logistic']

best_score = -1
best_params = {}

for solver in solvers:
    for lr in learning_rates:
        for layers in hidden_layers:
            for act in activations:

                mlp = MLPClassifier(
                    hidden_layer_sizes=layers,
                    activation=act,
                    solver=solver,
                    learning_rate_init=lr,
                    max_iter=100,       
                    random_state=42,
                    verbose=False       
                )

                mlp.fit(X_train, y_train)
                score = accuracy_score(y_valid, mlp.predict(X_valid))

                print(f"Solver={solver}, LR={lr}, Layers={layers}, Act={act}, Val={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = {
                        "solver": solver,
                        "learning_rate": lr,
                        "layers": layers,
                        "activation": act
                    }

print("\nBest params:", best_params)
print("Best validation accuracy:", best_score)


final_mlp = MLPClassifier(
    hidden_layer_sizes=best_params["layers"],
    activation=best_params["activation"],
    solver=best_params["solver"],
    learning_rate_init=best_params["learning_rate"],
    max_iter=150,
    random_state=42,
    verbose=False
)

final_mlp.fit(X_train, y_train)


test_pred = final_mlp.predict(X_test)
print("\nTest accuracy:", accuracy_score(y_test, test_pred))

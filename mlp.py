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

solvers = 'adam'
learning_rates = 0.001
hidden_layers = (256, 128)
activations = 'relu'
final_mlp = MLPClassifier(
    hidden_layer_sizes=hidden_layers,
    activation=activations,
    solver=solvers,
    learning_rate_init=learning_rates,
    max_iter=150,
    random_state=42,
    verbose=False
)

final_mlp.fit(X_train, y_train)


test_pred = final_mlp.predict(X_test)
print("\nTest accuracy:", accuracy_score(y_test, test_pred))


# Predictions
train_pred = final_mlp.predict(X_train)
valid_pred = final_mlp.predict(X_valid)
test_pred  = final_mlp.predict(X_test)

# Probabilities
train_proba = final_mlp.predict_proba(X_train)
valid_proba = final_mlp.predict_proba(X_valid)
test_proba  = final_mlp.predict_proba(X_test)

# Probability of the predicted class
train_conf = train_proba[np.arange(len(train_pred)), train_pred]
valid_conf = valid_proba[np.arange(len(valid_pred)), valid_pred]
test_conf  = test_proba[np.arange(len(test_pred)),  test_pred]

pd.DataFrame({
    "prediction": train_pred,
    "probability": train_conf,
    "label": y_train
}).to_csv("mlp_train_predictions.csv", index=False)

pd.DataFrame({
   
    "prediction": valid_pred,
    "probability": valid_conf,
     "label": y_valid
}).to_csv("mlp_valid_predictions.csv", index=False)

pd.DataFrame({
    "prediction": test_pred,
    "probability": test_conf,
    "label": y_test,

}).to_csv("mlp_test_predictions.csv", index=False)



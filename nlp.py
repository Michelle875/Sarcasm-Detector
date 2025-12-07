import pandas as pd
import numpy as np
from scipy.sparse import load_npz, hstack
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import csr_matrix


train_tfidf = load_npz("train_tfidf.npz")
valid_tfidf = load_npz("valid_tfidf.npz")
test_tfidf  = load_npz("test_tfidf.npz")

train_df = pd.read_csv("train_features.csv")
valid_df = pd.read_csv("valid_features.csv")
test_df  = pd.read_csv("test_features.csv")

# First column is label
y_train = train_df.iloc[:, 0].values
y_valid = valid_df.iloc[:, 0].values
y_test  = test_df.iloc[:, 0].values

# Remaining columns are engineered numeric features
X_train_extra = train_df.iloc[:, 1:].values
X_valid_extra = valid_df.iloc[:, 1:].values
X_test_extra  = test_df.iloc[:, 1:].values

numeric_cols = train_df.columns.difference(['text', 'tokens','label', 'pos_seq'])

X_train_extra = train_df[numeric_cols].astype(np.float64).values
X_valid_extra = valid_df[numeric_cols].astype(np.float64).values
X_test_extra  = test_df[numeric_cols].astype(np.float64).values
    
X_train = hstack([train_tfidf, csr_matrix(X_train_extra)])
X_valid = hstack([valid_tfidf, csr_matrix(X_valid_extra)])
X_test  = hstack([test_tfidf,  csr_matrix(X_test_extra)])

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=20,
    random_state=42,
    verbose=True
)

mlp.fit(X_train, y_train)

valid_pred = mlp.predict(X_valid)
print("Validation Accuracy:", accuracy_score(y_valid, valid_pred))
print(classification_report(y_valid, valid_pred))

test_pred = mlp.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))
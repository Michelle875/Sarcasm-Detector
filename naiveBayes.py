import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load TF-IDF matrices
X_train = load_npz("train_tfidf.npz")
X_valid = load_npz("valid_tfidf.npz")
X_test  = load_npz("test_tfidf.npz")

# Load labels
train_df = pd.read_csv("train_features.csv")
valid_df = pd.read_csv("valid_features.csv")
test_df  = pd.read_csv("test_features.csv")

y_train = train_df["label"].astype(int).values
y_valid = valid_df["label"].astype(int).values
y_test  = test_df["label"].astype(int).values

# Hyperparameter search
alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_acc = -1
best_alpha = None
best_model = None

for alpha in alphas:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, valid_pred)

    print(f"alpha={alpha} → Validation Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_alpha = alpha
        best_model = model

print("\nBest alpha:", best_alpha)
print("Best validation accuracy:", best_acc)

# Final test evaluation
test_pred = best_model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))



train_preds = best_model.predict(X_train)
train_probs = best_model.predict_proba(X_train)
train_pred_class_probs = train_probs[np.arange(len(train_preds)), train_preds]

pd.DataFrame({"prediction": train_preds, "probability": train_pred_class_probs, "label": y_train}).to_csv("naiveBayes_train_preds.csv", index=False)

valid_preds = best_model.predict(X_valid)
valid_probs = best_model.predict_proba(X_valid)
valid_pred_class_probs = valid_probs[np.arange(len(valid_preds)), valid_preds]
pd.DataFrame({"prediction": valid_preds, "probability": valid_pred_class_probs, "label": y_valid}).to_csv("naiveBayes_valid_preds.csv", index=False)

test_preds = best_model.predict(X_test)
test_probs = best_model.predict_proba(X_test)
test_pred_class_probs = test_probs[np.arange(len(test_preds)), test_preds]
pd.DataFrame({"prediction": test_preds, "probability": test_pred_class_probs, "label": y_test}).to_csv("naiveBayes_test_preds.csv", index=False)



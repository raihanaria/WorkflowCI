```python id="x3r4x0"
# =========================================
# IMPORT LIBRARIES
# =========================================

import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from imblearn.over_sampling import SMOTE


# =========================================
# ENABLE AUTOLOG
# =========================================

mlflow.sklearn.autolog()


# =========================================
# LOAD DATASET
# =========================================

X_train = pd.read_csv(
    "dataset_preprocessing/X_train.csv"
)

X_test = pd.read_csv(
    "dataset_preprocessing/X_test.csv"
)

y_train = pd.read_csv(
    "dataset_preprocessing/y_train.csv"
)

y_test = pd.read_csv(
    "dataset_preprocessing/y_test.csv"
)

print("Dataset preprocessing berhasil dimuat")


# =========================================
# DISTRIBUSI CLASS SEBELUM SMOTE
# =========================================

print("\nDistribusi class sebelum SMOTE:")

print(
    y_train.value_counts()
)


# =========================================
# SMOTE
# =========================================

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train.values.ravel()
)

print("\nSMOTE berhasil dilakukan")


# =========================================
# DISTRIBUSI CLASS SETELAH SMOTE
# =========================================

print("\nDistribusi class setelah SMOTE:")

print(
    pd.Series(y_train_smote).value_counts()
)


# =========================================
# MODEL TRAINING
# =========================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(
    X_train_smote,
    y_train_smote
)

print("\nTraining model berhasil")


# =========================================
# PREDICTION
# =========================================

y_pred = model.predict(
    X_test
)


# =========================================
# EVALUATION
# =========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# =========================================
# PRINT METRICS
# =========================================

print("\n=== Evaluation Metrics ===")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\n=== Classification Report ===")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================
# LOG MANUAL METRICS
# =========================================

mlflow.log_metric(
    "accuracy",
    accuracy
)

mlflow.log_metric(
    "precision",
    precision
)

mlflow.log_metric(
    "recall",
    recall
)

mlflow.log_metric(
    "f1_score",
    f1
)


print("\nModel berhasil disimpan ke MLflow")
```

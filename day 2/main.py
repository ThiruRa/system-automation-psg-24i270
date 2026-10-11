
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
print("=" * 60)
print("MACHINE CONDITION PREDICTION - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("machine_sensor_dataset.csv")

# 2. Clean column names
df.columns = df.columns.str.strip().str.lower()

# 3. Remove unnecessary ID column
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 4. Check required columns
required_columns = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm",
    "state"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )

# 5. Clean data
df = df.dropna(subset=required_columns)

feature_columns = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm"
]

X = df[feature_columns]
y = df["state"].astype(str).str.strip()

print(f"\nTotal dataset records: {len(df)}")
print("\nClass distribution:")
print(y.value_counts())

# 6. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# 7. Create and train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining model...")
model.fit(X_train, y_train)

# 8. Predict test data
y_pred = model.predict(X_test)

# 9. Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# 10. Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# 11. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12. Cross-validation accuracy
print("\nCalculating 5-Fold Cross-Validation Accuracy...")

minimum_class_count = y.value_counts().min()
cv_folds = min(5, int(minimum_class_count))

if cv_folds >= 2:
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv_folds,
        scoring="accuracy"
    )

    print(
        f"Cross-Validation Accuracy: "
        f"{cv_scores.mean() * 100:.2f}%"
    )
    print(
        f"Cross-Validation Standard Deviation: "
        f"{cv_scores.std() * 100:.2f}%"
    )
else:
    print("Cross-validation skipped: insufficient samples per class.")

# 13. Save trained model
joblib.dump(model, "machine_condition_model.pkl")

print("\nModel saved as machine_condition_model.pkl")
print("\nTraining completed successfully!")
print("=" * 60)


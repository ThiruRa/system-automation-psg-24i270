
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load automation CSV
df = pd.read_csv("day1_sensor_data.csv")

df.columns = df.columns.str.strip().str.lower()

# 2. Select sensor features and target
features = [
    "temperature",
    "pressure",
    "vibration",
    "current"
]

target = "action"

# 3. Validate required columns
required = features + [target]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# 4. Clean data
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[target] = df[target].astype("string").str.strip().str.upper()

df = df.dropna(subset=required)
df = df[df[target] != ""]

X = df[features]
y = df[target]

print("=" * 50)
print("MACHINE AUTOMATION - ML TRAINING")
print("=" * 50)

print("\nTotal records:", len(df))
print("\nRecords per action:")
print(y.value_counts())

# 5. Check whether stratified evaluation is possible
class_counts = y.value_counts()

if len(class_counts) < 2:
    raise ValueError(
        "At least two different action classes are required."
    )

if class_counts.min() < 2:
    raise ValueError(
        "A class has fewer than 2 records. Run the automation "
        "more times to collect more examples."
    )

# Use a test set while ensuring each class has enough examples
n_classes = y.nunique()
test_count = max(round(len(df) * 0.20), n_classes)
train_count = len(df) - test_count

if train_count < n_classes:
    raise ValueError(
        "Not enough records for a stratified train/test split. "
        "Collect more sensor readings first."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_count,
    random_state=42,
    stratify=y
)

# 6. Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 7. Test accuracy
y_pred = model.predict(X_test)

print("\n========== TEST RESULTS ==========")
print(f"Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print("Class order:", list(model.classes_))

# 8. Cross-validation on training data
minimum_class_count = y_train.value_counts().min()
n_splits = min(5, int(minimum_class_count))

if n_splits >= 2:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print("\n========== CROSS-VALIDATION ==========")
    print("Fold accuracies (%):", (scores * 100).round(2))
    print(f"Mean CV accuracy: {scores.mean() * 100:.2f}%")
    print(f"CV standard deviation: {scores.std() * 100:.2f}%")
else:
    print("\nCross-validation skipped: not enough examples per class.")

# 9. Save trained model
joblib.dump(model, "machine_condition_model.pkl")

print("\nSaved: machine_condition_model.pkl")
print("Training completed!")


import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# CONFIGURATION
# ==========================================

CSV_FILE = "day1_sensor_data.csv"
MODEL_FILE = "machine_condition_model.pkl"

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "current"
]

TARGET = "action"


# ==========================================
# 1. LOGICAL RULE-BASED DECISION
# ==========================================

def logical_check(row):
    temp = float(row["temperature"])
    pressure = float(row["pressure"])
    vibration = float(row["vibration"])
    current = float(row["current"])

    # Rule 1: Critical machine failure
    if temp > 80 and vibration > 3.5:
        return "EMERGENCY_STOP"

    # Rule 2: Overheating and high pressure
    elif temp > 80 and pressure > 4:
        return "EMERGENCY_STOP"

    # Rule 3: Motor overload
    elif current > 13 and vibration > 3.5:
        return "REDUCE_LOAD"

    # Rule 4: Extremely high temperature
    elif temp > 90:
        return "EMERGENCY_STOP"

    # Rule 5: High vibration
    elif vibration > 4:
        return "STOP_MACHINE"

    # Rule 6: High current
    elif current > 13:
        return "REDUCE_LOAD"

    # Rule 7: High temperature
    elif temp > 80:
        return "COOL_MACHINE"

    # Rule 8: High pressure
    elif pressure > 4:
        return "CHECK_PRESSURE"

    # No abnormal condition detected by these rules
    else:
        return "CONTINUE_NORMAL"


# ==========================================
# 2. EXPLAIN THE LOGICAL DECISION
# ==========================================

def explain_reading(row):
    temp = float(row["temperature"])
    pressure = float(row["pressure"])
    vibration = float(row["vibration"])
    current = float(row["current"])

    if temp > 80 and vibration > 3.5:
        return "High temperature and high vibration detected."

    elif temp > 80 and pressure > 4:
        return "High temperature and high pressure detected."

    elif current > 13 and vibration > 3.5:
        return "High current and high vibration indicate possible motor overload."

    elif temp > 90:
        return "Extremely high temperature detected."

    elif vibration > 4:
        return "Excessive vibration detected."

    elif current > 13:
        return "Current exceeds the configured threshold."

    elif temp > 80:
        return "High temperature detected; cooling is required."

    elif pressure > 4:
        return "Pressure exceeds the configured threshold."

    else:
        return "No abnormal condition detected by the configured rules."


# ==========================================
# 3. LOAD DAY 1 DATA
# ==========================================

print("\n========== DAY 2: SENSOR DATA ANALYSIS ==========")

if not os.path.exists(CSV_FILE):
    print("ERROR: Day 1 CSV file not found.")
    print("Run history.py first.")
    raise SystemExit(1)

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("ERROR: Missing columns:", missing_columns)
    print("Available columns:", list(df.columns))
    raise SystemExit(1)

# Convert sensor values to numeric
for column in FEATURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df[TARGET] = df[TARGET].astype("string").str.strip()

# Remove invalid or missing rows
df = df.dropna(subset=FEATURES + [TARGET]).copy()
df = df[df[TARGET] != ""]

if len(df) < 5:
    print("ERROR: Not enough valid sensor readings.")
    print("Generate more data using history.py.")
    raise SystemExit(1)

print("Total valid readings:", len(df))


# ==========================================
# 4. COMPARE DAY 1 ACTIONS WITH LOGIC
# ==========================================

df["logical_action"] = df.apply(logical_check, axis=1)
df["recorded_action"] = df[TARGET].astype(str)

matches = df["logical_action"] == df["recorded_action"]

print("\n========== RULE-BASED COMPARISON ==========")

print("Total readings:", len(df))
print("Matching actions:", int(matches.sum()))
print("Mismatched actions:", int((~matches).sum()))

agreement = matches.mean() * 100
print(f"Logical agreement: {agreement:.2f}%")

if (~matches).any():
    print("\nFirst mismatched readings:")

    mismatch_columns = FEATURES + [
        "recorded_action",
        "logical_action"
    ]

    print(
        df.loc[~matches, mismatch_columns]
        .head(10)
        .to_string(index=False)
    )
else:
    print("All recorded actions match the logical rules.")


# ==========================================
# 5. TRAIN MACHINE LEARNING MODEL
# ==========================================

print("\n========== MACHINE LEARNING TRAINING ==========")

X = df[FEATURES]
y = df["recorded_action"]

model = None

# Ensure there are enough samples of every action
class_counts = y.value_counts()

if y.nunique() < 2:
    print("ML training skipped: only one action class exists.")

elif class_counts.min() < 2:
    print("ML training skipped: some actions have too few samples.")

else:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create the model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # Train using Day 1 recorded actions
    model.fit(X_train, y_train)

    # Evaluate on unseen test rows
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Training completed successfully.")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save the trained model
    joblib.dump(model, MODEL_FILE)

    print("Model saved as:", MODEL_FILE)


# ==========================================
# 6. COMPARE RULE AND ML FOR LATEST READING
# ==========================================

latest = df.iloc[-1]
latest_X = latest[FEATURES].to_frame().T

rule_action = logical_check(latest)
recorded_action = latest["recorded_action"]

print("\n========== LATEST SENSOR READING ==========")

print(f"Temperature: {latest['temperature']}")
print(f"Pressure:    {latest['pressure']}")
print(f"Vibration:   {latest['vibration']}")
print(f"Current:     {latest['current']}")

print("\n========== RULE vs ML COMPARISON ==========")

print("Day 1 recorded action:", recorded_action)
print("Rule-based action:    ", rule_action)

ml_action = None

if model is not None:
    ml_action = model.predict(latest_X)[0]
    print("ML predicted action:  ", ml_action)

    if rule_action == ml_action:
        print("Comparison: BOTH AGREE")
    else:
        print("Comparison: DIFFERENT PREDICTIONS")
        print("Priority: RULE-BASED LOGIC")

else:
    print("ML prediction unavailable.")
    print("Rule-based logic will still decide the final action.")


# ==========================================
# 7. FINAL DECISION: RULE HAS PRIORITY
# ==========================================

print("\n========== FINAL DECISION ==========")

# IMPORTANT: ML never overrides the rule-based result.
final_action = rule_action

print("Final action:     ", final_action)
print("Decision source:  RULE-BASED LOGIC")
print("Reason:           ", explain_reading(latest))

if ml_action is not None:
    if ml_action != rule_action:
        print("ML disagreed with the rule.")
        print("The rule-based decision was followed.")
    else:
        print("ML agrees with the rule-based decision.")

print("\n========== ANALYSIS COMPLETED ==========")

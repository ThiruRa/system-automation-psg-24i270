
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

CSV_FILE = "day1_sensor_data.csv"

# 1. Read Day 1 sensor data
if not os.path.exists(CSV_FILE):
    print("ERROR: Day 1 CSV file not found!")
    print("Run history.py first.")
    raise SystemExit(1)

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

required = [
    "temperature", "pressure", "vibration",
    "current", "action"
]

if not all(col in df.columns for col in required):
    print("ERROR: CSV is missing required columns.")
    print("Available columns:", list(df.columns))
    raise SystemExit(1)

df = df.dropna(subset=required).copy()

if len(df) < 5:
    print("Not enough sensor readings. Run Day 1 again.")
    raise SystemExit(1)

# 2. Logical comparison using the same priority as Day 1
def logical_check(row):
    temp = float(row["temperature"])
    pressure = float(row["pressure"])
    vibration = float(row["vibration"])
    current = float(row["current"])

    if temp > 80 and vibration > 3.5:
        return "EMERGENCY_STOP"

    elif temp > 80 and pressure > 4:
        return "EMERGENCY_STOP"

    elif current > 13 and vibration > 3.5:
        return "REDUCE_LOAD"

    elif temp > 90:
        return "EMERGENCY_STOP"

    elif vibration > 4:
        return "STOP_MACHINE"

    elif current > 13:
        return "REDUCE_LOAD"

    elif temp > 80:
        return "COOL_MACHINE"

    elif pressure > 4:
        return "CHECK_PRESSURE"

    else:
        return "CONTINUE_NORMAL"


def explain_reading(row, action):
    temp = float(row["temperature"])
    pressure = float(row["pressure"])
    vibration = float(row["vibration"])
    current = float(row["current"])

    if temp > 80 and vibration > 3.5:
        return "High temperature + high vibration: critical machine risk."

    elif temp > 80 and pressure > 4:
        return "High temperature + high pressure: emergency risk."

    elif current > 13 and vibration > 3.5:
        return "High current + high vibration: possible motor overload."

    elif temp > 90:
        return "Temperature is extremely high."

    elif vibration > 4:
        return "Excessive vibration detected."

    elif current > 13:
        return "Current is above the safe threshold."

    elif temp > 80:
        return "High temperature detected; cooling is required."

    elif pressure > 4:
        return "Pressure is above the set threshold."

    else:
        return "All readings are within the configured rule thresholds."


# 3. Compare the logical result with Day 1's action
df["logical_action"] = df.apply(logical_check, axis=1)
df["recorded_action"] = df["action"].astype(str).str.strip()

matches = df["logical_action"] == df["recorded_action"]
print("\n========== DAY 1 vs LOGICAL CHECK ==========")
print("Total readings:", len(df))
print("Matching actions:", int(matches.sum()))
print("Mismatched actions:", int((~matches).sum()))

if len(df) > 0:
    print("Logical agreement: {:.2f}%".format(matches.mean() * 100))

if (~matches).any():
    print("\nFirst mismatched readings:")
    print(
        df.loc[~matches, [
            "temperature", "pressure", "vibration",
            "current", "recorded_action", "logical_action"
        ]].head(10).to_string(index=False)
    )

# 4. Train ML model to predict the recorded action
features = ["temperature", "pressure", "vibration", "current"]
X = df[features].astype(float)
y = df["recorded_action"]

model = None

if y.nunique() >= 2 and y.value_counts().min() >= 2:
    # Stratification keeps examples of each action in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    print("\n========== MACHINE LEARNING RESULTS ==========")
    print("Test accuracy: {:.2f}%".format(
        accuracy_score(y_test, test_predictions) * 100
    ))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0
    ))

    joblib.dump(model, "machine_condition_model.pkl")
    print("Model saved: machine_condition_model.pkl")

else:
    print("\nNot enough examples per action to evaluate ML reliably.")
    print("Run Day 1 again to generate more varied readings.")

# 5. Compare the latest sensor reading
latest = df.iloc[-1]
latest_X = latest[features].to_frame().T.astype(float)

logical_action = logical_check(latest)
recorded_action = str(latest["recorded_action"])

print("\n========== LATEST SENSOR READING ==========")
print(f"Temperature : {latest['temperature']}")
print(f"Pressure    : {latest['pressure']}")
print(f"Vibration   : {latest['vibration']}")
print(f"Current     : {latest['current']}")

print("\n========== LOGICAL ANALYSIS ==========")
print("Reason:", explain_reading(latest, logical_action))
print("Rule-based action:", logical_action)
print("Day 1 recorded action:", recorded_action)

if logical_action == recorded_action:
    print("Comparison: MATCH")
else:
    print("Comparison: MISMATCH - check Day 1 and Day 2 rules.")

if model is not None:
    ml_action = model.predict(latest_X)[0]
    print("ML predicted action:", ml_action)

    if ml_action == logical_action:
        print("ML vs Logic: MATCH")
    else:
        print("ML vs Logic: MISMATCH")

    if ml_action == recorded_action:
        print("ML vs Day 1: MATCH")
    else:
        print("ML vs Day 1: MISMATCH")
else:
    print("ML prediction unavailable; insufficient action classes.")

print("\n========== FINAL DECISION ==========")
print("Recommended action:", logical_action)
print("Reason:", explain_reading(latest, logical_action))


"""
Robotic Arm Joint Health Monitoring System
Bonus Challenge - System Automation

Features:
1. Immediate rule-based safety stop
2. ML-based predictive wear detection
3. Maintenance log analysis
4. Spoofed-sensor detection
5. One final decision per arm
6. Save the complete report to output.txt
"""

from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# 1. FILES AND THRESHOLDS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "historical_joint_data.csv"
LOG_FILE = BASE_DIR / "maintenance_log.txt"
OUTPUT_FILE = BASE_DIR / "output.txt"

FEATURES = ["torque_nm", "temperature_c", "vibration_g"]

# Illustrative educational thresholds, not certified robot limits.
EMERGENCY_LIMITS = {
    "torque_nm": 95.0,
    "temperature_c": 90.0,
    "vibration_g": 4.5
}

WEAR_KEYWORDS = [
    "vibration increase",
    "vibration is increasing",
    "increasing vibration",
    "overheating",
    "hot",
    "grinding",
    "bearing wear",
    "unusual noise",
    "wear"
]

SENSOR_KEYWORDS = [
    "sensor looks stuck",
    "sensor reading looks stuck",
    "sensor mismatch",
    "sensor spoof"
]


# --------------------------------------------------
# 2. SIMULATED SENSOR READINGS
# --------------------------------------------------

CURRENT_READINGS = [
    {
        "arm_id": "ARM-01", "joint": "shoulder",
        "torque_nm": 32, "temperature_c": 38,
        "vibration_g": 0.4, "reference_torque_nm": 33
    },
    {
        "arm_id": "ARM-02", "joint": "elbow",
        "torque_nm": 55, "temperature_c": 61,
        "vibration_g": 2.0, "reference_torque_nm": 56
    },
    {
        "arm_id": "ARM-03", "joint": "elbow",
        "torque_nm": 72, "temperature_c": 94,
        "vibration_g": 3.1, "reference_torque_nm": 73
    },
    {
        "arm_id": "ARM-04", "joint": "wrist",
        "torque_nm": 40, "temperature_c": 48,
        "vibration_g": 0.7, "reference_torque_nm": 41
    },
    {
        "arm_id": "ARM-05", "joint": "elbow",
        "torque_nm": 45, "temperature_c": 52,
        "vibration_g": 0.8, "reference_torque_nm": 78
    },
    {
        "arm_id": "ARM-06", "joint": "wrist",
        "torque_nm": 28, "temperature_c": 42,
        "vibration_g": 0.9, "reference_torque_nm": 29
    }
]


# --------------------------------------------------
# 3. TRAIN THE ML MODEL
# --------------------------------------------------

def train_model():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Run python generate_dataset.py first."
        )

    data = pd.read_csv(DATA_FILE)

    for column in FEATURES + ["wear_label"]:
        if column not in data.columns:
            raise ValueError(f"Missing dataset column: {column}")

    if len(data) < 50:
        raise ValueError("At least 50 labeled historical readings are required.")

    X = data[FEATURES]
    y = data["wear_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            class_weight="balanced"
        )
    )

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    return model, accuracy, len(data)


# --------------------------------------------------
# 4. READ AND ANALYZE MAINTENANCE NOTES
# --------------------------------------------------

def read_log():

    if not LOG_FILE.exists():
        print("WARNING: maintenance_log.txt not found.")
        return []

    return LOG_FILE.read_text(encoding="utf-8").splitlines()


def analyze_notes(arm_id, lines):

    matching = [
        line for line in lines
        if arm_id.lower() in line.lower()
    ]

    text = " ".join(matching).lower()

    wear_warning = any(
        keyword in text for keyword in WEAR_KEYWORDS
    )

    sensor_warning = any(
        keyword in text for keyword in SENSOR_KEYWORDS
    )

    return wear_warning, sensor_warning, matching


# --------------------------------------------------
# 5. IMMEDIATE SAFETY RULES
# --------------------------------------------------

def check_emergency(reading):

    crossed = []

    for sensor, limit in EMERGENCY_LIMITS.items():

        if float(reading[sensor]) >= limit:
            crossed.append(
                f"{sensor}={reading[sensor]} "
                f"(threshold={limit})"
            )

    return crossed


# --------------------------------------------------
# 6. SPOOFED-SENSOR CHECK
# --------------------------------------------------

def check_spoof(reading):

    primary = float(reading["torque_nm"])
    reference = float(reading["reference_torque_nm"])

    difference = abs(primary - reference)

    if difference > 20.0:
        return True, (
            f"Primary/reference torque mismatch: "
            f"{difference:.1f} Nm"
        )

    return False, "No major torque mismatch"


# --------------------------------------------------
# 7. DECISION ENGINE
# --------------------------------------------------

def decide(reading, probability, wear_warning, sensor_warning):

    # Priority 1: deterministic safety rule.
    # This check does not depend on the ML prediction.

    crossed = check_emergency(reading)

    if crossed:
        return (
            "EMERGENCY_STOP",
            "Safety threshold crossed: " + ", ".join(crossed)
        )

    # A large independent-reference discrepancy triggers a stop.

    spoofed, spoof_reason = check_spoof(reading)

    if spoofed:
        return (
            "EMERGENCY_STOP",
            "Possible sensor spoof/anomaly: " + spoof_reason
        )

    # Priority 2: schedule maintenance.

    if probability >= 0.70 and wear_warning:
        return (
            "SCHEDULE_MAINTENANCE",
            "High predicted wear plus maintenance warning"
        )

    if probability >= 0.85:
        return (
            "SCHEDULE_MAINTENANCE",
            "Very high predicted wear probability"
        )

    # Priority 3: monitor developing problems.

    if wear_warning or sensor_warning or probability >= 0.40:
        return (
            "MONITOR",
            "Potential developing issue requires observation"
        )

    # Priority 4: normal readings and no warning.

    return (
        "SAFE",
        "No emergency or significant warning detected"
    )


# --------------------------------------------------
# 8. RUN MONITORING
# --------------------------------------------------

def main():

    model, accuracy, row_count = train_model()
    log_lines = read_log()

    print("=" * 65)
    print("ROBOTIC ARM JOINT HEALTH MONITORING SYSTEM")
    print("=" * 65)
    print(f"Historical readings      : {row_count}")
    print(f"Synthetic test accuracy : {accuracy:.3f}")
    print("Safety rules independent of ML: YES")
    print("-" * 65)

    results = {}

    priority = {
        "SAFE": 0,
        "MONITOR": 1,
        "SCHEDULE_MAINTENANCE": 2,
        "EMERGENCY_STOP": 3
    }

    for reading in CURRENT_READINGS:

        arm_id = reading["arm_id"]

        features = pd.DataFrame([{
            feature: reading[feature]
            for feature in FEATURES
        }])

        probability = float(
            model.predict_proba(features)[0][1]
        )

        wear_warning, sensor_warning, matching = analyze_notes(
            arm_id, log_lines
        )

        decision, reason = decide(
            reading,
            probability,
            wear_warning,
            sensor_warning
        )

        spoofed, _ = check_spoof(reading)

        result = {
            "decision": decision,
            "joint": reading["joint"],
            "probability": probability,
            "wear_warning": wear_warning,
            "sensor_warning": sensor_warning,
            "spoofed": spoofed,
            "reason": reason,
            "notes": matching
        }

        previous = results.get(arm_id)

        # Keep the most severe decision if an arm has multiple joints.
        if (
            previous is None
            or priority[decision] > priority[previous["decision"]]
        ):
            results[arm_id] = result

    # --------------------------------------------------
    # 9. PRINT EACH ARM'S RESULT
    # --------------------------------------------------

    for arm_id in sorted(results):

        result = results[arm_id]

        print(f"\n{arm_id} | Joint: {result['joint']}")
        print(f"  ML wear probability : {result['probability']:.2f}")
        print(f"  Maintenance warning : {result['wear_warning']}")
        print(f"  Sensor-note warning : {result['sensor_warning']}")
        print(f"  Spoof detected      : {result['spoofed']}")
        print(f"  FINAL DECISION      : {result['decision']}")
        print(f"  Reason              : {result['reason']}")

        for note in result["notes"]:
            print(f"  Log                 : {note}")

    # --------------------------------------------------
    # 10. CHECK FOUR REQUIRED DECISIONS
    # --------------------------------------------------

    reached = {
        result["decision"]
        for result in results.values()
    }

    print("\n" + "=" * 65)
    print("DECISION COVERAGE")
    print("=" * 65)

    for decision in [
        "SAFE",
        "MONITOR",
        "SCHEDULE_MAINTENANCE",
        "EMERGENCY_STOP"
    ]:

        status = "REACHED" if decision in reached else "NOT REACHED"
        print(f"{decision:25s}: {status}")

    print("=" * 65)


# --------------------------------------------------
# 11. PRINT AND SAVE OUTPUT
# --------------------------------------------------

if __name__ == "__main__":

    captured = StringIO()

    try:
        # Capture the report generated by main().
        with redirect_stdout(captured):
            main()

        report = captured.getvalue()

        # Display the report in the VS Code terminal.
        print(report, end="")

        # Save the same report into output.txt.
        OUTPUT_FILE.write_text(report, encoding="utf-8")

        print(f"\nOutput saved successfully to: {OUTPUT_FILE}")

    except (FileNotFoundError, ValueError) as error:

        error_message = f"ERROR: {error}\n"
        print(error_message, end="", file=sys.stderr)

        # Save the error too, so the failure is recorded.
        OUTPUT_FILE.write_text(error_message, encoding="utf-8")

        raise SystemExit(1)

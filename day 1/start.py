
import time
import random
import csv
import os

# File names
CSV_FILE = "day1_sensor_data.csv"
LOG_FILE = "day1_log.txt"


# ============================================================
# STEP 1: GENERATE SENSOR DATA
# ============================================================
def generate_sensor_data():
    return {
        "temperature": random.randint(50, 95),
        "pressure": round(random.uniform(1.0, 5.0), 2),
        "vibration": round(random.uniform(0.1, 5.0), 2),
        "current": round(random.uniform(5.0, 15.0), 2)
    }


# ============================================================
# STEP 2: DEFINE AUTOMATION RULES
# ============================================================
RULES = [
    {
        "name": "CRITICAL_MACHINE_FAILURE",
        "priority": 1,
        "condition": lambda r: (
            r["temperature"] > 80 and r["vibration"] > 3.5
        ),
        "action": "EMERGENCY_STOP"
    },
    {
        "name": "OVERHEATING_PRESSURE_RISK",
        "priority": 2,
        "condition": lambda r: (
            r["temperature"] > 80 and r["pressure"] > 4.0
        ),
        "action": "EMERGENCY_STOP"
    },
    {
        "name": "MOTOR_OVERLOAD",
        "priority": 3,
        "condition": lambda r: (
            r["current"] > 13 and r["vibration"] > 3.5
        ),
        "action": "REDUCE_LOAD"
    },
    {
        "name": "EMERGENCY_HIGH_TEMP",
        "priority": 4,
        "condition": lambda r: r["temperature"] > 90,
        "action": "EMERGENCY_STOP"
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 5,
        "condition": lambda r: r["vibration"] > 4.0,
        "action": "STOP_MACHINE"
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 6,
        "condition": lambda r: r["current"] > 13,
        "action": "REDUCE_LOAD"
    },
    {
        "name": "HIGH_TEMPERATURE",
        "priority": 7,
        "condition": lambda r: r["temperature"] > 80,
        "action": "COOL_MACHINE"
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 8,
        "condition": lambda r: r["pressure"] > 4.0,
        "action": "CHECK_PRESSURE"
    }
]


# ============================================================
# STEP 3: EVALUATE AUTOMATION RULES
# ============================================================
def evaluate_rules(reading, rules):
    sorted_rules = sorted(
        rules,
        key=lambda rule: rule["priority"]
    )

    for rule in sorted_rules:
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# ============================================================
# STEP 4: SELECT ACTION MESSAGE
# ============================================================
def take_action(action):
    actions = {
        "EMERGENCY_STOP": "Emergency stop activated!",
        "STOP_MACHINE": "Machine stopped due to unsafe condition.",
        "REDUCE_LOAD": "Machine load reduced.",
        "COOL_MACHINE": "Cooling system activated.",
        "CHECK_PRESSURE": "Pressure is high. Inspection required.",
        "CONTINUE_NORMAL": "Machine operating normally."
    }

    return actions.get(action, "Unknown action")


# ============================================================
# STEP 5: SELF-RECOVERY CHECK
# ============================================================
def self_recovery(reading, action):
    if action in ["EMERGENCY_STOP", "STOP_MACHINE"]:
        if (
            reading["temperature"] < 70
            and reading["pressure"] < 3.5
            and reading["vibration"] < 2.5
            and reading["current"] < 10
        ):
            return "MACHINE_RECOVERED - Restart condition satisfied"

        return "MACHINE_STILL_UNSAFE - Keep machine stopped"

    return "NO_RECOVERY_NEEDED"


# ============================================================
# STEP 6: SAVE SENSOR DATA TO CSV
# ============================================================
def save_to_csv(tick, reading, rule_name, action, outcome, recovery):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "tick",
                "temperature",
                "pressure",
                "vibration",
                "current",
                "rule",
                "action",
                "outcome",
                "recovery"
            ]
        )

        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()

        writer.writerow({
            "tick": tick,
            "temperature": reading["temperature"],
            "pressure": reading["pressure"],
            "vibration": reading["vibration"],
            "current": reading["current"],
            "rule": rule_name,
            "action": action,
            "outcome": outcome,
            "recovery": recovery
        })


# ============================================================
# STEP 7: SAVE DECISIONS TO TEXT LOG
# ============================================================
def log_decision(tick, reading, rule_name, action, recovery):
    with open(LOG_FILE, "a") as file:
        file.write(
            f"Tick: {tick} | "
            f"Reading: {reading} | "
            f"Rule: {rule_name} | "
            f"Action: {action} | "
            f"Recovery: {recovery}\n"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    print("=" * 60)
    print("DAY 1: MACHINE AUTOMATION AND DATA COLLECTION")
    print("=" * 60)

    # Start fresh files for this run
    for filename in [CSV_FILE, LOG_FILE]:
        if os.path.exists(filename):
            os.remove(filename)

    total_readings = 200

    for tick in range(1, total_readings + 1):

        # Sense
        reading = generate_sensor_data()

        # Think
        rule_name, action = evaluate_rules(reading, RULES)

        # Act
        outcome = take_action(action)

        # Recovery check
        recovery = self_recovery(reading, action)

        # Display output
        print(f"\n[Tick {tick:03d}] SENSE -> {reading}")
        print(f"             THINK -> {rule_name}")
        print(f"             ACT   -> {outcome}")
        print(f"             RECOVERY -> {recovery}")

        # Save all information
        save_to_csv(
            tick,
            reading,
            rule_name,
            action,
            outcome,
            recovery
        )

        log_decision(
            tick,
            reading,
            rule_name,
            action,
            recovery
        )

        time.sleep(0.02)

    print("\n" + "=" * 60)
    print("DAY 1 DATA COLLECTION COMPLETED")
    print(f"Total readings saved: {total_readings}")
    print(f"CSV file: {CSV_FILE}")
    print(f"Log file: {LOG_FILE}")
    print("Ready for Day 2 machine learning.")
    print("=" * 60)


if __name__ == "__main__":
    main()

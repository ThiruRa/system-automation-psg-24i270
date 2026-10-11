
import time
import random
import csv
import os

LOG_FILE = "day1_log.txt"
CSV_FILE = "day1_sensor_data.csv"

# STEP 1: Generate sensor data
def generate_sensor_data():
    return {
        "temperature": random.randint(50, 95),
        "pressure": round(random.uniform(1.0, 5.0), 2),
        "vibration": round(random.uniform(0.1, 5.0), 2),
        "current": round(random.uniform(5.0, 15.0), 2)
    }


# STEP 2: Automation rules
RULES = [
    {
        "name": "CRITICAL_MACHINE_FAILURE",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 80
        and r["vibration"] > 3.5,
        "action": "EMERGENCY_STOP"
    },
    {
        "name": "OVERHEATING_PRESSURE_RISK",
        "priority": 2,
        "condition": lambda r: r["temperature"] > 80
        and r["pressure"] > 4.0,
        "action": "EMERGENCY_STOP"
    },
    {
        "name": "MOTOR_OVERLOAD",
        "priority": 3,
        "condition": lambda r: r["current"] > 13
        and r["vibration"] > 3.5,
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


# STEP 3: Evaluate rules
def evaluate_rules(reading, rules):
    sorted_rules = sorted(rules, key=lambda r: r["priority"])

    for rule in sorted_rules:
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# STEP 4: Perform action
def take_action(action):
    actions = {
        "EMERGENCY_STOP": "Emergency stop activated!",
        "STOP_MACHINE": "Machine stopped due to unsafe condition.",
        "REDUCE_LOAD": "Machine load reduced.",
        "COOL_MACHINE": "Cooling system activated.",
        "CHECK_PRESSURE": "Pressure is high. Inspection required.",
        "CONTINUE_NORMAL": "Machine operating normally."
    }

    return actions.get(action, f"No handler for action: {action}")


# STEP 5: Self-recovery
def self_recovery(reading, action):
    if action in ["EMERGENCY_STOP", "STOP_MACHINE"]:
        if (
            reading["temperature"] < 70
            and reading["pressure"] < 3.5
            and reading["vibration"] < 2.5
            and reading["current"] < 10
        ):
            return "MACHINE_RECOVERED - Restarting automatically"
        else:
            return "MACHINE_STILL_UNSAFE - Keep machine stopped"

    return "NO_RECOVERY_NEEDED"


# STEP 6: Save data to CSV
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


# STEP 7: Save decision to text log
def log_decision(reading, rule_name, action, recovery):
    with open(LOG_FILE, "a") as file:
        file.write(
            f"Reading: {reading} | "
            f"Rule: {rule_name} | "
            f"Action: {action} | "
            f"Recovery: {recovery}\n"
        )


# MAIN PROGRAM
def main():
    print("=" * 60)
    print("MACHINE AUTOMATION SYSTEM")
    print("=" * 60)

    for tick in range(1, 11):
        reading = generate_sensor_data()

        rule_name, action = evaluate_rules(reading, RULES)
        outcome = take_action(action)
        recovery = self_recovery(reading, action)

        print(f"\n[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}")
        print(f"           RECOVERY -> {recovery}")

        save_to_csv(
            tick, reading, rule_name, action, outcome, recovery
        )

        log_decision(reading, rule_name, action, recovery)

        time.sleep(0.2)

    print("\n" + "=" * 60)
    print(f"Sensor data saved to: {CSV_FILE}")
    print(f"Text log saved to: {LOG_FILE}")
    print("Automation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

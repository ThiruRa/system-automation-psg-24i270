
import time
import random
import os
import psycopg

LOG_FILE = "day1_log.txt"

# PostgreSQL connection settings
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": os.environ.get("PGPASSWORD", "p")
}

MACHINES = [
    "CNC Machine",
    "Conveyor Motor",
    "Hydraulic Press",
    "Air Compressor",
    "Industrial Pump"
]

RULES = [
    {
        "name": "EMERGENCY_HIGH_TEMP",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 90,
        "action": "EMERGENCY_STOP"
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 4.0,
        "action": "STOP_MACHINE"
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 3,
        "condition": lambda r: r["current"] > 13,
        "action": "REDUCE_LOAD"
    },
    {
        "name": "HIGH_TEMPERATURE",
        "priority": 4,
        "condition": lambda r: r["temperature"] > 80,
        "action": "COOL_MACHINE"
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 5,
        "condition": lambda r: r["pressure"] > 4.0,
        "action": "CHECK_PRESSURE"
    }
]

RISK_SCORES = {
    "EMERGENCY_HIGH_TEMP": 100,
    "HIGH_VIBRATION": 80,
    "HIGH_CURRENT": 60,
    "HIGH_TEMPERATURE": 50,
    "HIGH_PRESSURE": 40,
    "NO_RULE_FIRED": 0
}


def generate_sensor_data():
    """Generate simulated sensor readings."""
    return {
        "temperature": random.randint(50, 95),
        "pressure": round(random.uniform(1.0, 5.0), 2),
        "vibration": round(random.uniform(0.1, 5.0), 2),
        "current": round(random.uniform(5.0, 15.0), 2)
    }


def evaluate_rules(reading, rules):
    """Return the first matching rule and action."""
    for rule in sorted(rules, key=lambda r: r["priority"]):
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


def take_action(action):
    """Describe the simulated action."""
    actions = {
        "EMERGENCY_STOP": "Emergency stop activated!",
        "STOP_MACHINE": "Machine stopped due to unsafe condition.",
        "REDUCE_LOAD": "Machine load reduced.",
        "COOL_MACHINE": "Cooling system activated.",
        "CHECK_PRESSURE": "Pressure is high. Inspection required.",
        "CONTINUE_NORMAL": "Machine operating normally."
    }
    return actions.get(action, f"No handler for action: {action}")


def calculate_risk(rule_name):
    return RISK_SCORES.get(rule_name, 0)


def find_priority_machine(machine_data):
    return max(machine_data, key=lambda x: x["risk"])


def log_decision(machine, reading, rule_name, action):
    """Append the decision to the existing text log."""
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"Machine: {machine} | "
            f"Reading: {reading} | "
            f"Rule: {rule_name} | "
            f"Action: {action}\n"
        )


def save_to_database(conn, tick, item):
    """Save one machine decision to PostgreSQL."""
    rule_priority = next(
        (
            rule["priority"]
            for rule in RULES
            if rule["name"] == item["rule"]
        ),
        None
    )

    reading = item["reading"]

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO machine_automation_logs (
                tick_number,
                machine_name,
                temperature,
                pressure,
                vibration,
                current_value,
                rule_name,
                rule_priority,
                action,
                outcome,
                risk_score
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                tick,
                item["machine"],
                reading["temperature"],
                reading["pressure"],
                reading["vibration"],
                reading["current"],
                item["rule"],
                rule_priority,
                item["action"],
                item["outcome"],
                item["risk"]
            )
        )


def main():
    if DB_CONFIG["password"] == "YOUR_POSTGRES_PASSWORD":
        raise ValueError(
            "Set your PostgreSQL password in the PGPASSWORD "
            "environment variable or update DB_CONFIG."
        )

    saved_count = 0

    # Keep the database connection open during all 10 ticks.
    with psycopg.connect(**DB_CONFIG) as conn:
        print("Connected to PostgreSQL successfully!")

        for tick in range(1, 11):
            machine_data = []

            print("\n" + "=" * 75)
            print(f"FACTORY MONITORING - TICK {tick:02d}")
            print("=" * 75)

            for machine in MACHINES:
                reading = generate_sensor_data()
                rule_name, action = evaluate_rules(reading, RULES)
                outcome = take_action(action)
                risk = calculate_risk(rule_name)

                item = {
                    "machine": machine,
                    "reading": reading,
                    "rule": rule_name,
                    "action": action,
                    "outcome": outcome,
                    "risk": risk
                }

                machine_data.append(item)

                print(f"\nMachine : {machine}")
                print(f"SENSE   : {reading}")
                print(f"THINK   : {rule_name}")
                print(f"RISK    : {risk}/100")
                print(f"ACT     : {outcome}")

                # Save to the existing text file.
                log_decision(machine, reading, rule_name, action)

                # Save the same decision to PostgreSQL.
                save_to_database(conn, tick, item)
                saved_count += 1

            # Commit this tick's five database records.
            conn.commit()
            print("\nDatabase: 5 machine records saved.")

            priority_machine = find_priority_machine(machine_data)

            print("\n" + "-" * 75)
            print("FACTORY PRIORITY ANALYSIS")
            print("-" * 75)

            if priority_machine["risk"] >= 80:
                print("!!! IMMEDIATE ACTION REQUIRED !!!")
                print(f"Machine : {priority_machine['machine']}")
                print(f"Risk    : {priority_machine['risk']}/100")
                print(f"Problem : {priority_machine['rule']}")
                print(f"Action  : {priority_machine['action']}")

            elif priority_machine["risk"] > 0:
                print("ATTENTION REQUIRED")
                print(f"Machine : {priority_machine['machine']}")
                print(f"Risk    : {priority_machine['risk']}/100")

            else:
                print("All machines are operating normally.")

            time.sleep(0.5)

    print("\n" + "=" * 75)
    print("SIMULATION COMPLETED")
    print(f"Total machine records saved: {saved_count}")
    print("PostgreSQL: Records committed successfully.")
    print(f"Text log: {LOG_FILE}")
    print("=" * 75)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nPROGRAM ERROR:", error)
        print("Check your database settings and table creation.")

"""
Generate synthetic historical data for the Robotic Arm Joint Health Monitoring System.
The dataset is educational/synthetic and is not suitable for real robot safety certification.
"""
from pathlib import Path
import random
import pandas as pd

SEED = 42
ROWS = 300
OUT = Path(__file__).with_name("historical_joint_data.csv")

def main():
    rng = random.Random(SEED)
    rows = []

    # Generate balanced-ish synthetic examples for healthy, worn, and severe joint conditions.
    for i in range(ROWS):
        scenario = i % 3

        if scenario == 0:  # healthy
            torque = rng.uniform(10, 45)
            temperature = rng.uniform(25, 55)
            vibration = rng.uniform(0.05, 1.4)
            wear = 0
        elif scenario == 1:  # early wear / degradation
            torque = rng.uniform(35, 75)
            temperature = rng.uniform(40, 75)
            vibration = rng.uniform(0.8, 3.0)
            wear = 1
        else:  # severe wear / fault history
            torque = rng.uniform(60, 100)
            temperature = rng.uniform(60, 95)
            vibration = rng.uniform(2.0, 5.0)
            wear = 1

        rows.append({
            "reading_id": i + 1,
            "arm_id": f"ARM-{(i % 6) + 1:02d}",
            "joint": ["shoulder", "elbow", "wrist"][i % 3],
            "torque_nm": round(torque, 2),
            "temperature_c": round(temperature, 2),
            "vibration_g": round(vibration, 3),
            "wear_label": wear,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Dataset created: {OUT.name}")
    print(f"Rows: {len(df)}")
    print("Label counts (0 = healthy, 1 = wear/fault history):")
    print(df["wear_label"].value_counts().sort_index().to_string())

if __name__ == "__main__":
    main()

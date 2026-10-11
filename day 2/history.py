
import numpy as np
import pandas as pd

# ======================================
# 1. SET PARAMETERS
# ======================================
n_rows = 250
np.random.seed(42)

# ======================================
# 2. GENERATE SENSOR DATA
# ======================================
df = pd.DataFrame({
    "temperature": np.round(
        np.random.uniform(55, 98, n_rows), 2
    ),
    "pressure": np.round(
        np.random.uniform(1.0, 6.5, n_rows), 2
    ),
    "vibration": np.round(
        np.random.uniform(1.0, 3.5, n_rows), 2
    ),
    "current": np.round(
        np.random.uniform(8.0, 16.0, n_rows), 2
    ),
    "rpm": np.random.randint(1400, 2201, n_rows)
})

# ======================================
# 3. ADD ABNORMAL SENSOR READINGS
# ======================================

# High temperature examples
mask = np.arange(n_rows) % 10 == 1
df.loc[mask, "temperature"] = np.round(
    np.random.uniform(80, 98, mask.sum()), 2
)

# High vibration examples
mask = np.arange(n_rows) % 10 == 2
df.loc[mask, "vibration"] = np.round(
    np.random.uniform(5.5, 9.8, mask.sum()), 2
)

# High motor current examples
mask = np.arange(n_rows) % 10 == 3
df.loc[mask, "current"] = np.round(
    np.random.uniform(20, 29, mask.sum()), 2
)

# Low pressure examples
mask = np.arange(n_rows) % 10 == 4
df.loc[mask, "pressure"] = np.round(
    np.random.uniform(1.0, 2.9, mask.sum()), 2
)

# ======================================
# 4. CLASSIFY MACHINE CONDITION
# ======================================
conditions = [
    df["temperature"] >= 80,
    df["vibration"] >= 5.5,
    df["current"] >= 20,
    df["pressure"] < 3.0
]

labels = [
    "OVERHEATING",
    "HIGH_VIBRATION",
    "MOTOR_OVERLOAD",
    "LOW_PRESSURE"
]

df["state"] = np.select(
    conditions,
    labels,
    default="NORMAL"
)

# ======================================
# 5. SAVE DATASET AS CSV
# ======================================
filename = "machine_sensor_dataset.csv"

df.to_csv(filename, index=False)

# ======================================
# 6. DISPLAY RESULTS
# ======================================
print("Dataset generated successfully!")
print("File:", filename)
print("Number of rows:", len(df))
print("Columns:", list(df.columns))

print("\nState distribution:")
print(df["state"].value_counts())

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

print("\nMissing values:")
print(df.isnull().sum())

"""
Synthetic manufacturing sensor dataset generator.
Simulates equipment telemetry for predictive maintenance use case.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

N_SAMPLES = 5000
START_DATE = datetime(2023, 1, 1)


def generate_sensor_data(n: int = N_SAMPLES) -> pd.DataFrame:
    timestamps = [START_DATE + timedelta(hours=i) for i in range(n)]

    temperature = np.random.normal(loc=75, scale=5, size=n)
    vibration = np.random.normal(loc=0.5, scale=0.1, size=n)
    pressure = np.random.normal(loc=100, scale=8, size=n)
    rpm = np.random.normal(loc=3000, scale=150, size=n)
    oil_level = np.random.uniform(low=0.6, high=1.0, size=n)

    # Inject failure patterns in last 30 hrs before failure events
    n_failures = min(12, max(1, (n - 50) // 40))
    failure_pool = list(range(min(200, n // 2), n))
    failure_events = sorted(random.sample(failure_pool, min(n_failures, len(failure_pool))))
    labels = np.zeros(n, dtype=int)

    for fe in failure_events:
        window = range(max(0, fe - 30), fe + 1)
        for i in window:
            degradation = (i - (fe - 30)) / 30
            temperature[i] += degradation * 20
            vibration[i] += degradation * 0.4
            pressure[i] += degradation * 15
            rpm[i] -= degradation * 300
            oil_level[i] -= degradation * 0.2
        labels[fe] = 1

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": np.round(temperature, 2),
        "vibration_mms": np.round(np.abs(vibration), 3),
        "pressure_bar": np.round(pressure, 2),
        "rpm": np.round(rpm, 0).astype(int),
        "oil_level": np.round(np.clip(oil_level, 0, 1), 3),
        "machine_id": [f"M{random.randint(1,5):02d}" for _ in range(n)],
        "failure": labels
    })

    return df


if __name__ == "__main__":
    df = generate_sensor_data()
    out_path = os.path.join(os.path.dirname(__file__), "synthetic", "sensor_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows → {out_path}")
    print(df.describe())
    print(f"\nFailure rate: {df['failure'].mean():.2%}")

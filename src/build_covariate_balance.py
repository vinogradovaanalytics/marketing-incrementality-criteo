from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)

files = list(RAW_DATA_DIR.glob("criteo-uplift-v2.1*"))

if not files:
    raise FileNotFoundError(
        f"Criteo dataset was not found in: {RAW_DATA_DIR}"
    )

data_path = files[0]

FEATURES = [f"f{i}" for i in range(12)]
CHUNK_SIZE = 500_000


stats = {
    0: {
        feature: {"n": 0, "sum": 0.0, "sum_sq": 0.0}
        for feature in FEATURES
    },
    1: {
        feature: {"n": 0, "sum": 0.0, "sum_sq": 0.0}
        for feature in FEATURES
    },
}


for chunk_number, chunk in enumerate(
    pd.read_csv(
        data_path,
        usecols=FEATURES + ["treatment"],
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    for treatment in [0, 1]:

        group = chunk.loc[
            chunk["treatment"] == treatment,
            FEATURES
        ]

        n = len(group)

        for feature in FEATURES:
            values = group[feature]

            stats[treatment][feature]["n"] += n
            stats[treatment][feature]["sum"] += values.sum()
            stats[treatment][feature]["sum_sq"] += (
                values.pow(2).sum()
            )

    print(f"Chunk {chunk_number} processed")


def calculate_mean_variance(group_stats):
    n = group_stats["n"]
    total = group_stats["sum"]
    total_sq = group_stats["sum_sq"]

    mean = total / n

    variance = (
        total_sq - (total ** 2 / n)
    ) / (n - 1)

    return mean, variance


rows = []

for feature in FEATURES:

    control_mean, control_var = calculate_mean_variance(
        stats[0][feature]
    )

    experiment_mean, experiment_var = calculate_mean_variance(
        stats[1][feature]
    )

    pooled_sd = np.sqrt(
        (control_var + experiment_var) / 2
    )

    if pooled_sd == 0:
        smd = 0.0
    else:
        smd = (
            experiment_mean - control_mean
        ) / pooled_sd

    rows.append({
        "feature": feature,
        "control_mean": control_mean,
        "experiment_mean": experiment_mean,
        "smd": smd,
        "abs_smd": abs(smd),
    })


balance = pd.DataFrame(rows)

balance["balance_status"] = np.where(
    balance["abs_smd"] < 0.10,
    "Баланс приемлемый",
    "Требует внимания"
)

output_path = REPORTS_DIR / "covariate_balance.csv"

balance.to_csv(
    output_path,
    index=False
)

print("\n--- COVARIATE BALANCE ---")
print(balance)

print(f"\nSaved to: {output_path}")
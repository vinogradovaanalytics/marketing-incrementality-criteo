from pathlib import Path

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

CHUNK_SIZE = 500_000

group_totals = {
    0: {
        "users": 0,
        "visits": 0,
        "conversions": 0,
    },
    1: {
        "users": 0,
        "visits": 0,
        "conversions": 0,
    },
}


for chunk_number, chunk in enumerate(
    pd.read_csv(
        data_path,
        usecols=["treatment", "visit", "conversion"],
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    grouped = (
        chunk
        .groupby("treatment")
        .agg(
            users=("treatment", "size"),
            visits=("visit", "sum"),
            conversions=("conversion", "sum"),
        )
    )

    for treatment, row in grouped.iterrows():
        group_totals[treatment]["users"] += int(row["users"])
        group_totals[treatment]["visits"] += int(row["visits"])
        group_totals[treatment]["conversions"] += int(
            row["conversions"]
        )

    print(f"Chunk {chunk_number} processed")


group_metrics = pd.DataFrame.from_dict(
    group_totals,
    orient="index",
)

group_metrics.index.name = "treatment"
group_metrics = group_metrics.reset_index()

group_metrics["group"] = group_metrics["treatment"].map(
    {
        0: "Контрольная группа",
        1: "Экспериментальная группа",
    }
)

group_metrics["visit_rate"] = (
    group_metrics["visits"]
    / group_metrics["users"]
)

group_metrics["conversion_rate"] = (
    group_metrics["conversions"]
    / group_metrics["users"]
)

output_path = REPORTS_DIR / "group_metrics.csv"

group_metrics.to_csv(
    output_path,
    index=False,
)

print("\n--- GROUP METRICS ---")
print(group_metrics)

print(
    f"\nSaved to: {output_path}"
)
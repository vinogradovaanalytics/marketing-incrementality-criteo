from pathlib import Path
import json

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


EXPECTED_COLUMNS = [
    "f0", "f1", "f2", "f3", "f4", "f5",
    "f6", "f7", "f8", "f9", "f10", "f11",
    "treatment", "conversion", "visit", "exposure"
]

BINARY_COLUMNS = [
    "treatment",
    "conversion",
    "visit",
    "exposure"
]

CHUNK_SIZE = 500_000


total_rows = 0

missing_counts = {
    column: 0
    for column in EXPECTED_COLUMNS
}

value_counts = {
    column: {}
    for column in BINARY_COLUMNS
}

treatment_exposure_counts = {}


print(f"Dataset: {data_path.name}")
print(f"Size: {data_path.stat().st_size / 1024**3:.2f} GB")
print("\nScanning dataset...")


for chunk_number, chunk in enumerate(
    pd.read_csv(data_path, chunksize=CHUNK_SIZE),
    start=1
):

    if list(chunk.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Dataset columns do not match the expected schema."
        )

    total_rows += len(chunk)

    # Missing values
    chunk_missing = chunk.isna().sum()

    for column in EXPECTED_COLUMNS:
        missing_counts[column] += int(chunk_missing[column])

    # Values in binary columns
    for column in BINARY_COLUMNS:

        counts = chunk[column].value_counts(dropna=False)

        for value, count in counts.items():

            key = str(value)

            value_counts[column][key] = (
                value_counts[column].get(key, 0)
                + int(count)
            )

    # treatment × exposure consistency
    cross_counts = (
        chunk
        .groupby(["treatment", "exposure"])
        .size()
    )

    for (treatment, exposure), count in cross_counts.items():

        key = f"treatment={treatment}, exposure={exposure}"

        treatment_exposure_counts[key] = (
            treatment_exposure_counts.get(key, 0)
            + int(count)
        )

    print(
        f"Chunk {chunk_number}: "
        f"{total_rows:,} rows processed"
    )


summary = {
    "file": data_path.name,
    "file_size_gb": round(
        data_path.stat().st_size / 1024**3,
        3
    ),
    "rows": total_rows,
    "columns": EXPECTED_COLUMNS,
    "missing_values": missing_counts,
    "binary_value_counts": value_counts,
    "treatment_exposure": treatment_exposure_counts,
}


# Calculate overall rates

summary["rates"] = {
    "treatment_rate":
        value_counts["treatment"].get("1", 0) / total_rows,

    "conversion_rate":
        value_counts["conversion"].get("1", 0) / total_rows,

    "visit_rate":
        value_counts["visit"].get("1", 0) / total_rows,

    "exposure_rate":
        value_counts["exposure"].get("1", 0) / total_rows,
}


output_path = REPORTS_DIR / "data_audit_summary.json"

with output_path.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False
    )


print("\n--- DATA AUDIT COMPLETE ---")

print(f"Rows: {total_rows:,}")

print("\nRates:")

for metric, value in summary["rates"].items():
    print(f"{metric}: {value:.6f}")

print("\nMissing values:")

for column, count in missing_counts.items():
    print(f"{column}: {count:,}")

print(
    f"\nReport saved to: "
    f"{output_path}"
)
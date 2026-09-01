from pathlib import Path
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "iris.data"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "iris_clean.csv"


# Column names for the original UCI Iris dataset
COLUMN_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]


def load_raw_data():
    """Load the original Iris dataset."""
    df = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=COLUMN_NAMES
    )

    return df


def clean_data(df):
    """Perform basic cleaning and standardisation."""

    df = df.copy()

    # Remove completely empty rows
    df.dropna(how="all", inplace=True)

    # Make sure measurement columns are numeric
    numeric_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Standardise species names
    df["species"] = (
        df["species"]
        .str.strip()
        .str.replace("Iris-", "", regex=False)
        .str.lower()
    )

    return df


def save_processed_data(df):
    """Save cleaned dataset to the processed-data folder."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def main():
    df = load_raw_data()
    df = clean_data(df)

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nFirst 5 rows:")
    print(df.head())

    save_processed_data(df)

    print(f"\nCleaned dataset saved to:")
    print(PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
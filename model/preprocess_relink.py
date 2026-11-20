from pathlib import Path
import pandas as pd


def load_relink_arff(file_path):
    attributes = []
    data = []
    in_data = False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.lower().startswith("@attribute"):
                parts = line.split()
                attributes.append(parts[1])

            elif line.lower() == "@data":
                in_data = True
                continue

            elif in_data:
                if line.startswith("###") or not line:
                    continue

                data.append(line.split(","))

    return pd.DataFrame(data, columns=attributes)


def prepare_dataset():
    data_folder = Path("data/ReLink")
    datasets = []

    for file in data_folder.glob("*.arff"):
        df = load_relink_arff(file)

        # Convert all feature columns to numeric
        feature_columns = [
            column for column in df.columns
            if column != "isDefective"
        ]

        for column in feature_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Convert defect label to 0/1
        df["isDefective"] = df["isDefective"].map({
            "FALSE": 0,
            "TRUE": 1
        })

        # Handle the known invalid AvgLine overflow values
        df.loc[df["AvgLine"] > 1000, "AvgLine"] = pd.NA

        # Fill missing AvgLine values with the dataset median
        df["AvgLine"] = df["AvgLine"].fillna(df["AvgLine"].median())

        # Track source project
        df["project"] = file.stem

        datasets.append(df)

    combined = pd.concat(datasets, ignore_index=True)

    return combined


if __name__ == "__main__":
    df = prepare_dataset()

    print("=" * 60)
    print("ReLink dataset prepared")
    print("=" * 60)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nDefect labels:")
    print(df["isDefective"].value_counts())

    print("\nMissing values:", df.isnull().sum().sum())

    print("\nProjects:")
    print(df["project"].value_counts())

    print("\nAvgLine range:")
    print("Min:", df["AvgLine"].min())
    print("Max:", df["AvgLine"].max())
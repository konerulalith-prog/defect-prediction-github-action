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

                values = line.split(",")
                data.append(values)

    return pd.DataFrame(data, columns=attributes)


data_folder = Path("data/ReLink")

for file in data_folder.glob("*.arff"):
    df = load_relink_arff(file)

    # Convert feature columns to numeric values
    feature_columns = [col for col in df.columns if col != "isDefective"]

    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    print("\n" + "=" * 60)
    print("Dataset:", file.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nDefect labels:")
    print(df["isDefective"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nFeature summary:")
    print(df[feature_columns].describe().T[["min", "max", "mean"]])
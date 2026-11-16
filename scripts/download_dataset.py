"""Download and validate the two UNSW-NB15 partitions used by the notebook."""

from pathlib import Path
import shutil

import kagglehub
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASET = "mrwellsdavid/unsw-nb15"
EXPECTED = {
    175_341: "UNSW_NB15_training_set.csv",
    82_332: "UNSW_NB15_testing_set.csv",
}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    targets = [DATA / name for name in EXPECTED.values()]
    if not all(path.is_file() for path in targets):
        staging = DATA / ".download"
        for name in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
            kagglehub.dataset_download(DATASET, path=name, output_dir=str(staging))

        # Validate both files before assigning canonical names. Kaggle's names
        # are swapped relative to the partitions selected for this project.
        sources = {}
        schema = None
        for path in staging.rglob("*.csv"):
            frame = pd.read_csv(path)
            if frame.shape[1] != 45 or len(frame) not in EXPECTED:
                raise ValueError(f"Unexpected dataset dimensions: {path}: {frame.shape}")
            if not {"id", "label", "attack_cat"}.issubset(frame.columns):
                raise ValueError(f"Missing required dataset columns: {path}")
            if schema is not None and list(frame.columns) != schema:
                raise ValueError("Training and test schemas differ.")
            schema = list(frame.columns)
            if len(frame) in sources:
                raise ValueError(f"Multiple partitions with {len(frame):,} rows.")
            sources[len(frame)] = path
        if set(sources) != set(EXPECTED):
            raise ValueError("Download did not contain both expected partitions.")
        for rows, name in EXPECTED.items():
            temporary = DATA / (name + ".tmp")
            shutil.copyfile(sources[rows], temporary)
            temporary.replace(DATA / name)

    for rows, name in EXPECTED.items():
        path = DATA / name
        shape = pd.read_csv(path).shape
        if shape != (rows, 45):
            raise ValueError(f"Expected {(rows, 45)} for {path}; found {shape}")
        print(f"{name}: {shape[0]:,} rows, {shape[1]} columns")
    print(f"Notebook dataset directory: {DATA}")


if __name__ == "__main__":
    main()

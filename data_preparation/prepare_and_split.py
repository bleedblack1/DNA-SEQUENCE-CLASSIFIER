import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



class StratifiedDataSplitter:
    """Stratified data splitting with .npy file creation"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)


    def load_preprocessed_data(self, csv_file: str) -> pd.DataFrame:
        logger.info(f"Loading preprocessed data: {csv_file}")

        if not Path(csv_file).exists():
            raise FileNotFoundError(f"File not found: {csv_file}")

        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} sequences")

        return df


    def verify_data_integrity(self, df: pd.DataFrame) -> bool:
        logger.info("Verifying data integrity...")

        required_cols = ["sequence", "label", "label_name"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check sequence characters
        invalid = df["sequence"].apply(
            lambda x: not all(c in "ATGCN" for c in x)
        ).sum()

        if invalid > 0:
            logger.warning(f"{invalid} invalid sequences found")
            return False

        # Check sequence length
        lengths = df["sequence"].apply(len)
        if not (lengths == 256).all():
            logger.warning("Not all sequences are 256 bp")
            return False

        logger.info("Data integrity verified ✔")
        return True


    def encode_sequence_onehot(self, sequence: str) -> np.ndarray:
        """
        One-hot encode DNA sequence.
        A,T,G,C → one-hot
        N → uniform distribution
        """
        if len(sequence) != 256:
            raise ValueError("Sequence length must be exactly 256 bp")

        base_map = {"A": 0, "T": 1, "G": 2, "C": 3}
        encoded = np.zeros((256, 4), dtype=np.float32)

        for i, base in enumerate(sequence.upper()):
            if base in base_map:
                encoded[i, base_map[base]] = 1.0
            else:  # N
                encoded[i, :] = 0.25

        return encoded

    def stratified_train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        logger.info("STRATIFIED TRAIN / VAL / TEST SPLITTING")

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Train vs Temp
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=self.random_seed,
            stratify=df["label"],
        )

        # Val vs Test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=self.random_seed,
            stratify=temp_df["label"],
        )

        logger.info(f"Train: {len(train_df)}")
        logger.info(f"Val:   {len(val_df)}")
        logger.info(f"Test:  {len(test_df)}")

        return train_df, val_df, test_df

    def verify_stratification(self, train_df, val_df, test_df) -> bool:
        logger.info("Verifying stratification...")

        splits = {
            "TRAIN": train_df,
            "VAL": val_df,
            "TEST": test_df,
        }

        num_classes = train_df["label"].nunique()
        expected_pct = 100 / num_classes
        balanced = True

        for name, split in splits.items():
            logger.info(f"\n{name}")
            total = len(split)
            dist = split["label"].value_counts().sort_index()

            for label, count in dist.items():
                pct = (count / total) * 100
                is_balanced = abs(pct - expected_pct) < 5
                status = "Balanced" if is_balanced else "Not balanced"

                logger.info(
                    f" {status:12s} | Class {label}: {count} ({pct:.1f}%)"
                )

                if not is_balanced:
                    balanced = False

        return balanced


    def create_npy_files(
        self,
        train_df,
        val_df,
        test_df,
        output_dir="data/processed",
        encode_sequences=True,
    ) -> dict:
        logger.info("Creating .npy files...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "random_seed": self.random_seed,
            "total_sequences": len(train_df) + len(val_df) + len(test_df),
            "splits": {},
        }

        for name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            logger.info(f"Processing {name.upper()}")

            if encode_sequences:
                sequences = np.array(
                    [self.encode_sequence_onehot(seq) for seq in split_df["sequence"]],
                    dtype=np.float32,
                )
            else:
                sequences = split_df["sequence"].values

            labels = split_df["label"].values.astype(np.uint8)

            np.save(output_path / f"{name}_sequences.npy", sequences)
            np.save(output_path / f"{name}_labels.npy", labels)

            metadata["splits"][name] = {
                "count": len(split_df),
                "shape": sequences.shape,
                "classes": split_df["label"].value_counts().to_dict(),
            }

        return metadata

  
    def save_metadata(self, metadata: dict, output_dir="data/processed"):
        path = Path(output_dir) / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved → {path}")


    def print_summary(self, metadata: dict):
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)

        logger.info(f"Total sequences: {metadata['total_sequences']}")
        logger.info(f"Random seed: {metadata['random_seed']}")

        for split, info in metadata["splits"].items():
            logger.info(f"\n{split.upper()}")
            logger.info(f" Count : {info['count']}")
            logger.info(f" Shape : {info['shape']}")
            logger.info(f" Classes: {info['classes']}")



def main():
    logger.info("MODULE 1: DATA PREPARATION & VALIDATION")

    INPUT_CSV = "data/raw/preprocessed_dna.csv"
    OUTPUT_DIR = "data/processed"

    splitter = StratifiedDataSplitter(random_seed=42)

    logger.info("[1/5] Loading data")
    df = splitter.load_preprocessed_data(INPUT_CSV)

    logger.info("[2/5] Verifying integrity")
    splitter.verify_data_integrity(df)

    logger.info("[3/5] Stratified split")
    train_df, val_df, test_df = splitter.stratified_train_test_split(df)

    logger.info("[4/5] Verifying stratification")
    splitter.verify_stratification(train_df, val_df, test_df)

    logger.info("[5/5] Saving .npy files")
    metadata = splitter.create_npy_files(
        train_df, val_df, test_df, OUTPUT_DIR, encode_sequences=True
    )

    splitter.save_metadata(metadata, OUTPUT_DIR)
    splitter.print_summary(metadata)

    logger.info("MODULE 1 COMPLETE ")
    logger.info("Ready for Module 2: Feature Engineering")


if __name__ == "__main__":
    main()

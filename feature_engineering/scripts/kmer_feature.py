import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from itertools import product
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KmerFeatureExtractor:
    """Extract k-mer frequency features from DNA sequences."""
    
    def __init__(self, k_values: list = None):
        """
        Args:
            k_values: List of k values to generate
        """
        self.k_values = k_values or [1, 2, 3, 4, 5]
        self.nucleotides = ['A', 'T', 'G', 'C']
        
        # Pre-compute all possible k-mers for each k
        self.kmers_dict = {}
        for k in self.k_values:
            self.kmers_dict[k] = self._generate_kmers(k)
    
    def _generate_kmers(self, k: int) -> list:
        """Generate all possible k-mers."""
        return [''.join(p) for p in product(self.nucleotides, repeat=k)]
    
    def extract_kmers(self, sequence: str) -> dict:
        """
        Extract all k-mers from a sequence.
        
        Args:
            sequence: DNA sequence string
        
        Returns:
            Dictionary {k: [k-mer list]}
        """
        kmers = {}
        sequence = sequence.upper()
        
        for k in self.k_values:
            seq_kmers = []
            for i in range(len(sequence) - k + 1):
                seq_kmers.append(sequence[i:i+k])
            kmers[k] = seq_kmers
        
        return kmers
    
    def get_kmer_frequencies(self, sequence: str, k: int) -> np.ndarray:
        """
        Get k-mer frequency vector for a sequence.
        
        Returns array of shape (4^k,) with counts for each possible k-mer.
        """
        kmers = self.extract_kmers(sequence)[k]
        counter = Counter(kmers)
        
        # Create frequency vector in sorted k-mer order
        kmer_list = self.kmers_dict[k]
        frequencies = np.array([counter.get(kmer, 0) for kmer in kmer_list], dtype=np.float32)
        
        return frequencies
    
    def get_normalized_frequencies(self, sequence: str, k: int) -> np.ndarray:
        """Get normalized (probability) k-mer frequencies."""
        frequencies = self.get_kmer_frequencies(sequence, k)
        
        # Normalize by total count
        if frequencies.sum() > 0:
            frequencies = frequencies / frequencies.sum()
        
        return frequencies
    
    def extract_all_features(self, sequence: str, normalize: bool = True) -> dict:
        """
        Extract all k-mer features for all k values.
        
        Args:
            sequence: DNA sequence
            normalize: If True, use probability distributions
        
        Returns:
            Dictionary {k: feature_vector}
        """
        features = {}
        
        for k in self.k_values:
            if normalize:
                features[k] = self.get_normalized_frequencies(sequence, k)
            else:
                features[k] = self.get_kmer_frequencies(sequence, k)
        
        return features


class FeatureDatasetBuilder:
    """Build feature matrices for train/val/test splits."""
    
    def __init__(self, k_values: list = None):
        """
        Args:
            k_values: List of k values to extract
        """
        self.extractor = KmerFeatureExtractor(k_values)
        self.k_values = self.extractor.k_values
    
    def load_preprocessed_data(self, csv_file: str) -> pd.DataFrame:
        """Load preprocessed sequences from Module 0."""
        logger.info(f"Loading preprocessed data: {csv_file}")
        
        if not Path(csv_file).exists():
            raise FileNotFoundError(f"File not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        logger.info(f"   Loaded {len(df)} sequences")
        
        return df
    
    def load_metadata(self, metadata_file: str) -> dict:
        """Load split metadata from Module 1."""
        logger.info(f"Loading metadata: {metadata_file}")
        
        if not Path(metadata_file).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"   Loaded metadata for {len(metadata['splits'])} splits")
        return metadata
    
    def create_feature_matrices(
        self,
        df: pd.DataFrame,
        split_indices: dict,
        normalize: bool = True,
        output_dir: str = "data/processed"
    ) -> dict:
        """
        Create k-mer feature matrices for each split.
        
        Args:
            df: Preprocessed sequences DataFrame
            split_indices: Dict with 'train', 'val', 'test' index arrays
            normalize: If True, use probability distributions
            output_dir: Where to save .npy files
        
        Returns:
            Dictionary with feature statistics
        """
          
        logger.info("EXTRACTING k-MER FEATURES")
           
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        statistics = {}
        
        for split_name, indices in split_indices.items():
            
            logger.info(f"Processing {split_name.upper()} split ({len(indices)} sequences)")
            
            
            split_df = df.iloc[indices].reset_index(drop=True)
            
            # Extract features for each k
            for k in self.k_values:
                logger.info(f"\n  Extracting k={k} features...")
                
                # Get feature dimension
                feature_dim = 4 ** k
                
                # Initialize feature matrix
                features = np.zeros((len(split_df), feature_dim), dtype=np.float32)
                
                # Extract features for each sequence
                for i, sequence in enumerate(split_df['sequence']):
                    if normalize:
                        features[i] = self.extractor.get_normalized_frequencies(sequence, k)
                    else:
                        features[i] = self.extractor.get_kmer_frequencies(sequence, k)
                    
                    if (i + 1) % 1000 == 0:
                        logger.info(f"    Processed {i+1}/{len(split_df)} sequences")
                
                # Save features
                save_path = output_path / f"{split_name}_kmers_k{k}.npy"
                np.save(save_path, features)
                
                size_mb = save_path.stat().st_size / (1024 * 1024)
                logger.info(f"    Saved {save_path.name} ({size_mb:.2f} MB)")
                logger.info(f"     Shape: {features.shape}")
                logger.info(f"     Mean: {features.mean():.4f}, Std: {features.std():.4f}")
                
                # Store statistics
                key = f"{split_name}_k{k}"
                statistics[key] = {
                    'shape': features.shape,
                    'mean': float(features.mean()),
                    'std': float(features.std()),
                    'min': float(features.min()),
                    'max': float(features.max()),
                    'sparsity': float((features == 0).sum() / features.size)
                }
        
        return statistics
    
    def analyze_feature_importance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k: int,
        output_dir: str = "data/processed"
    ) -> dict:
        """
        Analyze which k-mers are most discriminative.
        
        Uses mutual information between k-mer frequency and class label.
        
        Args:
            X_train: Training feature matrix (N, feature_dim)
            y_train: Training labels (N,)
            k: k value
            output_dir: Where to save analysis
        
        Returns:
            Dictionary with top k-mers
        """
        logger.info(f"\n  Analyzing feature importance for k={k}...")
        
        # Get all possible k-mers
        kmer_list = self.extractor.kmers_dict[k]
        
        # Calculate correlation between each feature and labels
        correlations = []
        for feature_idx in range(X_train.shape):
            # Correlation between this feature and labels
            corr = np.corrcoef(X_train[:, feature_idx], y_train)[0, 1]
            correlations.append((kmer_list[feature_idx], corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x), reverse=True)
        
        logger.info(f"  Top 10 discriminative k-mers (k={k}):")
        top_kmers = {}
        for i, (kmer, corr) in enumerate(correlations[:10]):
            logger.info(f"    {i+1}. {kmer}: {corr:.4f}")
            top_kmers[kmer] = float(corr)
        
        return {
            'k': k,
            'top_kmers': top_kmers,
            'total_features': len(kmer_list)
        }


def extract_split_indices(
    df: pd.DataFrame,
    metadata: dict,
    random_seed: int = 42
) -> dict:
    """
    Extract train/val/test indices from original data.
    
    Requires knowing the exact splits used in Module 1.
    This function reconstructs them using the same random seed and stratification.
    """
    logger.info("\nReconstucting split indices from original data...")
    
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_seed)
    
    # First split: train (70%) vs temp (30%)
    train_idx, temp_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.3,
        random_state=random_seed,
        stratify=df['label']
    )
    
    # Second split: val (50%) vs test (50%) from temp
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=random_seed,
        stratify=df.iloc[temp_idx]['label']
    )
    
    logger.info(f"  Train indices: {len(train_idx)}")
    logger.info(f"  Val indices: {len(val_idx)}")
    logger.info(f"  Test indices: {len(test_idx)}")
    
    return {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }


def main():
    """Main execution for Module 2."""
    
    logger.info(" MODULE 2: FEATURE ENGINEERING & k-MER FEATURES")
    
    # Paths
    INPUT_CSV = "data/raw/preprocessed_dna.csv"
    METADATA_FILE = "data/processed/metadata.json"
    OUTPUT_DIR = "data/processed"
    K_VALUES = [1, 2, 3, 4, 5]
    
    # Load data
    logger.info("\n Loading preprocessed data...")
    builder = FeatureDatasetBuilder(k_values=K_VALUES)
    df = builder.load_preprocessed_data(INPUT_CSV)
    
    # Load metadata
    logger.info("\n Loading split metadata...")
    metadata = builder.load_metadata(METADATA_FILE)
    
    # Reconstruct split indices
    logger.info("\n Reconstructing split indices...")
    split_indices = extract_split_indices(df, metadata, random_seed=42)
    
    # Extract k-mer features
    logger.info("\n Extracting k-mer features...")
    statistics = builder.create_feature_matrices(
        df,
        split_indices,
        normalize=True,
        output_dir=OUTPUT_DIR
    )
    
    # Save statistics
    logger.info("\n Saving feature statistics...")
    stats_path = Path(OUTPUT_DIR) / "kmer_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    logger.info(f" Statistics saved: {stats_path}")
    
    
    logger.info(f"\nGenerated files in {OUTPUT_DIR}:")
    for k in K_VALUES:
        for split in ['train', 'val', 'test']:
            filename = f"{split}_kmers_k{k}.npy"
            filepath = Path(OUTPUT_DIR) / filename
            if filepath.exists():
                size = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"  • {filename:25s} ({size:6.2f} MB)")
    logger.info(f"  • kmer_statistics.json")
    
    


if __name__ == "__main__":
    main()

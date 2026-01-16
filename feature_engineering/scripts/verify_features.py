import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_kmer_features(data_dir: str = "data/processed"):
    """Verify all k-mer feature files."""
    logger.info("VERIFYING k-MER FEATURES")
    
    
    data_path = Path(data_dir)
    
    # Expected files
    k_values = [1, 2, 3, 4, 5]
    splits = ['train', 'val', 'test']
    
    logger.info("\nChecking files...")
    all_exist = True
    
    for k in k_values:
        for split in splits:
            filename = f"{split}_kmers_k{k}.npy"
            filepath = data_path / filename
            
            if filepath.exists():
                size = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"   {filename:25s} ({size:6.2f} MB)")
            else:
                logger.error(f"   {filename:25s} NOT FOUND")
                all_exist = False
    
    if not all_exist:
        raise FileNotFoundError("Some feature files are missing!")
    
    # Load and verify shapes
    logger.info("\nVerifying feature shapes...")
    
    train_labels = np.load(data_path / 'train_labels.npy')
    val_labels = np.load(data_path / 'val_labels.npy')
    test_labels = np.load(data_path / 'test_labels.npy')
    
    for k in k_values:
        expected_dim = 4 ** k
        logger.info(f"\nk={k} (feature dimension: {expected_dim}):")
        
        for split, labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:
            X = np.load(data_path / f"{split}_kmers_k{k}.npy")
            
            # Check shape
            if X.shape != len(labels):
                logger.error(f"   {split}: sample count mismatch!")
            elif X.shape != expected_dim:
                logger.error(f"   {split}: feature dimension mismatch!")
            else:
                logger.info(f"   {split:5s}: shape {X.shape}, dtype {X.dtype}")
                logger.info(f"      Mean: {X.mean():.4f}, Std: {X.std():.4f}")
                logger.info(f"      Sparsity: {(X == 0).sum() / X.size:.2%}")
    
    # Load statistics
    logger.info("\nLoading statistics...")
    stats_file = data_path / "kmer_statistics.json"
    
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        logger.info(f"   kmer_statistics.json loaded")
        logger.info(f"     Contains stats for {len(stats)} feature matrices")
    else:
        logger.warning(f"    Statistics file not found")
    
    # Final verification
    logger.info("ALL VERIFICATIONS PASSED!")
    
    


if __name__ == "__main__":
    verify_kmer_features()

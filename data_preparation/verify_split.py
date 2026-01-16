import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_npy_files(data_dir: str = "data/processed"):
    """Verify all .npy files exist and have correct shapes."""
    logger.info("VERIFYING .NPY FILES")
    
    
    data_path = Path(data_dir)
    
    required_files = [
        'train_sequences.npy', 'train_labels.npy',
        'val_sequences.npy', 'val_labels.npy',
        'test_sequences.npy', 'test_labels.npy',
        'metadata.json'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = data_path / filename
        exists = filepath.exists()
        status = "Exist" if exists else "Not exist"
        
        if exists:
            size = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"{status} {filename:30s} ({size:6.2f} MB)")
        else:
            logger.error(f"{status} {filename:30s} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        raise FileNotFoundError("Some .npy files are missing!")
    
    logger.info("\n All files present")
    
    # Verify file integrity
    logger.info("\nVerifying file integrity...")
    
    train_seq = np.load(data_path / 'train_sequences.npy')
    train_labels = np.load(data_path / 'train_labels.npy')
    val_seq = np.load(data_path / 'val_sequences.npy')
    val_labels = np.load(data_path / 'val_labels.npy')
    test_seq = np.load(data_path / 'test_sequences.npy')
    test_labels = np.load(data_path / 'test_labels.npy')
    
    logger.info(f"\nTRAIN:")
    logger.info(f"  Sequences: {train_seq.shape} (expected: (N, 256, 4))")
    logger.info(f"  Labels: {train_labels.shape}")
    logger.info(f"  Match: {len(train_seq) == len(train_labels)}")
    
    logger.info(f"\nVAL:")
    logger.info(f"  Sequences: {val_seq.shape}")
    logger.info(f"  Labels: {val_labels.shape}")
    logger.info(f"  Match: {len(val_seq) == len(val_labels)}")
    
    logger.info(f"\nTEST:")
    logger.info(f"  Sequences: {test_seq.shape}")
    logger.info(f"  Labels: {test_labels.shape}")
    logger.info(f"  Match: {len(test_seq) == len(test_labels)}")
    
    logger.info(f"\nClass distribution:")
    for split_name, labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"\n  {split_name}:")
        for label, count in zip(unique, counts):
            pct = count / len(labels) * 100
            logger.info(f"    Class {label}: {count:5d} ({pct:5.1f}%)")
    
    # Load and verify metadata
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"\nMetadata:")
    logger.info(f"  Random seed: {metadata['random_seed']}")
    logger.info(f"  Total sequences: {metadata['total_sequences']}")
    
    logger.info("\n All verifications passed!")


if __name__ == "__main__":
    verify_npy_files()

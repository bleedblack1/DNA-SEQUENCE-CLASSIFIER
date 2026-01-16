"""
Convert genomic coordinates (GRCh38/hg38) to DNA sequences.
Takes train/test CSV files with chr:start:end and produces sequence CSV.
"""

import pandas as pd
import pysam
from pathlib import Path
import logging
from tqdm import tqdm 

logging.basicConfig(
    level = logging.INFO,
    format= '%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class GRCh38SequenceExtractor:
    """Extract DNA sequences from GRCh38 coordinates."""
    def __init__(self, reference_fasta: str):
        """
        Args:
            reference_fasta: Path to hg38.fa file
        """
        ref_path = Path(reference_fasta)
        
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference Genome not found {reference_fasta}"
            )
        
        logger.info(f"Loading reference genome: {reference_fasta}")
        try:
           self.fasta = pysam.FastaFile(reference_fasta)
           logger.info("Reference genome loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to open FASTA: {e}\n"
                               f"Make sure pysam is installed: pip install pysam")
            
    def fetch_sequence(self, chromosome: str, start: int, end: int)-> str:
        """
        Fetch DNA sequence from GRCh38.
        
        Args:
            chromosome: Chromosome name (e.g., "chr11")
            start: Start position (0-based, from CSV)
            end: End position (exclusive, from CSV)
        
        Returns:
            DNA sequence (uppercase) or None if fetch fails
        """
        try:
            # pysam uses 0-based coordinates(same as your csv likely)
            seq = self.fasta.fetch(chromosome, start, end)
            if seq:
                return seq.upper()
            else:
                return None
        
        except Exception as e:
            logger.warning(f"Failed to fetch{chromosome}:{start}-{end}: {e}")
            return None
        
    def convert_csv(self, input_csv: str, label: str, output_csv: str = None) -> pd.DataFrame:
        """
        Convert coordinate CSV to sequence CSV.
        
        Args:
            input_csv: Path to input CSV 
                      Expected columns: id, region, start, end
            label: Label for sequences ("promoter", "enhancer", "non-regulatory")
            output_csv: Save to this file (optional)
        
        Returns:
            DataFrame with sequences
        """
        logger.info(f"Processing{input_csv}")
        logger.info(f"Label{label}")
        
        # Read input 
        logger.info("Reading CSV....")
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} genomic regions")
        
        # Expected columns
        if 'region' not in df.columns or 'start' not in df.columns or 'end' not in df.columns:
            raise ValueError(f"CSV must have columns: region, start, end\n"
                             f"Found: {list(df.columns)}")
        
        # Extract sequences
        logger.info("Extracting sequences from GRCh38.")   
        sequences = []
        chromosomes = []
        start_pos = []
        end_pos = []
        
        for idx, row in tqdm(df.iterrows(), total = len(df), desc=f"Converting {label}"):
            chromosome = row['region']
            start = int(row['start'])
            end = int(row['end'])
            
            seq = self.fetch_sequence(chromosome, start, end)
            
            sequences.append(seq)
            chromosomes.append(chromosome)
            start_pos.append(start)
            end_pos.append(end)
            
        # Create output DataFrame
        df_out = pd.DataFrame({
            'sequenced_id': [f"{label}_{i:06d}" for i in range(len(df))],
            'sequence': sequences,
            'label': label,
            'label_name':label,
            'source':f'GRCh38_{label}',
            'chromosome':chromosomes,
            'start': start_pos,
            'end': end_pos
        })    
         # Remove failed sequences
        failed = df_out['sequence'].isna().sum()
        if failed > 0:
            logger.warning(f"  Failed to fetch {failed} sequences")
            df_out = df_out.dropna(subset=['sequence'])
        
        logger.info(f" Successfully extracted {len(df_out)} sequences")
        
        # Save
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(output_csv, index=False)
            logger.info(f" Saved to: {output_csv}")
        
        return df_out
    
def main(reference_fasta: str, train_dir: str, test_dir: str, output_dir: str = "data_collection/raw_downloads"):
    """
    Process all CSV files (promoter, enhancer, ocr) from train and test.
    
    Args:
        reference_fasta: Path to hg38.fa
        train_dir: Path to train directory
        test_dir: Path to test directory
        output_dir: Where to save processed files
    """    
    logger.info("GRCh38 COORDINATE CONVERSION")
    
    # Initialize extractor
    extractor = GRCh38SequenceExtractor(reference_fasta)
    
    all_sequences = []
    
    # file_mapping
    file_mapping = [
        ("promoter.csv", "promoter"),
        ("enhancer.csv", "enhancer"),
        ("ocr.csv", "non-regulatory")
    ]
    
    # Process train dir
    logger.info("Train Set")
    
    for filename, label in file_mapping:
        input_path = Path(train_dir) / filename
        
        if not input_path.exists():
            logger.warning(f"⚠️  Not found: {input_path}")
            continue
        
        output_path = Path(output_dir) / f"train_{label}_sequences.csv"
        
        try: 
            df = extractor.convert_csv(
                input_csv=str(input_path),
                label = label,
                output_csv=str(output_path)
            )
            all_sequences.append(df)
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            
    # Process the test directory
    logger.info("Test Set")
    
    for filename, label in file_mapping:
        input_path = Path(test_dir) / filename
        
        if not input_path.exists():
            logger.warning(f" Not found: {input_path}")
            continue
        
        output_path = Path(output_dir) / f"test_{label}_sequences.csv"
        
        try:
            df = extractor.convert_csv(
                input_csv=str(input_path),
                label=label,
                output_csv=str(output_path)
            )
            all_sequences.append(df)
        except Exception as e:
            logger.error(f" Error processing {filename}: {e}")   
            
    if all_sequences:
        
        logger.info("COMBINING ALL SEQUENCES")
        
        df_combined = pd.concat(all_sequences, ignore_index=True)
        
        # Shuffle
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined
        combined_path = Path(output_dir) / "grch38_regulatory_sequences.csv"
        df_combined.to_csv(combined_path, index=False)
        
        logger.info(f"\n CONVERSION COMPLETE")
        
        logger.info(f"Total sequences: {len(df_combined):,}")
        logger.info(f"  • Promoters:       {(df_combined['label'] == 'promoter').sum():,}")
        logger.info(f"  • Enhancers:       {(df_combined['label'] == 'enhancer').sum():,}")
        logger.info(f"  • Non-regulatory:  {(df_combined['label'] == 'non-regulatory').sum():,}")
        logger.info(f"\nSaved to: {combined_path}")
        logger.info(f"File size: {combined_path.stat().st_size / (1024*1024):.2f} MB")
        
        return df_combined
    else:
        logger.error(" No sequences were processed!")
        return None   
    
if __name__ == "__main__":
    
    
    
    
    REFERENCE_FASTA = "data_collection/raw_downloads/reference_genomes/hg38.fa"

    TRAIN_DIR = "data_collection/raw_downloads/human_ensembl_regulatory/train"
    TEST_DIR  = "data_collection/raw_downloads/human_ensembl_regulatory/test"

    OUTPUT_DIR = "data_collection/raw_downloads/processed_sequences"
    
    
    df = main(
        reference_fasta=REFERENCE_FASTA,
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        output_dir=OUTPUT_DIR
    )                  
            
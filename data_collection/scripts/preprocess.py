import numpy as np
import pandas as pd 
from pathlib import Path
import logging
from typing import Tuple 

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Complete preprocessing pipeline for DNA sequences"""
    
    VALID_NUCLEOTIDES = {'A', 'T', 'G', 'C', 'N'}
    
    def __init__(self, target_length: int = 256):
        """
        Arg:
            target_length: Normalize all sequence to this length(default: 256bp)
        """
        self.target_length = target_length
        
    def is_valid_sequence(self, seq: str) -> bool:
        """Check if sequence contain only valid bases """
        if not seq:
            return False
        return all(base.upper() in DataPreprocessor.VALID_NUCLEOTIDES for base in seq)
    
    def is_valid_length(self, seq: str, min_len: int = 100, max_len: int = 1000)-> bool:
        """Check if sequence length is with in valid range """
        return min_len <= len(seq) <= max_len
    
    
    def clean_sequence(self, seq:str, remov_n: bool = False)-> str:
        """
        Clean sequence:
        - convert to uppercase
        - Remove gaps and whitespace
        - Optionally remove 'N' bases
        
        Args:
            seq: Input Sequence
            remove_n: If True, remove 'N' bases
            
        Returns:
            Cleaned sequence    
        """
        seq = str(seq).upper().strip()
        seq = seq.replace('-', '').replace(' ','').replace('\t', '')
        
        if remov_n:
            seq = seq.replace('N','')
            
        return seq
    
    
    def pad_or_trim_sequence(self, seq: str)->str:
        """
        Normalize sequence to target length.
        
        If shorter: pad with 'N' on right
        If longer: trim from right
        
        Args:
            seq: Input sequence
        
        Returns:
            Normalized sequence of exact target_length
        """
        if len(seq) < self.target_length:
            # Pad with N
            seq = seq + 'N' * (self.target_length - len(seq))
        else:
            # Trim
            seq = seq[:self.target_length]
            
        return seq
    
    def calculate_gc_content(self, seq: str) -> float:
        """
        Calculate GC content as percentage (0-100).
        
        GC content = (G count + C count) / total length * 100
        """
        if len(seq) == 0:
            return 0.0
        gc_count = seq.count('G') + seq.count('C')
        return (gc_count / len(seq)) * 100
    
    def count_n_bases(self, seq: str) -> int:
        """Count 'N' (unknown) bases in sequence."""
        return seq.count('N')
    
    def calculate_homopolymer_runs(self, seq: str, threshold: int = 5) -> int:
        """
        Count homopolymer runs (same base repeated > threshold times).
        
        Indicates potential quality issues.
        
        Args:
            seq: Input sequence
            threshold: If same base repeats > this, count as run
        
        Returns:
            Number of long homopolymer runs
        """
        runs = 0
        max_run = 1
        current_base = None
        
        for base in seq:
            if base == current_base:
                max_run += 1
            else:
                if max_run > threshold:
                    runs += 1
                current_base = base
                max_run = 1
                
        if max_run > threshold: 
            runs += 1
        
        return runs  
    
    def map_labels(self, df: pd.DataFrame, label_col: str = 'label') -> pd.DataFrame:
        """
        Convert string labels to numeric and create label_numeric column.
        
        Args:
            df: Input DataFrame
            label_col: Column name with labels
        
        Returns:
            DataFrame with label_numeric column
        """
        
        label_map = {
            'promoter': 1,
            'enhancer': 2,
            'non-regulatory': 0,
            'ocr': 0  # OCR = non-regulatory
        }
            
        # Create numeric labels
        df['label_numeric'] = df[label_col].str.lower().map(label_map)
        
        # Check for unplanned labels
        unmapped = df['label_numeric'].isna().sum()
        if unmapped > 0:
            logger.warning(f"  {unmapped} rows with unmapped labels")
            logger.warning(f"Unique labels found: {df[label_col].unique()}")
        
        return df
    
    def process_csv_complete(
        self,
        csv_file: str,
        sequence_col: str = 'sequence',
        label_col: str = 'label',
        output_file: str = None,
        remove_invalid: bool = True,
        min_length: int = 100,
        max_length: int = 1000
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            csv_file: Path to input CSV
            sequence_col: Column name with sequences
            label_col: Column name with labels
            output_file: Save processed data here (optional)
            remove_invalid: If True, remove invalid sequences
            min_length: Minimum sequence length (for validation)
            max_length: Maximum sequence length (for validation)
        
        Returns:
            Preprocessed DataFrame
        """
        
        logger.info(" DATA PREPROCESSING PIPELINE")
     
        
    
        logger.info(f"\n[1/6] Reading {csv_file}...")
        
        if not Path(csv_file).exists():
            raise FileNotFoundError(f" File not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        initial_count = len(df)
        
        logger.info(f"   Loaded {initial_count} sequences")
        
        # Verify expected columns
        if sequence_col not in df.columns:
            raise ValueError(f"Column '{sequence_col}' not found in CSV")
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found in CSV")
        
    
        logger.info(f"\n[2/6] Cleaning sequences...")
        
        df['sequence_clean'] = df[sequence_col].apply(self.clean_sequence)
        logger.info(f"   Cleaned {initial_count} sequences")
        
        
        logger.info(f"\n[3/6] Validating sequences...")
        
        df['is_valid_chars'] = df['sequence_clean'].apply(self.is_valid_sequence)
        df['is_valid_length'] = df['sequence_clean'].apply(
            lambda x: self.is_valid_length(x, min_length, max_length)
        )
        df['is_valid'] = df['is_valid_chars'] & df['is_valid_length']
        
        invalid_chars = (~df['is_valid_chars']).sum()
        invalid_len = (~df['is_valid_length']).sum()
        
        if invalid_chars > 0:
            logger.warning(f"   {invalid_chars} sequences with invalid characters")
        if invalid_len > 0:
            logger.warning(f"   {invalid_len} sequences with invalid length ({min_length}-{max_length} bp)")
        
        if remove_invalid:
            invalid_total = (~df['is_valid']).sum()
            if invalid_total > 0:
                logger.warning(f"  🗑️  Removing {invalid_total} invalid sequences")
                df = df[df['is_valid']].copy()
        
        logger.info(f"   Valid sequences: {len(df)}")
        
       
        logger.info(f"\n[4/6] Normalizing sequence length to {self.target_length} bp...")
        
        df['sequence_norm'] = df['sequence_clean'].apply(self.pad_or_trim_sequence)
        
        # Verify all sequences are correct length
        all_correct_length = (df['sequence_norm'].apply(len) == self.target_length).all()
        if all_correct_length:
            logger.info(f"   All {len(df)} sequences now {self.target_length} bp")
        else:
            logger.warning(f"    Some sequences not {self.target_length} bp")
        
        
        logger.info(f"\n[5/6] Calculating quality metrics...")
        
        df['gc_content'] = df['sequence_norm'].apply(self.calculate_gc_content)
        df['n_count'] = df['sequence_norm'].apply(self.count_n_bases)
        df['homopolymer_runs'] = df['sequence_norm'].apply(self.calculate_homopolymer_runs)
        
        logger.info(f"  Mean GC content:  {df['gc_content'].mean():.2f}%")
        logger.info(f"  Std dev GC:       {df['gc_content'].std():.2f}%")
        logger.info(f"  Mean N count:     {df['n_count'].mean():.2f}")
        

        logger.info(f"\n[6/6] Finalizing...")
        
        # Keep relevant columns
        columns_to_keep = [
            'sequence_norm',
            label_col,
            'gc_content',
            'n_count',
            'homopolymer_runs'
        ]
        
        #Add optional columns if they exist
        optional_cols = ['sequence_id', 'source', 'chromosome', 'start', 'end']
        for col in optional_cols:
            if col in df.columns:
                columns_to_keep.insert(1, col)  # Insert after sequence
        
        df = df[columns_to_keep].copy()
        
        # Rename sequence column
        df = df.rename(columns={'sequence_norm': 'sequence'})
        
        # Map string labels to numeric
        df = self.map_labels(df, label_col)

        # Preserve string labels explicitly
        df['label_name'] = df[label_col].str.lower()

        # Drop original string label column
        df = df.drop(columns=[label_col])

        # Rename numeric label column
        df = df.rename(columns={'label_numeric': 'label'})

        
        # Reorder columns nicely
        final_columns = ['sequence', 'label', 'label_name', 'gc_content', 'n_count', 'homopolymer_runs']
        
        # Add back optional columns
        for col in optional_cols:
            if col in df.columns:
                final_columns.insert(3, col)
        
        df = df[final_columns]
        
        logger.info(f"\n{'CLASS DISTRIBUTION':^80}")
        
        class_names = {0: 'Non-regulatory', 1: 'Promoter', 2: 'Enhancer'}
        class_dist = df['label'].value_counts().sort_index()
        
        for label, count in class_dist.items():
            pct = (count / len(df)) * 100
            class_name = class_names.get(label, f'Class {label}')
            bar_length = int(pct / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            logger.info(f"  {class_name:20s}: {count:6,d} ({pct:5.1f}%) {bar}")
        
        # SUMMARY 
        logger.info(f"\n{'SUMMARY':^80}")
        logger.info("-" * 80)
        logger.info(f"  Input rows:              {initial_count:,}")
        logger.info(f"  Output rows:             {len(df):,}")
        logger.info(f"  Rows removed:            {initial_count - len(df):,}")
        logger.info(f"  Sequence length:         {self.target_length} bp")
        logger.info(f"  Mean GC content:         {df['gc_content'].mean():.2f}%")
        logger.info(f"  Std GC content:          {df['gc_content'].std():.2f}%")
        logger.info(f"  Min/Max GC:              {df['gc_content'].min():.2f}% / {df['gc_content'].max():.2f}%")
        logger.info(f"  Mean N count:            {df['n_count'].mean():.2f}")
        logger.info(f"  Max N count:             {df['n_count'].max():.0f}")
        
        #  SAVE
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
            logger.info(f"\n Saved to: {output_file}")
            logger.info(f"   File size: {file_size_mb:.2f} MB")
        
      
        logger.info("PREPROCESSING COMPLETE")
      
        
        return df


def main():
    """
    Main execution function.
    
    Reads: grch38_regulatory_sequences.csv 
    Outputs: data/raw/preprocessed_dna.csv 
    """
    
    # Paths
    INPUT_CSV = "data_collection/raw_downloads/processed_sequences/grch38_regulatory_sequences.csv"
    OUTPUT_CSV = "data/raw/preprocessed_dna.csv"
    
    # Create preprocessor
    preprocessor = DataPreprocessor(target_length=256)
    
    try:
        # Run preprocessing
        df = preprocessor.process_csv_complete(
            csv_file=INPUT_CSV,
            sequence_col='sequence',
            label_col='label',
            output_file=OUTPUT_CSV,
            remove_invalid=True,
            min_length=100,
            max_length=1000
        )
        
        # Print first few rows
        logger.info("\n First 5 rows of preprocessed data:")
        
        print(df.head())
        
        # Print data types
        logger.info("\n Data types:")
        
        print(df.dtypes)
        
        return df
    
    except FileNotFoundError as e:
        logger.error(f" {e}")
        logger.error(f"\nMake sure you have run Step 5 (coordinate conversion) first:")
        logger.error(f"  python 0_data_collection/scripts/convert_coordinates_to_sequences.py")
        logger.error(f"\nThis should create: {INPUT_CSV}")
        return None
    
    except Exception as e:
        logger.error(f" Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    df = main()
    
    # Quick verification
    if df is not None:
        logger.info("\n Verification:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  All sequences 256 bp: {(df['sequence'].apply(len) == 256).all()}")
        logger.info(f"  Valid bases only: {df['sequence'].apply(lambda x: all(c in 'ATGCN' for c in x)).all()}")
        logger.info(f"  No duplicates: {df['sequence'].duplicated().sum() == 0}")    
            
            

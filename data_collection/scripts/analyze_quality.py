import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


class DataQualityAnalyzer:
    """Comprehensive data quality analysis and visualization."""
    
    def __init__(self, df: pd.DataFrame, sequence_col: str = 'sequence', label_col: str = 'label'):
        """
        Args:
            df: Input DataFrame with sequences and labels
            sequence_col: Column name with DNA sequences
            label_col: Column name with labels (numeric 0/1/2)
        """
        self.df = df.copy()
        self.sequence_col = sequence_col
        self.label_col = label_col
        
        # Add metrics if not present
        if 'gc_content' not in self.df.columns:
            self.df['gc_content'] = self.df[sequence_col].apply(
                lambda x: (x.count('G') + x.count('C')) / len(x) * 100 if len(x) > 0 else 0
            )
        
        # Label names for pretty printing
        self.label_names = {0: 'Non-regulatory', 1: 'Promoter', 2: 'Enhancer'}
    
    
    
    def generate_report(self) -> str:
        """Generate comprehensive text quality report."""
        report = []
       
        report.append(" DATA QUALITY ANALYSIS REPORT".center(90))
        
        
        # DATASET SIZE
        report.append(f"\n DATASET OVERVIEW")
        
        report.append(f"  Total sequences:         {len(self.df):,}")
        seq_lengths = self.df[self.sequence_col].apply(len)
        report.append(f"  Mean sequence length:    {seq_lengths.mean():.0f} bp")
        report.append(f"  Min/Max length:          {seq_lengths.min()}/{seq_lengths.max()} bp")
        report.append(f"  Std dev length:          {seq_lengths.std():.2f} bp")
        
        # Check if all sequences same length
        if seq_lengths.std() == 0:
            report.append(f"   All sequences uniform length")
        else:
            report.append(f"    Variable sequence lengths")
        
        # CLASS DISTRIBUTION
        report.append(f"  CLASS DISTRIBUTION")
        report.append("-" * 90)
        class_counts = self.df[self.label_col].value_counts().sort_index()
        
        total = len(self.df)
        for label in sorted(class_counts.index):
            count = class_counts[label]
            pct = (count / total) * 100
            class_name = self.label_names.get(label, f'Class {label}')
            
            # Check balance
            if 30 < pct < 40:
                balance = " Well balanced"
            elif 25 < pct < 50:
                balance = "✓ Acceptable"
            else:
                balance = " Imbalanced"
            
            report.append(f"  {class_name:20s}: {count:7,d} ({pct:5.1f}%) {balance}")
        
        # SEQUENCE COMPOSITION (sample)
        report.append(f"\n SEQUENCE COMPOSITION (Sample)")
        report.append("-" * 90)
        sample_seq = self.df[self.sequence_col].iloc[0]
        total_bases = len(sample_seq)
        
        for base in ['A', 'T', 'G', 'C', 'N']:
            count = sample_seq.count(base)
            pct = (count / total_bases) * 100
            report.append(f"  {base}: {pct:6.2f}% ({count:4d} bases)")
        
        # GC CONTENT ANALYSIS
        report.append(f"\n GC CONTENT ANALYSIS")
        report.append("-" * 90)
        gc = self.df['gc_content']
        report.append(f"  Mean GC content:         {gc.mean():.2f}%")
        report.append(f"  Std dev GC:              {gc.std():.2f}%")
        report.append(f"  Min/Max GC:              {gc.min():.2f}% / {gc.max():.2f}%")
        
        # Count in healthy range
        healthy = ((gc >= 40) & (gc <= 60)).sum()
        healthy_pct = (healthy / len(self.df)) * 100
        report.append(f"  In range 40-60%:         {healthy:,d} ({healthy_pct:.1f}%)")
        
        if healthy_pct > 90:
            report.append(f"   Good GC distribution")
        elif healthy_pct > 70:
            report.append(f"   Acceptable GC distribution")
        else:
            report.append(f"   GC distribution skewed")
        
        # QUALITY CHECKS
        report.append(f"\n QUALITY CHECKS")
        report.append("-" * 90)
        
        # Invalid sequences
        invalid = self.df[self.sequence_col].apply(
            lambda x: not all(c in 'ATGCN' for c in x)
        ).sum()
        report.append(f"  Invalid sequences:       {invalid:,d} {' PASS' if invalid == 0 else ' FAIL'}")
        
        # Duplicates
        duplicates = self.df[self.sequence_col].duplicated().sum()
        report.append(f"  Duplicate sequences:     {duplicates:,d} {' PASS' if duplicates == 0 else ' FAIL'}")
        
        # N count
        if 'n_count' in self.df.columns:
            high_n = (self.df['n_count'] > 10).sum()
            report.append(f"  Sequences with >10 N:    {high_n:,d} sequences")
        
        # Homopolymer runs
        if 'homopolymer_runs' in self.df.columns:
            high_homo = (self.df['homopolymer_runs'] > 2).sum()
            report.append(f"  High homopolymer runs:   {high_homo:,d} sequences")
        
        # CONFIDENCE SCORE
        report.append(f"\n OVERALL QUALITY SCORE")
        report.append("-" * 90)
        
        score = 100
        if invalid > 0: score -= 20
        if duplicates > 0: score -= 10
        if healthy_pct < 70: score -= 10
        if seq_lengths.std() > 0: score -= 5
        
        if score >= 90:
            quality = "***** Excellent"
        elif score >= 80:
            quality = "**** Good"
        elif score >= 70:
            quality = "*** Acceptable"
        elif score >= 60:
            quality = "** Marginal"
        else:
            quality = "* Poor"
        
        report.append(f"  Quality Score: {score}/100 - {quality}")
        report.append(f"\n   READY FOR MODULE 1" if score >= 70 else f"\n    Review data before proceeding")
        
        report.append("\n" + "=" * 90 + "\n")
        
        return "\n".join(report)
    
    
    
    def plot_class_distribution(self, save_path: str = None) -> str:
        """Plot class distribution bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_counts = self.df[self.label_col].value_counts().sort_index()
        class_labels = [self.label_names.get(i, f'Class {i}') for i in class_counts.index]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(class_counts)]
        
        bars = ax.bar(range(len(class_counts)), class_counts.values, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('DNA Sequence Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels(class_labels)
        
        # Add value labels on bars
        for bar, value in zip(bars, class_counts.values):
            pct = value / len(self.df) * 100
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{int(value)}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f" Saved: {save_path}")
        
        plt.close()
        return save_path or 'class_distribution.png'
    
    def plot_gc_content(self, save_path: str = None) -> str:
        """Plot GC content distribution by class."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram by class
        for label in sorted(self.df[self.label_col].unique()):
            mask = self.df[self.label_col] == label
            class_name = self.label_names.get(label, f'Class {label}')
            axes[0].hist(self.df[mask]['gc_content'], alpha=0.6, 
                        label=class_name, bins=30, edgecolor='black')
        
        axes[0].set_xlabel('GC Content (%)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('GC Content Distribution by Class', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Boxplot by class
        data_for_box = [self.df[self.df[self.label_col] == label]['gc_content'].values
                       for label in sorted(self.df[self.label_col].unique())]
        box_labels = [self.label_names.get(label, f'Class {label}') 
                     for label in sorted(self.df[self.label_col].unique())]
        
        bp = axes[1].boxplot(data_for_box, labels=box_labels, patch_artist=True)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors[:len(data_for_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('GC Content (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('GC Content by Class (Boxplot)', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f" Saved: {save_path}")
        
        plt.close()
        return save_path or 'gc_content.png'
    
    def plot_sequence_length(self, save_path: str = None) -> str:
        """Plot sequence length distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for label in sorted(self.df[self.label_col].unique()):
            mask = self.df[self.label_col] == label
            lengths = self.df[mask][self.sequence_col].apply(len)
            class_name = self.label_names.get(label, f'Class {label}')
            ax.hist(lengths, alpha=0.6, label=class_name, bins=20, edgecolor='black')
        
        ax.set_xlabel('Sequence Length (bp)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Sequence Length Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f" Saved: {save_path}")
        
        plt.close()
        return save_path or 'sequence_length.png'


def main():
    """Main execution function."""
    
    
    logger.info(" DATA QUALITY ANALYSIS & VISUALIZATION")
 
    
    # Load preprocessed data
    INPUT_CSV = "data/raw/preprocessed_dna.csv"
    OUTPUT_DIR = "results"
    
    logger.info(f"\nLoading data from: {INPUT_CSV}")
    
    try:
        df = pd.read_csv(INPUT_CSV)
        logger.info(f" Loaded {len(df)} sequences\n")
    except FileNotFoundError:
        logger.error(f" File not found: {INPUT_CSV}")
        logger.error(f"Make sure preprocessing script has run successfully")
        return
    
    # Create analyzer
    analyzer = DataQualityAnalyzer(df)
    
    # Generate text report
    logger.info("Generating quality report...\n")
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    report_path = Path(OUTPUT_DIR) / "quality_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f" Report saved to: {report_path}\n")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    analyzer.plot_class_distribution(f"{OUTPUT_DIR}/class_distribution.png")
    analyzer.plot_gc_content(f"{OUTPUT_DIR}/gc_content.png")
    analyzer.plot_sequence_length(f"{OUTPUT_DIR}/sequence_length.png")
    
    logger.info(f"\n All visualizations saved to: {OUTPUT_DIR}/")
    
    
    logger.info(" QUALITY ANALYSIS COMPLETE")
 
    logger.info(f"\nGenerated files:")
    logger.info(f"  • {OUTPUT_DIR}/class_distribution.png")
    logger.info(f"  • {OUTPUT_DIR}/gc_content.png")
    logger.info(f"  • {OUTPUT_DIR}/sequence_length.png")
    logger.info(f"  • {OUTPUT_DIR}/quality_report.txt\n")


if __name__ == "__main__":
    main()

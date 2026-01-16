import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class FeatureAnalyzer:
    """Analyze k-mer features and class discrimination."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
    
    def load_features_and_labels(self, split: str = 'train', k: int = 3):
        """Load k-mer features and labels."""
        X = np.load(self.data_dir / f"{split}_kmers_k{k}.npy")
        y = np.load(self.data_dir / f"{split}_labels.npy")
        return X, y
    
    def plot_feature_statistics(self, k: int = 3):
        """Plot feature statistics across splits."""
        logger.info(f"Creating feature statistics plot for k={k}...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, split in enumerate(['train', 'val', 'test']):
            X, y = self.load_features_and_labels(split, k)
            
            axes[idx].hist(X.flatten(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel('Feature Value (Normalized Frequency)')
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(f'{split.upper()} - k={k}')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / f'feature_distribution_k{k}.png', dpi=150)
        logger.info(f" Saved: feature_distribution_k{k}.png")
        plt.close()
    
    def plot_class_separation(self, k: int = 3):
        """Plot feature means by class."""
        logger.info(f"Creating class separation plot for k={k}...")
        
        X_train, y_train = self.load_features_and_labels('train', k)
        
        # Calculate mean feature values per class
        class_names = {0: 'Non-reg', 1: 'Promoter', 2: 'Enhancer'}
        class_means = {}
        
        for label in [0, 1, 2]:
            mask = y_train == label
            class_means[class_names[label]] = X_train[mask].mean(axis=0)
        
        # Plot top 20 features by variance
        all_means = np.array(list(class_means.values()))
        feature_variance = all_means.var(axis=0)
        top_indices = np.argsort(feature_variance)[-20:]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(top_indices))
        width = 0.25
        
        for idx, (class_name, means) in enumerate(class_means.items()):
            ax.bar(x + idx * width, means[top_indices], width, label=class_name)
        
        ax.set_xlabel('Feature Index (top 20 by variance)')
        ax.set_ylabel('Mean Normalized Frequency')
        ax.set_title(f'Top Discriminative Features (k={k})')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / f'class_separation_k{k}.png', dpi=150)
        logger.info(f" Saved: class_separation_k{k}.png")
        plt.close()
    
    def plot_sparsity(self):
        """Plot sparsity across k values and splits."""
        logger.info("Creating sparsity analysis...")
        
        k_values = [1, 2, 3, 4, 5]
        results = []
        
        for k in k_values:
            for split in ['train', 'val', 'test']:
                X, _ = self.load_features_and_labels(split, k)
                sparsity = (X == 0).sum() / X.size
                results.append({'k': k, 'split': split, 'sparsity': sparsity})
        
        df = pd.DataFrame(results)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for split in ['train', 'val', 'test']:
            split_data = df[df['split'] == split]
            ax.plot(split_data['k'], split_data['sparsity'], marker='o', label=split, linewidth=2)
        
        ax.set_xlabel('k (k-mer length)')
        ax.set_ylabel('Sparsity (fraction of zeros)')
        ax.set_title('Feature Sparsity vs k-mer Length')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'sparsity_analysis.png', dpi=150)
        logger.info(f" Saved: sparsity_analysis.png")
        plt.close()


def main():
    """Run feature analysis."""
   
    logger.info("FEATURE ANALYSIS & VISUALIZATION")
    
    
    analyzer = FeatureAnalyzer()
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    
    for k in [1, 2, 3, 4, 5]:
        analyzer.plot_feature_statistics(k=k)
        if k <= 3:  # Only for smaller k (otherwise too many features)
            analyzer.plot_class_separation(k=k)
    
    analyzer.plot_sparsity()
  
    logger.info(" ANALYSIS COMPLETE")
    
    logger.info("\nGenerated plots:")
    logger.info("  • feature_distribution_k*.png")
    logger.info("  • class_separation_k*.png")
    logger.info("  • sparsity_analysis.png\n")


if __name__ == "__main__":
    main()

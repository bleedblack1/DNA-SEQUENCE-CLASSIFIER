"""
Comprehensive evaluation of all models (PyTorch only).
Loads trained models, evaluates on test set, creates comparison plots.
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


# ---------------------------------------------------------------------
# CNN MODEL DEFINITION (must match training exactly)
# ---------------------------------------------------------------------
class DNACNNModel(nn.Module):
    def __init__(self, n_bases=4, n_classes=3):
        super().__init__()

        self.conv1 = nn.Conv1d(n_bases, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3_k5 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv3_k7 = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.conv3_k11 = nn.Conv1d(128, 64, kernel_size=11, padding=5)

        self.bn3 = nn.BatchNorm1d(192)
        self.pool3 = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(192, 192)
        self.fc2 = nn.Linear(192, 128)
        self.fc3 = nn.Linear(128, n_classes)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))

        x = torch.cat(
            [self.conv3_k5(x), self.conv3_k7(x), self.conv3_k11(x)],
            dim=1
        )

        x = self.pool3(self.relu(self.bn3(x)))
        x = x.view(x.size(0), -1)

        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        return self.fc3(x)


# ---------------------------------------------------------------------
# EVALUATOR
# ---------------------------------------------------------------------
class ModelEvaluator:
    def __init__(self, models_dir="modeling/models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.models_dir = Path(models_dir)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        self.metrics = {}

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------
    def load_models(self):
        logger.info("Loading trained models...")

        self.baseline_models = {}

        for name in ["logistic_regression", "random_forest", "xgboost"]:
            path = self.models_dir / f"{name}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self.baseline_models[name] = pickle.load(f)
                logger.info(f" Loaded {name}")
            else:
                logger.warning(f" Missing {name}")

        cnn_path = self.models_dir / "cnn_model.pt"
        if cnn_path.exists():
            self.cnn = DNACNNModel(n_classes=3).to(self.device)
            self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
            self.cnn.eval()
            logger.info(" Loaded CNN (PyTorch)")
        else:
            logger.warning(" Missing CNN model")

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------
    def load_test_data(self):
        logger.info("Loading test data...")

        self.X_test_kmers = np.load("data/processed/test_kmers_k3.npy")
        self.X_test_seq = np.load("data/processed/test_sequences.npy")
        self.y_test = np.load("data/processed/test_labels.npy")

        self.X_test_seq = torch.FloatTensor(self.X_test_seq.transpose(0, 2, 1))
        self.y_test_torch = torch.LongTensor(self.y_test)

        logger.info(f"k-mers: {self.X_test_kmers.shape}")
        logger.info(f"Sequences: {self.X_test_seq.shape}")
        logger.info(f"Labels: {self.y_test.shape}")

    # ---------------------------------------------------------
    # Evaluate all models
    # ---------------------------------------------------------
    def evaluate(self):
        logger.info("Evaluating models...")

        # Baselines
        for name, model in self.baseline_models.items():
            y_pred = model.predict(self.X_test_kmers)
            y_proba = model.predict_proba(self.X_test_kmers)

            self.metrics[name] = self._compute_metrics(y_pred, y_proba)

        # CNN
        if hasattr(self, "cnn"):
            loader = DataLoader(
                TensorDataset(self.X_test_seq),
                batch_size=64
            )

            probs = []
            with torch.no_grad():
                for (x,) in loader:
                    x = x.to(self.device)
                    probs.append(torch.softmax(self.cnn(x), dim=1).cpu().numpy())

            y_proba = np.vstack(probs)
            y_pred = y_proba.argmax(axis=1)

            self.metrics["cnn"] = self._compute_metrics(y_pred, y_proba)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    def _compute_metrics(self, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1_macro": f1_score(self.y_test, y_pred, average="macro"),
            "auc": roc_auc_score(self.y_test, y_proba, multi_class="ovr"),
            "preds": y_pred
        }

    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------
    def plot_results(self):
        names = list(self.metrics.keys())
        accs = [self.metrics[n]["accuracy"] for n in names]

        # Accuracy comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, accs, edgecolor="black")
        plt.ylabel("Accuracy")
        plt.title("Model Comparison (Test Set)")
        plt.ylim(0, 1)

        for bar, acc in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width()/2, acc, f"{acc:.3f}",
                     ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(self.results_dir / "all_models_accuracy.png", dpi=150)
        plt.close()

        # Best model confusion matrix
        best = max(self.metrics.items(), key=lambda x: x[1]["accuracy"])
        name, data = best

        cm = confusion_matrix(self.y_test, data["preds"])

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-reg", "Promoter", "Enhancer"],
            yticklabels=["Non-reg", "Promoter", "Enhancer"]
        )
        plt.title(f"Confusion Matrix: {name.upper()}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.results_dir / "best_model_confusion_matrix.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    
    logger.info(" COMPREHENSIVE MODEL EVALUATION (PyTorch ONLY)")


    evaluator = ModelEvaluator()
    evaluator.load_models()
    evaluator.load_test_data()
    evaluator.evaluate()
    evaluator.plot_results()

    logger.info("\nModel Performance Summary:")
    for name, m in sorted(evaluator.metrics.items(),
                          key=lambda x: x[1]["accuracy"], reverse=True):
        logger.info(f"{name:25s} | Acc={m['accuracy']:.4f} | F1={m['f1_macro']:.4f} | AUC={m['auc']:.4f}")

    best = max(evaluator.metrics.items(), key=lambda x: x[1]["accuracy"])
    logger.info(f"\n Best Model: {best[0].upper()} ({best[1]['accuracy']:.4f})")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

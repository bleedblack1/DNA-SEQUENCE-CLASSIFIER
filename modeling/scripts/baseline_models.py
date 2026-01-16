import numpy as np
import pickle
from pathlib import Path
import logging
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import xgboost as xgb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaselineModelTrainer:
    """Train and evaluate baseline ML models on k-mer features."""

    def __init__(self, output_dir: str = "modeling/models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.metrics = {}

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    def load_kmer_data(self, k: int = 3):
        logger.info(f"Loading k={k} k-mer features...")

        X_train = np.load(f"data/processed/train_kmers_k{k}.npy")
        y_train = np.load("data/processed/train_labels.npy")

        X_val = np.load(f"data/processed/val_kmers_k{k}.npy")
        y_val = np.load("data/processed/val_labels.npy")

        X_test = np.load(f"data/processed/test_kmers_k{k}.npy")
        y_test = np.load("data/processed/test_labels.npy")

        logger.info(f"Train: {X_train.shape}")
        logger.info(f"Val:   {X_val.shape}")
        logger.info(f"Test:  {X_test.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # --------------------------------------------------
    # Models
    # --------------------------------------------------
    def train_logistic_regression(self, X_train, y_train):
        logger.info("Training Logistic Regression...")

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=2.0,
            class_weight="balanced",
            max_iter=3000,
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)
        self.models["logistic_regression"] = model
        return model

    def train_random_forest(self, X_train, y_train):
        logger.info("Training Random Forest...")

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_split=16,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)
        self.models["random_forest"] = model
        return model

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        logger.info("Training XGBoost...")

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            predictor="cpu_predictor" 
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        self.models["xgboost"] = model
        return model

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    def evaluate_model(self, name, model, X_train, y_train, X_val, y_val, X_test, y_test):
        logger.info(f"Evaluating {name}...")

        def compute_metrics(X, y):
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)

            return {
                "accuracy": accuracy_score(y, y_pred),
                "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
                "auc_macro": roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
            }

        self.metrics[name] = {
            "train": compute_metrics(X_train, y_train),
            "val": compute_metrics(X_val, y_val),
            "test": compute_metrics(X_test, y_test)
        }

        logger.info(
            f"{name} | Test Acc: {self.metrics[name]['test']['accuracy']:.4f} "
            f"| Test F1: {self.metrics[name]['test']['f1_macro']:.4f}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    def save_models(self):
        logger.info("Saving models and metrics...")

        for name, model in self.models.items():
            path = self.output_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Saved {path}")

        metrics_path = self.output_dir / "baseline_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_path}")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    def plot_comparison(self):
        logger.info("Creating comparison plots...")

        models = list(self.metrics.keys())
        acc = [self.metrics[m]["test"]["accuracy"] for m in models]
        f1 = [self.metrics[m]["test"]["f1_macro"] for m in models]
        auc = [self.metrics[m]["test"]["auc_macro"] for m in models]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].bar(models, acc)
        axes[0].set_title("Test Accuracy")
        axes[0].set_ylim(0, 1)

        axes[1].bar(models, f1)
        axes[1].set_title("Test F1 (Macro)")
        axes[1].set_ylim(0, 1)

        axes[2].bar(models, auc)
        axes[2].set_title("Test AUC (Macro)")
        axes[2].set_ylim(0, 1)

        plt.tight_layout()
        out_path = self.output_dir.parent / "baseline_comparison.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        logger.info(f"Saved plot: {out_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    logger.info("BASELINE MODEL TRAINING PIPELINE")

    trainer = BaselineModelTrainer()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.load_kmer_data(k=3)

    #  CRITICAL: SCALE FEATURES FOR LOGISTIC REGRESSION
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    lr = trainer.train_logistic_regression(X_train_scaled, y_train)
    rf = trainer.train_random_forest(X_train, y_train)
    xgb_model = trainer.train_xgboost(X_train, y_train, X_val, y_val)

    trainer.evaluate_model(
        "logistic_regression",
        lr,
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test
    )

    trainer.evaluate_model(
        "random_forest",
        rf,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

    trainer.evaluate_model(
        "xgboost",
        xgb_model,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

    trainer.save_models()
    trainer.plot_comparison()

    logger.info("COMPLETE")


if __name__ == "__main__":
    main()

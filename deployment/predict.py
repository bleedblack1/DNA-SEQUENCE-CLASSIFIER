import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XGBOOST_BUILD_WITH_CUDA"] = "0"

import numpy as np
import pickle
from pathlib import Path
import logging
import torch
from typing import Dict, List
import sys


# ABSOLUTE PROJECT ROOT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modeling.scripts.train_cnn_pytorch import DNACNNModel  # noqa: E402


# LOGGING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SAFE PICKLE LOADER 

def safe_load_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"   Failed to load {path.name}: {e}")
        return None


class PredictionPipeline:
    """Unified prediction pipeline for trained DNA models."""

    def __init__(self, models_dir: str | None = None):
        self.device = torch.device("cpu")  

        self.class_labels = ["Non-regulatory", "Promoter", "Enhancer"]

        if models_dir is None:
            self.models_dir = PROJECT_ROOT / "modeling" / "models"
        else:
            self.models_dir = Path(models_dir).resolve()

        logger.info(f"Looking for models in: {self.models_dir}")

        self.baseline_models = {}
        self.cnn_model = None
        self._load_models()

        logger.info(" Prediction pipeline initialized")

    
    def _load_models(self):
        logger.info("Loading models...")

        for model_name in ["logistic_regression", "random_forest", "xgboost"]:
            path = self.models_dir / f"{model_name}.pkl"

            if not path.exists():
                logger.warning(f"   Missing {model_name}.pkl")
                continue

            model = safe_load_pickle(path)
            if model is None:
                continue

            self.baseline_models[model_name] = model
            logger.info(f"   Loaded {model_name}")

        cnn_path = self.models_dir / "cnn_model.pt"
        logger.info(f"Looking for CNN model at: {cnn_path}")
        logger.info(f"CNN exists: {cnn_path.exists()}")

        if cnn_path.exists():
            self.cnn_model = DNACNNModel(n_bases=4, n_classes=3)
            self.cnn_model.load_state_dict(
                torch.load(cnn_path, map_location="cpu")
            )
            self.cnn_model.eval()
            logger.info("   Loaded PyTorch CNN")
        else:
            logger.warning(" CNN model not found")
            self.cnn_model = None

    
    def dna_to_onehot(self, sequence: str) -> np.ndarray:
        sequence = sequence.upper()
        base_map = {"A": 0, "T": 1, "G": 2, "C": 3}

        if len(sequence) < 256:
            sequence += "N" * (256 - len(sequence))
        else:
            sequence = sequence[:256]

        onehot = np.zeros((256, 4), dtype=np.float32)
        for i, base in enumerate(sequence):
            if base in base_map:
                onehot[i, base_map[base]] = 1.0
            else:
                onehot[i, :] = 0.25

        return onehot

    def dna_to_kmers(self, sequence: str, k: int = 3) -> np.ndarray:
        sequence = sequence.upper()
        base_map = {"A": 0, "T": 1, "G": 2, "C": 3}

        if len(sequence) < 256:
            sequence += "N" * (256 - len(sequence))
        else:
            sequence = sequence[:256]

        counts = np.zeros(4 ** k, dtype=np.float32)

        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            if all(b in base_map for b in kmer):
                idx = 0
                for b in kmer:
                    idx = idx * 4 + base_map[b]
                counts[idx] += 1

        if counts.sum() > 0:
            counts /= counts.sum()

        return counts

    
    def predict_with_cnn(self, sequence: str):
        if self.cnn_model is None:
            return None, None

        x = torch.tensor(
            self.dna_to_onehot(sequence)
        ).unsqueeze(0).transpose(1, 2)

        with torch.no_grad():
            probs = torch.softmax(self.cnn_model(x), dim=1).numpy()[0]
            pred = int(np.argmax(probs))

        return pred, probs

    def predict_with_baseline(self, sequence: str, model_name: str):
        model = self.baseline_models.get(model_name)
        if model is None:
            return None, None

        kmers = self.dna_to_kmers(sequence)
        pred = int(model.predict([kmers])[0])
        probs = model.predict_proba([kmers])[0]

        return pred, probs

    def predict(self, sequence: str, use_model: str = "cnn") -> Dict:
        sequence = sequence.strip().upper()

        if len(sequence) < 50:
            return {"error": "Sequence too short (min 50 bp)"}

        if len(sequence) > 5000:
            return {"error": "Sequence too long (max 5000 bp)"}

        if use_model == "cnn":
            pred, probs = self.predict_with_cnn(sequence)
        else:
            pred, probs = self.predict_with_baseline(sequence, use_model)

        if pred is None:
            return {"error": f"Model {use_model} not available"}

        return {
            "sequence_length": len(sequence),
            "model_used": use_model,
            "prediction": self.class_labels[pred],
            "confidence": float(probs[pred]),
            "probabilities": {
                self.class_labels[i]: float(probs[i])
                for i in range(len(self.class_labels))
            },
        }


if __name__ == "__main__":
    pipeline = PredictionPipeline()
    test_seq = "ATG" * 120

    print("\nCNN TEST")
    print(pipeline.predict(test_seq, use_model="cnn"))

    print("\nXGBOOST TEST")
    print(pipeline.predict(test_seq, use_model="xgboost"))

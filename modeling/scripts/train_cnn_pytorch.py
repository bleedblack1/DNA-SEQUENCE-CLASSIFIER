import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import logging

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def log_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    logger.info(f"\n{name} CONFUSION MATRIX (raw):\n{cm}")
    logger.info(f"\n{name} CONFUSION MATRIX (normalized):\n{np.round(cm_norm, 3)}")


def find_thresholds(y_true, y_proba):
    thresholds = []
    for c in range(y_proba.shape[1]):
        fpr, tpr, thr = roc_curve((y_true == c).astype(int), y_proba[:, c])
        thresholds.append(thr[np.argmax(tpr - fpr)])
    return np.array(thresholds)

def apply_thresholds(y_proba, thresholds):
    scores = y_proba / thresholds
    return scores.argmax(axis=1)


class CNNTrainer:
    def __init__(self, output_dir="modeling/models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None

    def load_data(self):
        X_train = np.load("data/processed/train_sequences.npy")
        y_train = np.load("data/processed/train_labels.npy")
        X_val = np.load("data/processed/val_sequences.npy")
        y_val = np.load("data/processed/val_labels.npy")
        X_test = np.load("data/processed/test_sequences.npy")
        y_test = np.load("data/processed/test_labels.npy")

        X_train = torch.FloatTensor(X_train.transpose(0, 2, 1))
        X_val = torch.FloatTensor(X_val.transpose(0, 2, 1))
        X_test = torch.FloatTensor(X_test.transpose(0, 2, 1))

        return (
            (X_train, torch.LongTensor(y_train)),
            (X_val, torch.LongTensor(y_val)),
            (X_test, torch.LongTensor(y_test)),
        )

    def build_model(self, n_classes=3):
        self.model = DNACNNModel(n_classes=n_classes).to(self.device)
        logger.info(self.model)
        logger.info(f"Trainable params: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train, val, epochs=50, batch_size=32):
        X_train, y_train = train
        X_val, y_val = val

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size
        )

        counts = np.bincount(y_train.numpy())
        weights = torch.tensor(
            (1.0 / counts) / (1.0 / counts).sum(),
            dtype=torch.float32
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=0.05
        )

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=5e-4,
            weight_decay=1e-4
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5
        )

        best_val = float("inf")
        patience, counter = 15, 0

        for epoch in range(epochs):
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    val_loss += criterion(self.model(x), y).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Val Loss {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    logger.info("Early stopping")
                    break

        self.model.load_state_dict(torch.load(self.output_dir / "best_model.pt"))

    def predict_proba(self, X):
        self.model.eval()
        probs = []
        loader = DataLoader(TensorDataset(X), batch_size=64)
        with torch.no_grad():
            for (x,) in loader:
                x = x.to(self.device)
                probs.append(torch.softmax(self.model(x), dim=1).cpu().numpy())
        return np.vstack(probs)


def main():
    trainer = CNNTrainer()

    train, val, test = trainer.load_data()
    trainer.build_model(n_classes=3)
    trainer.train(train, val)

 
    test_probs = trainer.predict_proba(test[0])
    test_preds = test_probs.argmax(axis=1)

    logger.info(
        f"TEST PHASE-1 | "
        f"Acc {accuracy_score(test[1], test_preds):.4f} | "
        f"F1 {f1_score(test[1], test_preds, average='macro'):.4f} | "
        f"AUC {roc_auc_score(test[1], test_probs, multi_class='ovr'):.4f}"
    )

    log_confusion_matrix(test[1].numpy(), test_preds, "TEST PHASE-1")

    
    val_probs = trainer.predict_proba(val[0])
    thresholds = find_thresholds(val[1].numpy(), val_probs)

    tuned_preds = apply_thresholds(test_probs, thresholds)

    logger.info(
        f"TEST PHASE-2 | "
        f"Acc {accuracy_score(test[1], tuned_preds):.4f} | "
        f"F1 {f1_score(test[1], tuned_preds, average='macro'):.4f}"
    )

    log_confusion_matrix(test[1].numpy(), tuned_preds, "TEST PHASE-2")

    logger.info("TRAINING + EVALUATION COMPLETE")

if __name__ == "__main__":
    main()

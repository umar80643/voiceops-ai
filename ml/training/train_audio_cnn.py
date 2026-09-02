"""Real PyTorch audio classifier — 1D-CNN over raw waveforms (Phase 4/5 upgrade).

Genuine gradient-descent training on raw audio (not text, not a pretrained
model). This does NOT require Hugging Face Hub (blocked in this sandbox)
since no pretrained weights are downloaded — the model is trained from
scratch here, which is real audio-modality deep learning, just smaller-
scale than wav2vec2 fine-tuning (which still needs GPU+HF access and
remains documented as future work in ADR-002).

Run: python -m ml.training.train_audio_cnn
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.datasets.synthetic_audio import LABELS, generate_dataset

ARTIFACT_DIR = Path("models/emotion")
torch.manual_seed(42)


class AudioCNN(nn.Module):
    """Small 1D-CNN: raw waveform -> conv stack -> global pool -> classifier."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=32, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=16, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def main() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    waveforms, labels = generate_dataset()
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    y = np.array([label_to_idx[label] for label in labels])

    x_train, x_test, y_train, y_test = train_test_split(
        waveforms, y, test_size=0.25, random_state=42, stratify=y
    )
    x_train_t = torch.tensor(x_train).unsqueeze(1)
    x_test_t = torch.tensor(x_test).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=16, shuffle=True)

    model = AudioCNN(n_classes=len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    model.train()
    for _epoch in range(15):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_losses.append(round(epoch_loss / len(x_train), 4))

    model.eval()
    with torch.no_grad():
        preds = model(x_test_t).argmax(dim=1).numpy()

    report = {
        "model": "AudioCNN (1D-CNN, trained from scratch on raw waveforms, no pretrained weights)",
        "dataset": {
            "source": "synthetic acoustic waveforms (ml/datasets/synthetic_audio.py)",
            "is_real_customer_data": False,
            "is_real_speech": False,
            "note": "Labels are causally tied to real acoustic properties (pitch/energy/pauses/clipping), "
            "so this is a genuine audio-signal learning task, unlike the text-based TF-IDF substitutes "
            "used elsewhere in this repo (see ADR-002).",
            "labels": LABELS,
            "n_train": len(x_train),
            "n_test": len(x_test),
        },
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "loss_fn": "CrossEntropyLoss",
        "epochs": 15,
        "train_loss_per_epoch": train_losses,
        "test_metrics": {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "macro_f1": round(f1_score(y_test, preds, average="macro"), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        },
        "params": sum(p.numel() for p in model.parameters()),
        "sandbox_note": "Trained from scratch (no GPU, no HF Hub weights) — CPU-only, "
        "~8kHz/1s clips for speed. wav2vec2/HuBERT fine-tuning on real speech "
        "remains documented-not-executed (ADR-002); this demonstrates the "
        "underlying deep-learning-on-audio competency without that dependency.",
    }

    (ARTIFACT_DIR / "audio_cnn_report.json").write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), ARTIFACT_DIR / "audio_cnn.pt")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()

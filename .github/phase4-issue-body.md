## Objective

Implement deep learning approach using spectrograms and CNNs for bearing fault classification.

## Dependencies

- #2 Phase 1: Data Pipeline Foundation

## Tasks

- [ ] Implement STFT spectrogram generation
- [ ] Implement CWT (Continuous Wavelet Transform) option
- [ ] Create 2D CNN for spectrogram classification
- [ ] Create 1D CNN for raw signal classification
- [ ] Implement PyTorch training loop with logging
- [ ] Add learning rate scheduling and early stopping
- [ ] Save model checkpoints

## Technical Details

### Spectrogram Generation

```python
# src/spectrogram.py
import numpy as np
import librosa

def compute_stft_spectrogram(segment: np.ndarray, fs: int = 12000,
                              n_fft: int = 256, hop_length: int = 64) -> np.ndarray:
    """Compute STFT magnitude spectrogram."""
    stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    # Convert to log scale
    log_spec = librosa.amplitude_to_db(magnitude, ref=np.max)
    return log_spec

def resize_spectrogram(spec: np.ndarray, target_size: tuple = (128, 128)) -> np.ndarray:
    """Resize spectrogram to fixed dimensions."""
    from skimage.transform import resize
    return resize(spec, target_size, mode='reflect', anti_aliasing=True)
```

### 2D CNN Architecture

```python
# src/models.py
import torch
import torch.nn as nn

class CNN2D(nn.Module):
    """2D CNN for spectrogram-based classification."""

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
```

### 1D CNN Architecture

```python
class CNN1D(nn.Module):
    """1D CNN for raw signal classification."""

    def __init__(self, n_classes: int = 4, input_length: int = 2048):
        super().__init__()
        self.features = nn.Sequential(
            # Large kernel to capture low frequencies
            nn.Conv1d(1, 64, kernel_size=64, stride=8),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=32, stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=16, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
```

### Training Loop

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return total_loss / len(dataloader), correct / total
```

## Verification

```bash
python src/train.py --model cnn2d --epochs 50 --batch-size 32
# Expected: Test Accuracy > 97%
```

## Definition of Done

- [ ] Spectrograms generated and visualized
- [ ] Both CNN architectures implemented
- [ ] Training converges with decreasing loss
- [ ] Test accuracy > 97%
- [ ] Model checkpoints saved

## Files to Create/Modify

| File | Description |
|------|-------------|
| `src/spectrogram.py` | Spectrogram generation |
| `src/models.py` | Add CNN architectures |
| `src/train.py` | Add PyTorch training loop |
| `src/dataset.py` | PyTorch Dataset classes |

---
Parent: #1

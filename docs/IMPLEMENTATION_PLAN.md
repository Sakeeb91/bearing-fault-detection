# Implementation Plan: Bearing Fault Detection System

## Expert Role: Signal Processing ML Engineer

This role is optimal because the project combines:
- Vibration signal processing (filtering, envelope analysis, spectrograms)
- Feature engineering from time and frequency domains
- Classical machine learning pipelines
- Deep learning for pattern recognition

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
├─────────────────────────────────────────────────────────────────────┤
│  CWRU .mat Files                                                    │
│  ├── Normal: 97.mat, 98.mat, 99.mat, 100.mat (0-3 HP)              │
│  ├── Inner Race: 105-108.mat, 169-172.mat, 209-212.mat             │
│  ├── Outer Race: 130-133.mat, 197-200.mat, 234-237.mat             │
│  └── Ball Fault: 118-121.mat, 185-188.mat, 222-225.mat             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  data_loader.py          │  preprocessing.py                        │
│  ├── load_mat_file()     │  ├── segment_signal()                   │
│  ├── parse_filename()    │  ├── bandpass_filter()                  │
│  └── create_dataset()    │  └── normalize()                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│   FEATURE EXTRACTION      │   │   SPECTROGRAM GENERATION  │
├───────────────────────────┤   ├───────────────────────────┤
│  feature_extraction.py    │   │  spectrogram.py           │
│  ├── time_features()      │   │  ├── compute_stft()       │
│  ├── frequency_features() │   │  ├── compute_cwt()        │
│  └── envelope_features()  │   │  └── resize_spectrogram() │
└───────────────────────────┘   └───────────────────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│   CLASSICAL ML MODELS     │   │   DEEP LEARNING MODELS    │
├───────────────────────────┤   ├───────────────────────────┤
│  models.py                │   │  models.py                │
│  ├── SVMClassifier        │   │  ├── CNN2D (spectrograms) │
│  ├── RandomForestModel    │   │  ├── CNN1D (raw signal)   │
│  └── XGBoostModel         │   │  └── HybridModel          │
└───────────────────────────┘   └───────────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       EVALUATION LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  evaluate.py                         │  visualize.py                │
│  ├── compute_metrics()               │  ├── plot_confusion_matrix() │
│  ├── cross_validate()                │  ├── plot_tsne()             │
│  └── cross_load_test()               │  └── plot_spectrogram()      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  inference.py                                                        │
│  ├── FaultDetector class                                            │
│  ├── load_model()                                                   │
│  ├── preprocess_signal()                                            │
│  └── predict() → (fault_type, confidence)                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.8+ | Industry standard for ML |
| Signal Processing | scipy, numpy | Robust, well-documented |
| Spectrograms | librosa | Standard audio/signal library |
| Classical ML | scikit-learn, xgboost | Fast iteration, interpretable |
| Deep Learning | PyTorch | Flexible, debugging-friendly |
| Visualization | matplotlib, seaborn | Publication-quality plots |
| Data Format | .mat (scipy.io) | CWRU native format |

**Tradeoffs:**
- PyTorch vs TensorFlow: PyTorch chosen for easier debugging and dynamic graphs
- librosa vs custom STFT: librosa provides optimized, tested implementations

**Fallbacks:**
- If GPU unavailable: Use smaller batch sizes, fewer epochs
- If XGBoost slow: Fall back to Random Forest

---

## Phased Implementation Plan

### Phase 1: Data Pipeline Foundation

**Objective:** Load CWRU data and create reproducible dataset

**Scope:**
- `src/data_loader.py`: Load .mat files, parse metadata
- `src/preprocessing.py`: Segment signals, basic normalization
- `notebooks/EDA.ipynb`: Exploratory analysis

**Deliverables:**
- [ ] Function to download CWRU dataset
- [ ] Function to load and parse .mat files
- [ ] Segmentation into fixed-length windows
- [ ] Train/test split by load condition
- [ ] EDA notebook with signal visualizations

**Verification:**
```bash
python -c "from src.data_loader import load_cwru_data; X, y = load_cwru_data(); print(X.shape, y.shape)"
# Expected: (N, 2048) (N,) where N > 1000
```

**Technical Challenges:**
- .mat file structure varies between files
- Handling different sampling rates (12kHz vs 48kHz)
- Avoiding data leakage in time series

**Definition of Done:**
- [ ] All .mat files load without errors
- [ ] Segments are balanced across classes
- [ ] EDA notebook renders with visualizations

**Code Skeleton:**
```python
# src/data_loader.py
import scipy.io
import numpy as np
from pathlib import Path

CWRU_URL = "https://engineering.case.edu/bearingdatacenter/download/"

FILE_MAPPING = {
    'normal': {'0hp': '97.mat', '1hp': '98.mat', '2hp': '99.mat', '3hp': '100.mat'},
    'inner_007': {'0hp': '105.mat', '1hp': '106.mat', '2hp': '107.mat', '3hp': '108.mat'},
    'outer_007': {'0hp': '130.mat', '1hp': '131.mat', '2hp': '132.mat', '3hp': '133.mat'},
    'ball_007': {'0hp': '118.mat', '1hp': '119.mat', '2hp': '120.mat', '3hp': '121.mat'},
}

def load_mat_file(filepath: str) -> np.ndarray:
    """Load vibration data from CWRU .mat file."""
    mat = scipy.io.loadmat(filepath)
    # Find the DE (drive end) accelerometer key
    de_key = [k for k in mat.keys() if 'DE' in k and 'time' not in k.lower()][0]
    return mat[de_key].flatten()

def segment_signal(signal: np.ndarray, segment_length: int = 2048, overlap: float = 0.5) -> np.ndarray:
    """Segment signal into fixed-length windows."""
    step = int(segment_length * (1 - overlap))
    n_segments = (len(signal) - segment_length) // step + 1
    segments = np.zeros((n_segments, segment_length))
    for i in range(n_segments):
        start = i * step
        segments[i] = signal[start:start + segment_length]
    return segments
```

---

### Phase 2: Feature Engineering

**Objective:** Extract discriminative features from vibration signals

**Scope:**
- `src/feature_extraction.py`: Time, frequency, envelope features
- Feature selection and importance analysis

**Deliverables:**
- [ ] 30-40 features per segment
- [ ] Time domain: RMS, kurtosis, crest factor, etc.
- [ ] Frequency domain: spectral centroid, band energies
- [ ] Envelope analysis: Hilbert transform, envelope spectrum

**Verification:**
```bash
python -c "from src.feature_extraction import extract_features; import numpy as np; f = extract_features(np.random.randn(2048)); print(len(f))"
# Expected: 35-45 features
```

**Technical Challenges:**
- Bearing fault frequencies are load-dependent
- Envelope spectrum requires proper bandpass selection
- Feature scaling for different algorithms

**Definition of Done:**
- [ ] Feature extractor returns consistent feature vector
- [ ] Features documented with physical meaning
- [ ] Correlation analysis shows discriminative power

**Code Skeleton:**
```python
# src/feature_extraction.py
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert, butter, filtfilt

def time_features(segment: np.ndarray) -> dict:
    """Extract time-domain features."""
    return {
        'mean': np.mean(segment),
        'std': np.std(segment),
        'rms': np.sqrt(np.mean(segment**2)),
        'peak': np.max(np.abs(segment)),
        'peak_to_peak': np.ptp(segment),
        'crest_factor': np.max(np.abs(segment)) / np.sqrt(np.mean(segment**2)),
        'shape_factor': np.sqrt(np.mean(segment**2)) / np.mean(np.abs(segment)),
        'impulse_factor': np.max(np.abs(segment)) / np.mean(np.abs(segment)),
        'skewness': skew(segment),
        'kurtosis': kurtosis(segment),
    }

def frequency_features(segment: np.ndarray, fs: int = 12000) -> dict:
    """Extract frequency-domain features."""
    fft = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(len(segment), 1/fs)
    magnitude = np.abs(fft)

    spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)

    return {
        'spectral_centroid': spectral_centroid,
        'spectral_energy': np.sum(magnitude**2),
        'dominant_freq': freqs[np.argmax(magnitude)],
    }

def envelope_features(segment: np.ndarray, fs: int = 12000) -> dict:
    """Extract envelope spectrum features for bearing faults."""
    # Bandpass filter around structural resonance
    b, a = butter(4, [2000/(fs/2), 5000/(fs/2)], btype='band')
    filtered = filtfilt(b, a, segment)

    # Hilbert envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # Envelope spectrum
    env_fft = np.abs(np.fft.rfft(envelope))

    return {
        'envelope_rms': np.sqrt(np.mean(envelope**2)),
        'envelope_peak': np.max(envelope),
        'envelope_kurtosis': kurtosis(envelope),
    }
```

---

### Phase 3: Classical ML Pipeline

**Objective:** Train and evaluate classical ML models

**Scope:**
- `src/models.py`: SVM, Random Forest, XGBoost wrappers
- `src/train.py`: Training pipeline with cross-validation
- `src/evaluate.py`: Metrics computation

**Deliverables:**
- [ ] Trained SVM, RF, XGBoost models
- [ ] 5-fold cross-validation results
- [ ] Confusion matrices and classification reports
- [ ] Feature importance rankings

**Verification:**
```bash
python src/train.py --model rf --cv 5
# Expected: Accuracy > 95%, F1 > 0.94
```

**Technical Challenges:**
- Class imbalance handling
- Hyperparameter tuning without overfitting
- Cross-load generalization

**Definition of Done:**
- [ ] All three models train successfully
- [ ] Cross-validation metrics logged
- [ ] Models saved to disk

**Code Skeleton:**
```python
# src/models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

def create_rf_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42))
    ])

def create_svm_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
    ])

def create_xgb_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1))
    ])
```

---

### Phase 4: Spectrogram Generation and CNN

**Objective:** Implement deep learning approach using spectrograms

**Scope:**
- `src/spectrogram.py`: STFT and CWT computation
- `src/models.py`: Add CNN architectures
- Training loop with PyTorch

**Deliverables:**
- [ ] Spectrogram generation pipeline
- [ ] 2D CNN for spectrogram classification
- [ ] 1D CNN for raw signal classification
- [ ] Training curves and checkpoints

**Verification:**
```bash
python src/train.py --model cnn2d --epochs 50
# Expected: Accuracy > 97%
```

**Technical Challenges:**
- Spectrogram size vs information tradeoff
- GPU memory management
- Avoiding overfitting with limited data

**Definition of Done:**
- [ ] Spectrograms generated consistently
- [ ] CNN trains without errors
- [ ] Test accuracy > 97%

**Code Skeleton:**
```python
# src/models.py (CNN addition)
import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

---

### Phase 5: Evaluation and Visualization

**Objective:** Comprehensive evaluation and publication-ready visualizations

**Scope:**
- `src/evaluate.py`: Cross-load testing, metrics
- `src/visualize.py`: All visualization functions

**Deliverables:**
- [ ] Cross-load generalization results
- [ ] Confusion matrices for all models
- [ ] t-SNE visualization of features
- [ ] Spectrogram examples by class
- [ ] ROC curves

**Verification:**
```bash
python src/evaluate.py --all
# Generates plots in results/ directory
```

**Definition of Done:**
- [ ] All visualizations generated
- [ ] Results documented in README
- [ ] Cross-load accuracy > 90%

---

### Phase 6: Inference and Demo

**Objective:** Production-ready inference pipeline

**Scope:**
- `src/inference.py`: FaultDetector class
- Demo script with sample data

**Deliverables:**
- [ ] FaultDetector class with clean API
- [ ] Support for both ML and CNN models
- [ ] Demo script with example output

**Verification:**
```bash
python src/inference.py --input data/test_sample.mat
# Expected: Fault Type: Inner Race, Confidence: 0.94
```

**Definition of Done:**
- [ ] Inference runs in < 100ms per sample
- [ ] Clear output format
- [ ] README updated with usage

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CWRU download fails | Medium | High | Mirror data to personal storage |
| GPU not available | High | Medium | CPU fallback with smaller models |
| Overfitting on small dataset | Medium | High | Cross-validation, regularization |
| Cross-load generalization poor | Medium | Medium | Data augmentation, domain adaptation |
| .mat parsing errors | Low | Medium | Robust error handling, logging |

**Early Warning Signs:**
- Training loss not decreasing: Check learning rate, data loading
- Test accuracy << train accuracy: Overfitting, add regularization
- Long training times: Reduce batch size, model complexity

---

## Testing Strategy

**Unit Tests:**
1. `test_data_loader.py`: Verify .mat loading, segmentation
2. `test_features.py`: Check feature dimensions, valid ranges
3. `test_models.py`: Model forward pass, output shapes

**Integration Tests:**
- End-to-end pipeline from raw data to prediction
- Cross-validation produces expected metrics structure

**First Three Tests:**
```python
# tests/test_data_loader.py
def test_load_mat_file():
    """Verify .mat file loads correctly."""
    signal = load_mat_file('data/97.mat')
    assert len(signal) > 100000
    assert signal.dtype == np.float64

def test_segment_signal():
    """Verify segmentation produces correct shapes."""
    signal = np.random.randn(50000)
    segments = segment_signal(signal, segment_length=2048)
    assert segments.shape[1] == 2048
    assert segments.shape[0] > 20

def test_feature_dimensions():
    """Verify feature extraction returns consistent dimensions."""
    segment = np.random.randn(2048)
    features = extract_all_features(segment)
    assert len(features) >= 30
```

---

## First Concrete Task

**File:** `src/data_loader.py`

**Function Signature:**
```python
def load_mat_file(filepath: str) -> np.ndarray:
    """
    Load vibration signal from CWRU .mat file.

    Args:
        filepath: Path to .mat file

    Returns:
        1D numpy array of vibration values
    """
```

**Starter Code:**
```python
"""CWRU Bearing Dataset Loader."""
import scipy.io
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import urllib.request
import os

# Dataset configuration
CWRU_BASE_URL = "https://engineering.case.edu/bearingdatacenter/download/file/"
DATA_DIR = Path(__file__).parent.parent / "data"

# File mapping: condition -> {load -> filename}
FILE_MAPPING = {
    'normal': {
        '0hp': '97.mat', '1hp': '98.mat', '2hp': '99.mat', '3hp': '100.mat'
    },
    'inner_007': {
        '0hp': '105.mat', '1hp': '106.mat', '2hp': '107.mat', '3hp': '108.mat'
    },
    'outer_007': {
        '0hp': '130.mat', '1hp': '131.mat', '2hp': '132.mat', '3hp': '133.mat'
    },
    'ball_007': {
        '0hp': '118.mat', '1hp': '119.mat', '2hp': '120.mat', '3hp': '121.mat'
    },
}

CLASS_LABELS = {'normal': 0, 'inner_007': 1, 'outer_007': 2, 'ball_007': 3}


def load_mat_file(filepath: str) -> np.ndarray:
    """
    Load vibration signal from CWRU .mat file.

    Args:
        filepath: Path to .mat file

    Returns:
        1D numpy array of vibration values (drive end accelerometer)
    """
    mat = scipy.io.loadmat(filepath)
    # Find drive end (DE) accelerometer key
    de_keys = [k for k in mat.keys() if 'DE' in k and 'time' not in k.lower()]
    if not de_keys:
        raise ValueError(f"No DE accelerometer data found in {filepath}")
    return mat[de_keys[0]].flatten()


if __name__ == "__main__":
    # Test loading
    print("Testing data loader...")
    # Add test code here
```

**First Commit Message:**
```
feat: add CWRU dataset loader with .mat file parsing

- Implement load_mat_file() for drive end accelerometer data
- Define file mapping for normal and fault conditions
- Set up class labels for 4-class classification
```

---

## Dependencies

```
# requirements.txt
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
torch>=1.9.0
librosa>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
pytest>=6.2.0
```

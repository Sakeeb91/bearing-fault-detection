## Objective

Create production-ready inference pipeline and interactive demo for real-time fault detection.

## Dependencies

- #4 Phase 3: Classical ML Pipeline
- #5 Phase 4: Spectrogram Generation and CNN
- #6 Phase 5: Evaluation and Visualization

## Tasks

- [ ] Create `FaultDetector` class with clean API
- [ ] Support both classical ML and CNN models
- [ ] Implement batch and single-sample inference
- [ ] Add confidence scores to predictions
- [ ] Create command-line demo script
- [ ] Optimize inference speed (<100ms per sample)
- [ ] Add example usage to README

## Technical Details

### FaultDetector Class

```python
# src/inference.py
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Tuple, Union, List

class FaultDetector:
    """Production-ready bearing fault detector."""

    CLASS_NAMES = ['Normal', 'Inner Race Fault', 'Outer Race Fault', 'Ball Fault']

    def __init__(self, model_path: str, model_type: str = 'rf'):
        """
        Initialize fault detector.

        Args:
            model_path: Path to saved model
            model_type: 'rf', 'svm', 'xgb', 'cnn1d', or 'cnn2d'
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_type in ['rf', 'svm', 'xgb']:
            self.model = joblib.load(model_path)
            self.is_deep = False
        else:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            self.is_deep = True

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess raw vibration signal."""
        # Ensure correct length
        if len(signal) < 2048:
            signal = np.pad(signal, (0, 2048 - len(signal)))
        elif len(signal) > 2048:
            signal = signal[:2048]

        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        return signal

    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract features for classical ML models."""
        from src.feature_extraction import extract_all_features
        return extract_all_features(signal)

    def predict(self, signal: np.ndarray) -> Tuple[str, float]:
        """
        Predict fault type from vibration signal.

        Args:
            signal: Raw vibration signal (1D numpy array)

        Returns:
            Tuple of (fault_type, confidence)
        """
        signal = self.preprocess(signal)

        if self.is_deep:
            return self._predict_deep(signal)
        else:
            return self._predict_classical(signal)

    def _predict_classical(self, signal: np.ndarray) -> Tuple[str, float]:
        """Prediction using classical ML model."""
        features = self.extract_features(signal).reshape(1, -1)
        proba = self.model.predict_proba(features)[0]
        pred_class = np.argmax(proba)
        confidence = proba[pred_class]
        return self.CLASS_NAMES[pred_class], confidence

    def _predict_deep(self, signal: np.ndarray) -> Tuple[str, float]:
        """Prediction using deep learning model."""
        with torch.no_grad():
            if self.model_type == 'cnn1d':
                x = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(self.device)
            else:  # cnn2d
                from src.spectrogram import compute_stft_spectrogram, resize_spectrogram
                spec = compute_stft_spectrogram(signal)
                spec = resize_spectrogram(spec, (128, 128))
                x = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(self.device)

            outputs = self.model(x)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(proba)
            confidence = proba[pred_class]

        return self.CLASS_NAMES[pred_class], confidence

    def predict_batch(self, signals: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Batch prediction for multiple signals."""
        return [self.predict(s) for s in signals]
```

### Demo Script

```python
# src/demo.py
import argparse
import time
from pathlib import Path
from src.inference import FaultDetector
from src.data_loader import load_mat_file

def main():
    parser = argparse.ArgumentParser(description='Bearing Fault Detection Demo')
    parser.add_argument('--input', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--model', type=str, default='models/rf_model.pkl')
    parser.add_argument('--model-type', type=str, default='rf',
                        choices=['rf', 'svm', 'xgb', 'cnn1d', 'cnn2d'])
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model_type} model from {args.model}...")
    detector = FaultDetector(args.model, args.model_type)

    # Load signal
    print(f"Loading signal from {args.input}...")
    signal = load_mat_file(args.input)

    # Make prediction
    start_time = time.time()
    fault_type, confidence = detector.predict(signal[:2048])
    inference_time = (time.time() - start_time) * 1000

    # Display results
    print("\n" + "="*50)
    print("BEARING FAULT DETECTION RESULT")
    print("="*50)
    print(f"Fault Type:  {fault_type}")
    print(f"Confidence:  {confidence:.2%}")
    print(f"Inference:   {inference_time:.1f} ms")
    print("="*50)

if __name__ == "__main__":
    main()
```

### Usage Examples

```bash
# Using Random Forest model
python src/demo.py --input data/test_sample.mat --model models/rf_model.pkl --model-type rf

# Using CNN model
python src/demo.py --input data/test_sample.mat --model models/cnn2d_model.pth --model-type cnn2d
```

## Verification

```bash
python src/demo.py --input data/105.mat --model models/rf_model.pkl --model-type rf
# Expected output:
# ================================================
# BEARING FAULT DETECTION RESULT
# ================================================
# Fault Type:  Inner Race Fault
# Confidence:  94.52%
# Inference:   45.2 ms
# ================================================
```

## Definition of Done

- [ ] FaultDetector class works with all model types
- [ ] Inference time < 100ms per sample
- [ ] Demo script produces clear output
- [ ] README updated with usage examples
- [ ] Example predictions documented

## Files to Create/Modify

| File | Description |
|------|-------------|
| `src/inference.py` | FaultDetector class |
| `src/demo.py` | Interactive demo script |
| `README.md` | Add usage examples |

---
Parent: #1

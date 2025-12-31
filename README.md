# Bearing Fault Detection System

A production-ready vibration-based fault detection system for predictive maintenance of rotating machinery. This project implements signal processing pipelines and machine learning models to classify bearing health conditions from accelerometer data.

## Problem Statement

Unplanned equipment failures cost industries billions annually, with bearing failures accounting for 40-50% of rotating machinery breakdowns. This system enables condition-based maintenance by automatically detecting and classifying bearing faults from vibration signals.

## Industry Applications

- Manufacturing predictive maintenance
- Wind turbine monitoring
- Aerospace (jet engine bearings)
- Automotive (wheel bearings, transmissions)
- Rail transport (axle bearings)
- HVAC systems

## Dataset

**CWRU Bearing Dataset** (Case Western Reserve University)
- Most widely used benchmark in bearing fault diagnosis
- Vibration data from accelerometers at 12kHz and 48kHz
- 4 conditions: Normal, Inner Race Fault, Outer Race Fault, Ball Fault
- Multiple fault sizes and load conditions

## Classification Scheme

| Class | Description |
|-------|-------------|
| 0 | Normal |
| 1 | Inner Race Fault |
| 2 | Outer Race Fault |
| 3 | Ball Fault |

## Technical Architecture

```
Raw Vibration Signal
        |
        v
+------------------+
|  Preprocessing   |
|  - Segmentation  |
|  - Filtering     |
|  - Normalization |
+------------------+
        |
        v
+------------------+     +------------------+
| Feature Extract  | --> | Classical ML     |
| - Time Domain    |     | - SVM            |
| - Frequency      |     | - Random Forest  |
| - Envelope       |     | - XGBoost        |
+------------------+     +------------------+
        |
        v
+------------------+     +------------------+
| Spectrogram Gen  | --> | CNN Classifier   |
| - STFT           |     | - 2D CNN         |
| - CWT            |     | - 1D CNN         |
+------------------+     +------------------+
        |
        v
+------------------+
|    Inference     |
|  Fault Type +    |
|  Confidence      |
+------------------+
```

## Project Structure

```
bearing-fault-detection/
├── data/                    # Raw and processed data
├── src/
│   ├── data_loader.py       # CWRU .mat file loading
│   ├── preprocessing.py     # Segmentation, filtering
│   ├── feature_extraction.py # Time, frequency, envelope features
│   ├── spectrogram.py       # STFT/CWT computation
│   ├── models.py            # Classical ML and CNN models
│   ├── train.py             # Training with cross-validation
│   ├── evaluate.py          # Metrics and visualizations
│   ├── inference.py         # Real-time fault detection
│   └── visualize.py         # Spectrograms, envelope spectra, t-SNE
├── notebooks/
│   └── EDA.ipynb            # Exploratory data analysis
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── tests/
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/Sakeeb91/bearing-fault-detection.git
cd bearing-fault-detection

# Install dependencies
pip install -r requirements.txt

# Download data
python src/data_loader.py --download

# Train models
python src/train.py

# Evaluate and generate visualizations
python src/evaluate.py

# Run inference demo
python src/inference.py --input sample.mat
```

## Expected Results

| Model | Same-Load Accuracy | Cross-Load Accuracy |
|-------|-------------------|---------------------|
| Feature-based ML | 95-98% | 90-95% |
| CNN on Spectrograms | 98-99% | 92-96% |
| 1D CNN on Raw Signal | 97-99% | 91-95% |

## Key Features

- **Multiple approaches**: Classical ML with hand-crafted features, CNN on spectrograms, 1D CNN on raw signals
- **Envelope analysis**: Critical technique for bearing fault frequency detection
- **Cross-load generalization**: Test model robustness across operating conditions
- **Visualization suite**: Spectrograms, envelope spectra, t-SNE feature plots

## Requirements

- Python 3.8+
- PyTorch 1.9+
- scikit-learn
- scipy
- numpy
- pandas
- matplotlib
- seaborn

## License

MIT License

## Author

Sakeeb Rahman - [GitHub](https://github.com/Sakeeb91)

## Objective

Extract discriminative features from vibration signals for classical ML models.

## Dependencies

- #2 Phase 1: Data Pipeline Foundation

## Tasks

- [ ] Implement time-domain features (RMS, kurtosis, crest factor, etc.)
- [ ] Implement frequency-domain features (spectral centroid, band energies)
- [ ] Implement envelope analysis (Hilbert transform, envelope spectrum)
- [ ] Create feature extraction pipeline
- [ ] Analyze feature importance and correlations
- [ ] Document feature physical meanings

## Technical Details

### Time Domain Features (~12 features)

```python
def time_features(segment: np.ndarray) -> dict:
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
        'clearance_factor': np.max(np.abs(segment)) / (np.mean(np.sqrt(np.abs(segment))))**2,
        'margin_factor': np.max(np.abs(segment)) / np.mean(np.abs(segment)),
    }
```

### Frequency Domain Features (~10 features)

```python
def frequency_features(segment: np.ndarray, fs: int = 12000) -> dict:
    fft = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(len(segment), 1/fs)
    magnitude = np.abs(fft)

    spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * magnitude) / np.sum(magnitude))

    # Band energies for bearing fault frequencies
    band_50_150 = np.sum(magnitude[(freqs >= 50) & (freqs < 150)]**2)
    band_150_300 = np.sum(magnitude[(freqs >= 150) & (freqs < 300)]**2)
    band_300_500 = np.sum(magnitude[(freqs >= 300) & (freqs < 500)]**2)

    return {
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread,
        'spectral_energy': np.sum(magnitude**2),
        'dominant_freq': freqs[np.argmax(magnitude)],
        'band_50_150': band_50_150,
        'band_150_300': band_150_300,
        'band_300_500': band_300_500,
    }
```

### Envelope Analysis (~8 features)

```python
def envelope_features(segment: np.ndarray, fs: int = 12000) -> dict:
    # Bandpass filter around structural resonance (2-5 kHz)
    b, a = butter(4, [2000/(fs/2), 5000/(fs/2)], btype='band')
    filtered = filtfilt(b, a, segment)

    # Hilbert envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    return {
        'envelope_rms': np.sqrt(np.mean(envelope**2)),
        'envelope_peak': np.max(envelope),
        'envelope_kurtosis': kurtosis(envelope),
        'envelope_crest': np.max(envelope) / np.sqrt(np.mean(envelope**2)),
    }
```

## Verification

```bash
python -c "from src.feature_extraction import extract_all_features; import numpy as np; f = extract_all_features(np.random.randn(2048)); print(f'Features: {len(f)}')"
# Expected: Features: 30-40
```

## Definition of Done

- [ ] Feature extractor returns consistent 30-40 feature vector
- [ ] Features documented with physical meaning
- [ ] Correlation analysis shows discriminative power
- [ ] Unit tests pass for all feature functions

## Files to Create/Modify

| File | Description |
|------|-------------|
| `src/feature_extraction.py` | All feature extraction functions |
| `tests/test_features.py` | Unit tests for features |

---
Parent: #1

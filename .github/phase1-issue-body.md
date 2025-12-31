## Objective

Load CWRU bearing dataset and create reproducible data pipeline with proper train/test splits.

## Dependencies

None - this is the foundation phase.

## Tasks

- [ ] Create `src/data_loader.py` with .mat file loading
- [ ] Implement signal segmentation (2048/4096 samples)
- [ ] Set up class label mapping (Normal, Inner, Outer, Ball)
- [ ] Create train/test split by load condition
- [ ] Build `notebooks/EDA.ipynb` with signal visualizations
- [ ] Add download script for CWRU data

## Technical Details

### File Mapping

```python
FILE_MAPPING = {
    'normal': {'0hp': '97.mat', '1hp': '98.mat', '2hp': '99.mat', '3hp': '100.mat'},
    'inner_007': {'0hp': '105.mat', '1hp': '106.mat', '2hp': '107.mat', '3hp': '108.mat'},
    'outer_007': {'0hp': '130.mat', '1hp': '131.mat', '2hp': '132.mat', '3hp': '133.mat'},
    'ball_007': {'0hp': '118.mat', '1hp': '119.mat', '2hp': '120.mat', '3hp': '121.mat'},
}
```

### Key Functions

```python
def load_mat_file(filepath: str) -> np.ndarray:
    """Load vibration data from CWRU .mat file."""
    mat = scipy.io.loadmat(filepath)
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

## Verification

```bash
python -c "from src.data_loader import load_cwru_data; X, y = load_cwru_data(); print(X.shape, y.shape)"
# Expected: (N, 2048) (N,) where N > 1000
```

## Definition of Done

- [ ] All .mat files load without errors
- [ ] Segments are balanced across classes
- [ ] EDA notebook renders with visualizations
- [ ] Unit tests pass for data loading

## Files to Create

| File | Description |
|------|-------------|
| `src/data_loader.py` | Main data loading module |
| `src/preprocessing.py` | Segmentation and normalization |
| `notebooks/EDA.ipynb` | Exploratory analysis |
| `tests/test_data_loader.py` | Unit tests |

---
Parent: #1

## Objective

Comprehensive evaluation of all models with publication-ready visualizations and cross-load generalization testing.

## Dependencies

- #4 Phase 3: Classical ML Pipeline
- #5 Phase 4: Spectrogram Generation and CNN

## Tasks

- [ ] Implement cross-load generalization testing
- [ ] Generate confusion matrices for all models
- [ ] Create t-SNE visualization of learned features
- [ ] Plot spectrogram examples by fault class
- [ ] Generate ROC curves (one-vs-rest)
- [ ] Compare all models in summary table
- [ ] Update README with results

## Technical Details

### Cross-Load Testing

```python
def cross_load_evaluation(model, data_by_load: dict):
    """Test model trained on some loads against held-out loads."""
    results = {}

    # Train on 0hp, 1hp, 2hp - test on 3hp
    train_loads = ['0hp', '1hp', '2hp']
    test_load = '3hp'

    X_train = np.vstack([data_by_load[l]['X'] for l in train_loads])
    y_train = np.hstack([data_by_load[l]['y'] for l in train_loads])
    X_test = data_by_load[test_load]['X']
    y_test = data_by_load[test_load]['y']

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results['cross_load_accuracy'] = accuracy_score(y_test, y_pred)
    results['cross_load_f1'] = f1_score(y_test, y_pred, average='weighted')

    return results
```

### Visualization Functions

```python
# src/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_tsne(features, labels, class_names, save_path=None):
    """t-SNE visualization of feature space."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        mask = labels == i
        plt.scatter(embedded[mask, 0], embedded[mask, 1],
                   label=name, alpha=0.6, s=20)
    plt.legend()
    plt.title('t-SNE Feature Visualization')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_spectrogram_examples(spectrograms, labels, class_names, save_path=None):
    """Plot example spectrograms for each class."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (ax, name) in enumerate(zip(axes, class_names)):
        idx = np.where(labels == i)[0][0]
        ax.imshow(spectrograms[idx], aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(name)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_proba, class_names, save_path=None):
    """Plot ROC curves for multi-class (one-vs-rest)."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Results Summary Table

```python
def generate_results_table(results: dict) -> str:
    """Generate markdown table of results."""
    header = "| Model | Same-Load Acc | Cross-Load Acc | F1 Score |\n"
    header += "|-------|---------------|----------------|----------|\n"

    rows = []
    for model_name, metrics in results.items():
        row = f"| {model_name} | {metrics['same_load_acc']:.2%} | "
        row += f"{metrics['cross_load_acc']:.2%} | {metrics['f1']:.3f} |"
        rows.append(row)

    return header + "\n".join(rows)
```

## Verification

```bash
python src/evaluate.py --all
# Generates: results/confusion_matrix_*.png, results/tsne.png, results/roc_curves.png
```

## Definition of Done

- [ ] All visualizations generated in `results/` directory
- [ ] Cross-load accuracy > 90% for best model
- [ ] README updated with result images and table
- [ ] Comparison analysis written

## Files to Create/Modify

| File | Description |
|------|-------------|
| `src/evaluate.py` | Comprehensive evaluation script |
| `src/visualize.py` | Visualization functions |
| `results/` | Output directory for figures |
| `README.md` | Update with results |

---
Parent: #1

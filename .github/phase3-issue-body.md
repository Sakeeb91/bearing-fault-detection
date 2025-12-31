## Objective

Train and evaluate classical machine learning models (SVM, Random Forest, XGBoost) for bearing fault classification.

## Dependencies

- #2 Phase 1: Data Pipeline Foundation
- #3 Phase 2: Feature Engineering

## Tasks

- [ ] Create model wrappers with sklearn pipelines
- [ ] Implement training script with cross-validation
- [ ] Add hyperparameter tuning (optional, GridSearch)
- [ ] Compute evaluation metrics (accuracy, F1, confusion matrix)
- [ ] Generate feature importance plots
- [ ] Save trained models to disk

## Technical Details

### Model Pipelines

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

def create_rf_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

def create_svm_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        ))
    ])

def create_xgb_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ))
    ])
```

### Training Script Structure

```python
# src/train.py
import argparse
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train_model(model_name: str, X: np.ndarray, y: np.ndarray, cv: int = 5):
    model = get_model(model_name)

    # Stratified K-Fold for class balance
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"{model_name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    # Final training on all data
    model.fit(X, y)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rf', 'svm', 'xgb'], default='rf')
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()

    X, y = load_features()
    model = train_model(args.model, X, y, args.cv)
    save_model(model, f"models/{args.model}_model.pkl")
```

### Evaluation Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred
    }
```

## Verification

```bash
python src/train.py --model rf --cv 5
# Expected output:
# rf CV Accuracy: 0.9650 (+/- 0.0120)
# Model saved to models/rf_model.pkl
```

## Definition of Done

- [ ] All three models train successfully
- [ ] Cross-validation accuracy > 95%
- [ ] Confusion matrices show good per-class performance
- [ ] Models saved and loadable
- [ ] Feature importance rankings generated

## Files to Create/Modify

| File | Description |
|------|-------------|
| `src/models.py` | Model pipeline definitions |
| `src/train.py` | Training script |
| `src/evaluate.py` | Evaluation functions |
| `models/` | Directory for saved models |

---
Parent: #1

# Random Forest for Network Intrusion Detection

This directory implements a standalone Random Forest classifier for network intrusion detection using the KitsuneSystemAttackData dataset from Hugging Face. The implementation includes data preprocessing, model training, and evaluation with various performance metrics.

## Files

- `load_data.py`: Handles dataset loading, preprocessing, standardization, and class balancing
- `rf_model.py`: Contains the Random Forest model implementation, training, and evaluation logic

## Features

- **Data Preprocessing**:
  - Class balancing with undersampling or oversampling options
  - StandardScaler feature normalization
  - Stratified train/test split
  - Efficient caching of processed data

- **Model Training**:
  - Random Forest classification with configurable hyperparameters
  - Optional hyperparameter tuning with RandomizedSearchCV
  - Multi-core training for improved performance

- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Log loss
  - False positive rate
  - False negative rate

- **Feature Analysis**:
  - Automatic feature importance ranking and logging

## Usage

### Data Preprocessing

```python
from load_data import load_preprocessed_data

# Default: undersampling with 1:1 ratio
X_train, X_test, y_train, y_test = load_preprocessed_data()

# With custom parameters
X_train, X_test, y_train, y_test = load_preprocessed_data(
    balance_method='oversample',  # 'undersample', 'oversample', or None
    sampling_ratio=2.0            # Ratio for balancing
)
```

### Model Training and Evaluation

```python
from rf_model import train_rf_model

# Train with default parameters
model, metrics = train_rf_model()

# Train with custom parameters
model, metrics = train_rf_model(
    balance_method='undersample',  # Method for class balancing
    sampling_ratio=1.0,            # Ratio for class balancing
    n_estimators=100,              # Number of trees in forest
    max_depth=None,                # Maximum depth of trees
    min_samples_split=2,           # Min samples to split a node
    min_samples_leaf=1,            # Min samples at leaf node
    tune_hyperparams=False         # Enable hyperparameter tuning
)
```

### Running the Complete Pipeline

```bash
python rf_model.py
```

## Output Files

The implementation creates the following directories and files:

- `data_cache/`: Contains cached preprocessed data and fitted scaler
- `models/`: Stores trained model files
- `results/`: Contains evaluation metrics and feature importance rankings

## Customizing the Model

You can modify parameters in the main() function of rf_model.py to customize the model:

```python
model, metrics = train_rf_model(
    balance_method='undersample',  # Change to 'oversample' or None
    sampling_ratio=1.0,            # Adjust ratio as needed
    n_estimators=100,              # Increase for potentially better performance
    tune_hyperparams=True          # Set to True to find optimal parameters
)
``` 
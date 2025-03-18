import numpy as np
import pandas as pd
import logging
import os
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import RandomizedSearchCV
from load_data import load_preprocessed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RandomForest")

# Create directories for model and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def train_rf_model(balance_method='undersample', sampling_ratio=1.0, n_estimators=100, 
                  max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                  tune_hyperparams=False):
    """Train a Random Forest model.
    
    Args:
        balance_method: Method for class balancing ('undersample', 'oversample', or None)
        sampling_ratio: Sampling ratio for class balancing
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum number of samples required to split a node
        min_samples_leaf: Minimum number of samples required at a leaf node
        tune_hyperparams: Whether to perform hyperparameter tuning
    
    Returns:
        Trained model and evaluation metrics
    """
    logger.info(f"Loading preprocessed data (balance_method={balance_method}, sampling_ratio={sampling_ratio})")
    X_train, X_test, y_train, y_test = load_preprocessed_data(balance_method, sampling_ratio)
    
    logger.info(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
    
    if tune_hyperparams:
        logger.info("Performing hyperparameter tuning using RandomizedSearchCV")
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        # Randomized search for faster tuning
        rf_tuned = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=20,
            scoring='f1',
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        rf_tuned.fit(X_train, y_train)
        model = rf_tuned.best_estimator_
        
        logger.info(f"Best parameters: {rf_tuned.best_params_}")
    else:
        logger.info(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train, y_train)
    
    # Save the trained model
    model_filename = f"models/rf_model_{balance_method}_{sampling_ratio}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_filename}")
    
    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test)
    
    return model, metrics

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with the same metrics as in results.csv.
    
    Args:
        model: Trained RandomForest model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test data")
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_test, y_pred)
    
    # Calculate loss
    metrics['loss'] = log_loss(y_test, y_proba)
    
    # Calculate false positive and false negative rates
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(y_test)):
        if y_test[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_test[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            fn += 1
        else:  # y_test[i] == 1 and y_pred[i] == 1
            tp += 1
    
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Log results
    logger.info(f"Evaluation metrics on test set:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.6f}")
    
    # Save metrics to CSV
    timestamp = int(time.time())
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(f"results/rf_metrics_{timestamp}.csv", index=False)
    
    return metrics

def main():
    """Main function to train and evaluate the model."""
    # Train with default settings
    logger.info("Starting Random Forest training")
    
    # Train the base model
    model, metrics = train_rf_model(
        balance_method='undersample',  # Use undersampling for class balance
        sampling_ratio=1.0,           # Equal class distribution
        n_estimators=100,             # Number of trees in the forest
        tune_hyperparams=False        # Set to True to perform hyperparameter tuning
    )
    
    # Log feature importance
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': [f"Feature_{i}" for i in range(len(feature_importances))],
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df.to_csv("results/feature_importance.csv", index=False)
    
    logger.info("Top 10 important features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"{row['Feature']}: {row['Importance']:.6f}")
    
    return metrics

if __name__ == "__main__":
    main()

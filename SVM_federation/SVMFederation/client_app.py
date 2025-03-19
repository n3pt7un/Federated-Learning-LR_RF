"""new-new-federation: A Flower / sklearn app."""

import warnings
import numpy as np
from sklearn.metrics import (
    log_loss, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
from sklearn.linear_model import SGDClassifier

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from .task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)


def analyze_class_balance(y):
    """Analyze class balance in dataset."""
    unique_labels, counts = np.unique(y, return_counts=True)
    distribution = dict(zip([int(label) for label in unique_labels], counts))
    
    total = sum(counts)
    proportions = {label: count/total for label, count in distribution.items()}
    
    is_imbalanced = max(proportions.values()) > 0.7  # Arbitrary threshold
    
    return {
        "distribution": distribution,
        "proportions": proportions,
        "is_imbalanced": is_imbalanced,
        "total_samples": total
    }


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, loss):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.loss = loss
        
        # Analyze the test set distribution
        self.test_balance = analyze_class_balance(y_test)
        print(f"Client test set distribution: {self.test_balance}")

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Apply learning rate scheduling based on round number if SGDClassifier
        if isinstance(self.model, SGDClassifier) and "round" in config:
            current_round = config["round"]
            # Decrease learning rate as rounds progress for better convergence
            lr = 0.01 / (1 + 0.1 * current_round)
            self.model.eta0 = lr
            print(f"Setting learning rate to {lr:.5f} for round {current_round}")
            
        # Double the iterations for later rounds to improve convergence
        if "round" in config and config["round"] > 5:
            if isinstance(self.model, SGDClassifier):
                original_max_iter = self.model.max_iter
                self.model.max_iter = original_max_iter * 2
                print(f"Doubling max_iter to {self.model.max_iter} for round {config['round']}")

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        # Get predictions
        y_pred = self.model.predict(self.X_test)
        
        # Log prediction distribution
        pred_positive = np.sum(y_pred == 1)
        pred_negative = np.sum(y_pred == 0)
        positive_samples = np.sum(self.y_test == 1)
        negative_samples = np.sum(self.y_test == 0)
        
        print(f"Test set - Class distribution: 0: {negative_samples}, 1: {positive_samples}")
        print(f"Predictions - Distribution: 0: {pred_negative}, 1: {pred_positive}")
        
        # Calculate metrics
        try:
            # Accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Get confusion matrix (ensure it's valid)
            cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
            
            if cm.shape == (2, 2):  # We have a valid 2x2 confusion matrix
                tn, fp, fn, tp = cm.ravel()
                
                # Determine appropriate averaging method based on class balance
                if self.test_balance["is_imbalanced"]:
                    average_method = 'weighted'
                else:
                    average_method = 'binary'
                
                # Precision, recall, and F1 score with appropriate averaging
                precision = precision_score(self.y_test, y_pred, 
                                          zero_division=0, 
                                          average=average_method)
                recall = recall_score(self.y_test, y_pred, 
                                     zero_division=0, 
                                     average=average_method)
                f1 = f1_score(self.y_test, y_pred, 
                             zero_division=0, 
                             average=average_method)
                
                # False positive rate and false negative rate
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                # Log confusion matrix for debugging
                print(f"Confusion Matrix:\n{cm}")
                print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
            else:
                print(f"Warning: Unexpected confusion matrix shape: {cm.shape}")
                precision = recall = f1 = 0.0
                fpr = fnr = 0.0
            
            # Calculate loss based on model type
            if self.loss == 'hinge':
                # For SGDClassifier with hinge loss
                y_decision = self.model.decision_function(self.X_test)
                loss = np.mean(np.maximum(0, 1 - self.y_test * y_decision))
            else:
                # For LogisticRegression
                try:
                    y_proba = self.model.predict_proba(self.X_test)
                    loss = log_loss(self.y_test, y_proba)
                except Exception as e:
                    print(f"Error calculating log_loss: {e}")
                    loss = 1.0 - accuracy  # Fallback
                
        except Exception as e:
            # Fallback for metrics calculation
            print(f"Error calculating metrics: {e}")
            accuracy = self.model.score(self.X_test, self.y_test)
            precision = recall = f1 = fpr = fnr = 0.0
            loss = 1.0 - accuracy  # Simple fallback loss
        
        # Return all metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr)
        }
        
        return float(loss), len(self.X_test), metrics


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Get balancing parameters from run_config if available, otherwise use defaults 
    balance_method = context.run_config.get("balance-method", "undersample")
    sampling_ratio = float(context.run_config.get("sampling-ratio", "1.0"))
    
    print(f"Client {partition_id} - Using balance method: {balance_method}, ratio: {sampling_ratio}")
    
    # Load data with balanced sampling
    X_train, X_test, y_train, y_test = load_data(
        partition_id, 
        num_partitions, 
        balance_method=balance_method,
        sampling_ratio=sampling_ratio
    )
    
    # Log class distribution in this partition
    train_balance = analyze_class_balance(y_train)
    test_balance = analyze_class_balance(y_test)
    print(f"Client {partition_id} - Training set: {train_balance}")
    print(f"Client {partition_id} - Test set: {test_balance}")

    # Create Model based on configuration
    loss = context.run_config["loss"]
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs, loss)

    # Setting initial parameters
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test, loss).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
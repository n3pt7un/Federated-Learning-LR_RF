"""new-new-federation: A Flower / sklearn app."""
from typing import List, Tuple, Dict, Any, Optional, Union
from flwr.common import Context, ndarrays_to_parameters, EvaluateRes, Scalar, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, DifferentialPrivacyServerSideAdaptiveClipping
from .task import get_model, get_model_params, set_initial_params, set_model_params, load_data
from .custom_strategy import CustomFedAvg
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, log_loss
)


def aggregated_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from multiple clients."""
    if not metrics:
        return {}

    # Initialize aggregated metrics
    metric_names = [
        "accuracy", "precision", "recall", "f1_score", 
        "false_positive_rate", "false_negative_rate"
    ]
    
    # Calculate weighted average for each metric
    weighted_metrics = {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    if total_examples == 0:
        return {name: 0.0 for name in metric_names}
    
    for name in metric_names:
        # Skip metrics that might not be present in all client reports
        try:
            weighted = sum(num_examples * m.get(name, 0.0) for num_examples, m in metrics)
            weighted_metrics[name] = weighted / total_examples
        except Exception as e:
            print(f"Error aggregating {name}: {e}")
            weighted_metrics[name] = 0.0

    return weighted_metrics


def analyze_dataset(X, y):
    """Analyze the dataset distribution and return a summary."""
    unique_labels, counts = np.unique(y, return_counts=True)
    distribution = dict(zip([int(label) for label in unique_labels], counts))
    
    # Calculate class proportions
    total = sum(counts)
    proportions = {label: count/total for label, count in distribution.items()}
    
    # Check for class imbalance
    is_imbalanced = max(proportions.values()) > 0.7  # Arbitrary threshold
    
    return {
        "distribution": distribution,
        "proportions": proportions,
        "is_imbalanced": is_imbalanced,
        "total_samples": total,
        "num_features": X.shape[1] if X.shape[0] > 0 else 0
    }


def get_evaluate_fn(model, loss_type, balance_method='undersample', sampling_ratio=1.0):
    """Return a callback that evaluates the global model."""
    # Load a separate test dataset for central evaluation
    try:
        # Use multiple partitions to get a more representative test set
        test_partitions = [0, 1, 2]  # Use first 3 partitions for diversity
        X_tests = []
        y_tests = []
        
        for partition_id in test_partitions:
            # Load data with balancing applied
            _, X_test_part, _, y_test_part = load_data(
                partition_id, 25, 
                balance_method=balance_method, 
                sampling_ratio=sampling_ratio
            )
            X_tests.append(X_test_part)
            y_tests.append(y_test_part)
            
        # Concatenate the test sets
        X_test = np.vstack(X_tests)
        y_test = np.concatenate(y_tests)
        
        # Analyze the test dataset
        dataset_info = analyze_dataset(X_test, y_test)
        print(f"Central evaluation dataset: {dataset_info}")
        print(f"Central evaluation - Using balance method: {balance_method}, ratio: {sampling_ratio}")
        
        def evaluate(server_round, parameters_ndarrays, config):
            """Evaluate global model using provided centralized testset."""
            # Update model with the latest parameters
            model_copy = set_model_params(model, parameters_ndarrays)
            
            # Double-check and log basic distribution info on each evaluation
            print(f"Evaluation round {server_round}: Test set has {len(y_test)} samples")
            positive_samples = np.sum(y_test == 1)
            negative_samples = np.sum(y_test == 0)
            print(f"Class distribution: 0 (negative): {negative_samples}, 1 (positive): {positive_samples}")
            
            # Get predictions
            y_pred = model_copy.predict(X_test)
            
            # Log prediction distribution
            pred_positive = np.sum(y_pred == 1)
            pred_negative = np.sum(y_pred == 0)
            print(f"Prediction distribution: 0: {pred_negative}, 1: {pred_positive}")
            
            try:
                # Accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                # Get confusion matrix (make sure we have both classes in the true labels)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                
                if cm.shape == (2, 2):  # Ensure we have a valid 2x2 confusion matrix
                    tn, fp, fn, tp = cm.ravel()
                    
                    # Handle class imbalance - use different averaging if needed
                    if dataset_info["is_imbalanced"]:
                        average_method = 'weighted'
                    else:
                        average_method = 'binary'
                    
                    # Precision, recall, and F1 score with appropriate averaging
                    precision = precision_score(y_test, y_pred, 
                                                zero_division=0, 
                                                average=average_method)
                    recall = recall_score(y_test, y_pred, 
                                          zero_division=0, 
                                          average=average_method)
                    f1 = f1_score(y_test, y_pred, 
                                  zero_division=0, 
                                  average=average_method)
                    
                    # False positive rate and false negative rate
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    
                    # Log full confusion matrix for debugging
                    print(f"Confusion Matrix:\n{cm}")
                    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
                else:
                    print(f"Warning: Unexpected confusion matrix shape: {cm.shape}")
                    precision = recall = f1 = 0.0
                    fpr = fnr = 0.0
                
                # Calculate loss based on model type
                if loss_type == 'hinge':
                    # For SGDClassifier with hinge loss
                    y_decision = model_copy.decision_function(X_test)
                    loss = np.mean(np.maximum(0, 1 - y_test * y_decision))
                else:
                    # For LogisticRegression, with error handling
                    try:
                        y_proba = model_copy.predict_proba(X_test)
                        loss = log_loss(y_test, y_proba)
                    except Exception as e:
                        print(f"Error calculating log_loss: {e}")
                        loss = 1.0 - accuracy  # Fallback
                    
            except Exception as e:
                print(f"Error in evaluate_fn: {e}")
                accuracy = model_copy.score(X_test, y_test)
                precision = recall = f1 = fpr = fnr = 0.0
                loss = 1.0 - accuracy  # Fallback
            
            # Return all metrics
            metrics = {
                "central_accuracy": float(accuracy),
                "central_precision": float(precision),
                "central_recall": float(recall),
                "central_f1_score": float(f1),
                "central_false_positive_rate": float(fpr),
                "central_false_negative_rate": float(fnr)
            }
                
            return float(loss), metrics
        
        return evaluate
    except Exception as e:
        print(f"Could not create evaluate_fn: {e}")
        return None


def server_fn(context: Context):
    """Initialize and configure the server for federated learning."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    num_clients = context.run_config["num-clients"]
    
    # Get balancing parameters from run_config if available, otherwise use defaults 
    balance_method = context.run_config.get("balance-method", "undersample")
    sampling_ratio = float(context.run_config.get("sampling-ratio", "1.0"))
    
    # Create model based on configuration
    loss = context.run_config["loss"]
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs, loss)

    # Setting initial parameters
    set_initial_params(model)
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.8,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(model, loss, balance_method, sampling_ratio),
        evaluate_metrics_aggregation_fn=aggregated_metrics,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
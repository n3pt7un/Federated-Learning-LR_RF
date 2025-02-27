"""new-new-federation: A Flower / sklearn app."""
from typing import List, Tuple, Dict, Any, Optional, Union
from flwr.common import Context, ndarrays_to_parameters, EvaluateRes, Scalar, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, DifferentialPrivacyServerSideAdaptiveClipping
from .task import get_model, get_model_params, set_initial_params, set_model_params, load_data
from .custom_strategy import CustomFedAvg
import numpy as np


def aggregated_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from multiple clients."""
    if not metrics:
        return {}
        
    accuracy = [num_examples * m.get('accuracy', 0.0) for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    if total_examples == 0:
        return {'accuracy': 0.0}
    
    return {'accuracy': sum(accuracy)/total_examples}


def get_evaluate_fn(model, loss_type):
    """Return a callback that evaluates the global model."""
    # Load a separate test dataset for central evaluation
    try:
        # This is a simplified version - in production, you'd want to load a separate test dataset
        # Here we're using partition 0 as a proxy for simplicity
        _, X_test, _, y_test = load_data(0, 25)  
        
        def evaluate(server_round, parameters_ndarrays, config):
            """Evaluate global model using provided centralized testset."""
            # Update model with the latest parameters
            model_copy = set_model_params(model, parameters_ndarrays)
            
            # Calculate accuracy
            accuracy = model_copy.score(X_test, y_test)
            
            # For loss calculation, handle different model types
            try:
                if loss_type == 'hinge':
                    # For SGDClassifier with hinge loss
                    y_pred = model_copy.decision_function(X_test)
                    loss = np.mean(np.maximum(0, 1 - y_test * y_pred))
                else:
                    # For LogisticRegression
                    loss = 1.0 - accuracy  # Simplified loss if log_loss can't be calculated
            except Exception as e:
                print(f"Error in evaluate_fn: {e}")
                loss = 1.0 - accuracy  # Fallback
                
            return loss, {"central_accuracy": accuracy}
        
        return evaluate
    except Exception as e:
        print(f"Could not create evaluate_fn: {e}")
        return None


def server_fn(context: Context):
    """Initialize and configure the server for federated learning."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    num_clients = context.run_config["num-clients"]
    
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
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(model, loss),
        evaluate_metrics_aggregation_fn=aggregated_metrics,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
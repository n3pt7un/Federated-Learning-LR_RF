"""new-new-new-federation: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from typing import List, Tuple, Dict, Any, Optional, Union
from .our_strategy import CustomStrat
from new_new_new_federation.task import get_model, get_model_params, set_initial_params


def fit_config_fn(server_round: int) -> Dict[str, Any]:
    """
    Generate configuration for client training in each round.
    
    Args:
        server_round: Current server round number
        
    Returns:
        Dictionary with configuration for client training
    """
    config = {
        'n_estimators': 10,  # Base number of trees
        'warm_start': True  # Enable warm_start to grow trees incrementally
    }
    
    # Add more trees in later rounds
    if server_round >= 2:
        config['n_estimators'] = 15
    if server_round >= 4:
        config['n_estimators'] = 20
        
    return config


def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate evaluation metrics from multiple clients.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
        
    Returns:
        Dictionary with aggregated metrics
    """
    # Extract accuracy values weighted by number of examples
    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Calculate weighted average accuracy
    if total_examples > 0:
        return {'accuracy': sum(accuracies) / total_examples}
    else:
        return {'accuracy': 0.0}


def server_fn(context: Context) -> ServerAppComponents:
    """
    Initialize and configure the federated learning server.
    
    Args:
        context: Server context with configuration
        
    Returns:
        ServerAppComponents instance with strategy and configuration
    """
    # Read configuration
    num_rounds = int(context.run_config["num-server-rounds"])
    n_trees = int(context.run_config["n_trees"])
    max_depth = int(context.run_config["max_depth"])
    
    # Create and initialize server model
    model = get_model(n_trees=n_trees, max_depth=max_depth)
    set_initial_params(model)
    
    # Get initial parameters
    initial_parameters = ndarrays_to_parameters(get_model_params(model))
    
    # Configure the strategy
    strategy = CustomStrat(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        on_fit_config_fn=fit_config_fn,
        target_n_trees=n_trees,
    )
    
    # Configure the server
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Create and expose ServerApp
app = ServerApp(server_fn=server_fn)
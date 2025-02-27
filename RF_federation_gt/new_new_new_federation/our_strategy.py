from flwr.common import FitRes, Parameters, parameters_to_ndarrays, NDArrays, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from flwr.server.strategy.aggregate import aggregate
from flwr.common.typing import Parameters
import numpy as np
import json
from joblib import dump, load
import io
import joblib
from sklearn.ensemble import RandomForestClassifier

from .task import set_model_params, get_model, get_model_params, load_test_data


def prune_random_forest(rf_model, X_test, y_test, n_trees):
    """
    Prune a Random Forest by selecting the best n_trees based on test accuracy.

    Args:
        rf_model: Trained RandomForestClassifier model
        X_test: Test features
        y_test: Test labels
        n_trees: Number of trees to select
        
    Returns:
        A new RandomForestClassifier with only the best n_trees
    """
    # Evaluate importance of trees by their accuracy on test set
    importances = np.array([tree.score(X_test, y_test) for tree in rf_model.estimators_])
    
    # Get indices of the best trees
    best_trees_indices = np.argsort(importances)[-n_trees:]
    
    # Create a new Random Forest with the best trees
    rf_pruned = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=rf_model.random_state,
        max_depth=rf_model.max_depth,
        min_samples_split=rf_model.min_samples_split,
        min_samples_leaf=rf_model.min_samples_leaf,
        max_features=rf_model.max_features
    )
    
    # Set the selected trees in the new model
    rf_pruned.estimators_ = [rf_model.estimators_[i] for i in best_trees_indices]
    
    # Copy required attributes
    rf_pruned.classes_ = rf_model.classes_
    rf_pruned.n_classes_ = rf_model.n_classes_
    rf_pruned.n_outputs_ = rf_model.n_outputs_
    rf_pruned.n_features_in_ = rf_model.n_features_in_
    
    return rf_pruned


def merge_forest_trees(forest_models):
    """
    Merge multiple Random Forest models by combining all their trees.
    
    Args:
        forest_models: List of RandomForestClassifier models
        
    Returns:
        A merged RandomForestClassifier containing all trees from input models
    """
    if not forest_models:
        raise ValueError("No forest models provided to merge")
    
    # Collect all trees from all forests
    all_trees = []
    for forest in forest_models:
        all_trees.extend(forest.estimators_)
    
    # Create a new Random Forest with all trees
    merged_forest = RandomForestClassifier(
        n_estimators=len(all_trees),
        random_state=forest_models[0].random_state,
        max_depth=forest_models[0].max_depth,
        min_samples_split=forest_models[0].min_samples_split,
        min_samples_leaf=forest_models[0].min_samples_leaf,
        max_features=forest_models[0].max_features
    )
    
    # Set the combined trees in the merged model
    merged_forest.estimators_ = all_trees
    
    # Copy other required attributes from the first forest
    merged_forest.classes_ = forest_models[0].classes_
    merged_forest.n_classes_ = forest_models[0].n_classes_
    merged_forest.n_outputs_ = forest_models[0].n_outputs_
    merged_forest.n_features_in_ = forest_models[0].n_features_in_
    
    return merged_forest


def merge_and_prune_forests(forests, X_test, y_test, n_trees):
    """
    Merge multiple Random Forest models and prune to best n_trees.
    
    Args:
        forests: List of RandomForestClassifier models
        X_test: Test features
        y_test: Test labels
        n_trees: Number of trees to select for final model
        
    Returns:
        A RandomForestClassifier with best n_trees selected from all models
    """
    # Merge all forests
    merged_forest = merge_forest_trees(forests)
    
    # Prune the merged forest
    pruned_forest = prune_random_forest(merged_forest, X_test, y_test, n_trees)
    
    return pruned_forest


def federated_aggregate(results, target_n_trees=None):
    """
    Custom federated aggregation function for Random Forest models.
    
    Args:
        results: List of (parameters, num_examples) tuples from client models
        target_n_trees: Target number of trees in the final model
        
    Returns:
        List of NDArrays representing the aggregated global model parameters
    """
    # Convert parameters to models
    client_params = [params for params, _ in results]
    model = get_model(n_trees=10, max_depth=3)  # Base model for deserializing
    models = []
    
    # Deserialize each client's model parameters
    for params in client_params:
        client_model = get_model(n_trees=10, max_depth=3)
        models.append(set_model_params(client_model, params))
    
    # Load test data for evaluating tree performance
    X_test, y_test = load_test_data()
    
    # If target_n_trees not specified, use a reasonable default
    if target_n_trees is None:
        # Use half of total trees from all clients (but at least 10)
        total_client_trees = sum(len(model.estimators_) for model in models)
        target_n_trees = max(10, total_client_trees // 2)
    
    # Merge and prune the forests
    global_model = merge_and_prune_forests(models, X_test, y_test, target_n_trees)
    
    # Serialize the aggregated global model
    global_params = get_model_params(global_model)
    
    return global_params


class CustomStrat(FedAvg):
    """
    Custom strategy for federated Random Forest training.
    
    This strategy handles the merging and pruning of trees from client models.
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        initial_parameters: Optional[Parameters] = None,
        target_n_trees: int = 10,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
        )
        
        self.target_n_trees = target_n_trees
        self.results_to_save = {}  # Store metrics from each round
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using custom tree merging and pruning."""
        if not results:
            return None, {}
            
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # Convert results to the format expected by federated_aggregate
        formatted_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Use our custom aggregation
        aggregated_ndarrays = federated_aggregate(
            formatted_results, 
            target_n_trees=self.target_n_trees
        )
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Save the global model for this round
        model = get_model(n_trees=self.target_n_trees, max_depth=3)
        global_model = set_model_params(model, parameters_aggregated)
        dump(global_model, f'global_tree{server_round}.joblib')
        
        metrics_aggregated = {}
        
        return parameters_aggregated, metrics_aggregated
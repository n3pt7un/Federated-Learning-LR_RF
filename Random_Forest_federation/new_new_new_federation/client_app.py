"""new-new-new-federation: A Flower / sklearn app."""

import warnings
from sklearn.metrics import log_loss

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from new_new_new_federation.task import (
    get_model,
    get_model_params,
    load_data,
    set_model_params,
    set_initial_params,
)


class FlowerClient(NumPyClient):
    """
    Client for federated Random Forest model training.
    
    Handles fit and evaluate operations for client-side model training.
    """
    
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        """
        Train model on the client's local data.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration dictionary with training parameters
            
        Returns:
            Tuple of (parameters, num_examples, metrics)
        """
        # Apply server parameters to local model
        set_model_params(self.model, parameters)

        # Check if warm_start is enabled and handle tree growing
        warm_start = config.get('warm_start', False)
        if warm_start:
            # If warm_start is enabled, we may want to grow additional trees
            current_trees = len(self.model.estimators_) if hasattr(self.model, 'estimators_') else 0
            new_trees = int(config.get('n_estimators', current_trees))
            
            if new_trees > current_trees:
                self.model.set_params(n_estimators=new_trees)
                print(f"Growing forest from {current_trees} to {new_trees} trees")
        
        # Train the model on local data
        self.model.fit(self.X_train, self.y_train)
        
        # Get updated model parameters to send back to server
        updated_params = get_model_params(self.model)
        
        return updated_params, len(self.X_train), {}

    def evaluate(self, parameters, config):
        """
        Evaluate model on the client's local test data.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration dictionary with evaluation parameters
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Apply server parameters to local model
        set_model_params(self.model, parameters)

        # Calculate metrics
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        
        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context):
    """
    Initialize and configure a client with the appropriate data partition.
    
    Args:
        context: Flower client context containing configuration information
        
    Returns:
        Configured Flower client instance
    """
    print("Initializing client function...")

    # Get partition information
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    print("Partition configuration loaded")

    # Load data partition
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)
    print("Data partition loaded")

    # Create RandomForestClassifier model with parameters from context
    n_trees = int(context.run_config["n_trees"])
    max_depth = int(context.run_config["max_depth"])
    print(f"Creating model with n_trees: {n_trees}, max_depth: {max_depth}")

    model = get_model(n_trees=n_trees, max_depth=max_depth)
    print("Model created")

    # Initialize model parameters
    set_initial_params(model)
    print("Model initialized with parameters")

    print("Creating and returning FlowerClient instance")
    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
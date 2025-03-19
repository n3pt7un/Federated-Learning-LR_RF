"""new-new-new-federation: A Flower / sklearn app."""

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump
from io import BytesIO

fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition Kitsune data."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="n3p7un/KitsuneSystemAttackData_osScanDataset",
            partitioners={"train": partitioner},
        )

    dataset = fds.load_partition(partition_id, "train").with_format("numpy")

    # Get feature column names (exclude label and Unnamed: 0)
    feature_columns = [col for col in dataset.column_names if col not in ["label", "Unnamed: 0"]]

    # Extract features as a list of 1D arrays and stack them into 2D
    feature_arrays = [dataset[col] for col in feature_columns]
    X = np.column_stack(feature_arrays)  # Now shape (num_samples, num_features)
    y = dataset["label"]

    # Split data
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)):]

    return X_train, X_test, y_train, y_test


def get_model(n_trees: int, max_depth: int):
    """Create a Random Forest classifier model.
    
    Args:
        n_trees: Number of estimators (trees) in the forest
        max_depth: Maximum depth of each tree
        
    Returns:
        A RandomForestClassifier model
    """
    return RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        random_state=42,
        warm_start=True,
    )


def get_model_params(model):
    """Serialize the entire forest model to byte array.
    
    Args:
        model: RandomForestClassifier model to serialize
        
    Returns:
        List containing a single numpy array of bytes representing the serialized model
    """
    buffer = BytesIO()
    dump(model, buffer)
    buffer.seek(0)
    return [np.frombuffer(buffer.getvalue(), dtype=np.uint8)]


def set_model_params(model, params):
    """Deserialize model parameters and update the provided model.
    
    Args:
        model: RandomForestClassifier model to update
        params: List containing a single numpy array of bytes representing a serialized model
        
    Returns:
        Updated RandomForestClassifier model
    """
    buffer = BytesIO(params[0].tobytes())
    aggregated_model = load(buffer)
    
    # Copy all necessary attributes from the loaded model
    model.n_estimators = len(aggregated_model.estimators_)
    model.estimators_ = aggregated_model.estimators_
    
    # Copy other required attributes
    if hasattr(aggregated_model, "classes_"):
        model.classes_ = aggregated_model.classes_
    if hasattr(aggregated_model, "n_classes_"):
        model.n_classes_ = aggregated_model.n_classes_
    if hasattr(aggregated_model, "n_outputs_"):
        model.n_outputs_ = aggregated_model.n_outputs_
    if hasattr(aggregated_model, "n_features_in_"):
        model.n_features_in_ = aggregated_model.n_features_in_
    
    return model


def set_initial_params(model):
    """Initialize model with dummy data to ensure it has all required attributes.
    
    Args:
        model: RandomForestClassifier model to initialize
    """
    n_classes = 2  # Dataset has 2 classes
    n_features = 115  # Number of features in dataset
    n_samples = 100
    X = np.random.randn(n_samples, n_features)  # Random features from a normal distribution
    y = np.random.randint(0, n_classes, size=n_samples)
    model.fit(X, y)


def load_test_data():
    """Load centralized test dataset for model evaluation.
    
    Returns:
        X_test: Test features
        y_test: Test labels
    """
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=2)  # Minimum partitioning
        fds = FederatedDataset(
            dataset="n3p7un/KitsuneSystemAttackData_osScanDataset",
            partitioners={"train": partitioner},
        )
    
    # Use partition 0 as a proxy for test data
    dataset = fds.load_partition(0, "train").with_format("numpy")
    
    # Get feature column names (exclude label and Unnamed: 0)
    feature_columns = [col for col in dataset.column_names if col not in ["label", "Unnamed: 0"]]
    
    # Extract features as a list of 1D arrays and stack them into 2D
    feature_arrays = [dataset[col] for col in feature_columns]
    X = np.column_stack(feature_arrays)
    y = dataset["label"]
    
    # Use 20% of the data as test set (different from what clients use)
    test_size = int(0.2 * len(X))
    X_test = X[:test_size]
    y_test = y[:test_size]
    
    return X_test, y_test
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

    #print(f"X_train shape: {X_train.shape}")  # Should be (samples, features)
    #print(f"X_test shape: {X_test.shape}")  # Should be 2D
    return X_train, X_test, y_train, y_test



def get_model(n_trees: int, max_depth: int) -> RandomForestClassifier:
    """Create a new RandomForestClassifier with specified parameters."""
    return RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        random_state=42,
        warm_start=True,
    )


def get_model_params(model):
    """Qui stiamo serailizzando l'intera foresta"""
    buffer = BytesIO()
    dump(model, buffer)
    buffer.seek(0)
    return [np.frombuffer(buffer.getvalue(), dtype=np.uint8)]


def set_model_params(model, params):
    """Deserializzazione"""
    buffer = BytesIO(params[0].tobytes())
    aggregated_model = load(buffer)
    model.n_estimators = len(aggregated_model.estimators_)
    model.estimators_ = aggregated_model.estimators_
    return model


def set_initial_params(model):
    n_classes = 2  # Dataset has 2 classes
    n_features = 115  # Number of features in dataset
    n_samples = 100
    X = np.random.randn(n_samples, n_features)  # Random features from a normal distribution
    y = np.random.randint(0, n_classes, size=n_samples)
    model.fit(X, y)

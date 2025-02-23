"""new-new-federation: A Flower / sklearn app."""

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression

fds = None  # Cache FederatedDataset


def load_data_old(partition_id: int, num_partitions: int):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="n3p7un/KitsuneSystemAttackData_osScanDataset",
            partitioners={"train": partitioner},
        )

    dataset = fds.load_partition(partition_id, "train").with_format("numpy")
    features = np.column_stack(dataset.remove_columns(['label', 'Unnamed: 0']))
    features = features.reshape(features.shape[0], -1)
    X, y = features, dataset["label"]

    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]
    print(f"X_train shape: {X_train.shape}")  # Should be (num_samples, num_features)
    print(f"X_test shape: {X_test.shape}")  # Should be 2D
    return X_train, X_test, y_train, y_test

def load_data(partition_id: int, num_partitions: int):
    """Load partition MNIST data."""
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


def get_model(penalty: str, local_epochs: int):

    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
    )


def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    n_classes = 2  # Dataset has 10 classes
    n_features = 115  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

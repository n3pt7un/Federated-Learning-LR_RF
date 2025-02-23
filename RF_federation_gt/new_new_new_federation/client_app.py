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
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        # parameters.append(config['warm_start'])
        set_model_params(self.model, parameters)

        # Aggiorniamo n_estimators solo se warm_start è abilitato
        if config.get('warm_start', False):
            current_trees = self.model.n_estimators
            new_trees = current_trees + config.get('n_estimators_increment', 0)
            self.model.set_params(n_estimators=new_trees)

        self.model.fit(self.X_train, self.y_train)
        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context):
    print("Initializing client function...")

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    print("Partition done")

    # Load data
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)
    print("Data loaded")

    # Create LogisticRegression Model
    n_trees = context.run_config["n_trees"]
    max_depth = context.run_config["max_depth"]
    print(f"Model parameters - n_trees: {n_trees}, max_depth: {max_depth}")

    model = get_model(n_trees, max_depth)
    print("Model initialized.")

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)
    print("Initial model parameters set.")

    print("Returning FlowerClient instance.")
    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)

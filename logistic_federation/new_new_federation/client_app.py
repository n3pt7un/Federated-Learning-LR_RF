"""new-new-federation: A Flower / sklearn app."""

import warnings
import numpy as np
from sklearn.metrics import log_loss
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


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, loss):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.loss = loss

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        try:
            if self.loss == 'hinge':
                # SGDClassifier with hinge loss doesn't have predict_proba
                y_pred = self.model.decision_function(self.X_test)
                # Can't use log_loss with hinge SVM, use another metric
                loss = np.mean(np.maximum(0, 1 - self.y_test * y_pred))
            else:
                # For LogisticRegression
                y_proba = self.model.predict_proba(self.X_test)
                loss = log_loss(self.y_test, y_proba)
        except Exception as e:
            # Fallback to a simple loss calculation if predict_proba fails
            print(f"Error calculating loss: {e}")
            y_pred = self.model.predict(self.X_test)
            loss = np.mean(y_pred != self.y_test)

        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

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
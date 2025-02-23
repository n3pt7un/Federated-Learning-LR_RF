"""new-new-new-federation: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from new_new_new_federation.task import get_model, get_model_params, set_initial_params
from typing import List, Tuple, Dict, Any, Optional, Union
from .our_strategy import CustomStrat


#def get_evaluate_fn(testloader, device):


def on_fit_config(server_round: int) -> Metrics:
    n_trees = 10
    if server_round >= 1:
        n_trees += 5
    return  {'n_estimators': n_trees}

def aggregated_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracy = [num_examples * m['accuracy'] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {'accuracy': sum(accuracy)/total_examples}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Create RF Model
    n_trees = context.run_config["n_trees"]
    max_depth = context.run_config["max_depth"]
    model = get_model(n_trees, max_depth)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy
    strategy = CustomStrat(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=aggregated_metrics,
        on_fit_config_fn=on_fit_config,
        context=context,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

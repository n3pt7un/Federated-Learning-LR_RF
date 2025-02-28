"""new-new-federation: A Flower / sklearn app."""
from typing import List, Tuple, Dict, Any, Optional, Union
from flwr.common import Context, ndarrays_to_parameters, EvaluateRes, Scalar, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from new_new_federation.task import get_model, get_model_params, set_initial_params


def aggregated_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracy = [num_examples * m['accuracy'] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {'accuracy': sum(accuracy)/total_examples}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=aggregated_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

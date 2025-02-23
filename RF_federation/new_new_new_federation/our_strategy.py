from flwr.common import FitRes, Parameters, parameters_to_ndarrays,ndarrays_to_parameters, NDArrays, Scalar, Context
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Any, Optional, Union
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

import json
from joblib import dump, load

from .task import set_model_params, get_model, get_model_params, fds

from sklearn.ensemble import RandomForestClassifier
import numpy as np


#ho già testato questa funzione e (gioco di parole) funziona
def prune_random_forest(rf_model, X_test, y_test, n):
    """
    Funzione per potare una Random Forest già addestrata.

    Parametri:
    - rf_model: Modello RandomForestClassifier già addestrato.
    - X_test: Dati di test (features).
    - y_test: Etichette di test (target).
    - n: Numero di alberi da selezionare.

    Restituisce:
    - rf_pruned: Random Forest potata con i primi n alberi migliori.
    """
    # Qui stiamo valutando l'importanza degli alberi utilizzando l'accuratezza sul test set
    importances = np.array([tree.score(X_test, y_test) for tree in rf_model.estimators_])

    best_trees_indices = np.argsort(importances)[-n:]

    # Creiamo una nuova Random Forest con i migliori alberi
    rf_pruned = RandomForestClassifier(
        n_estimators=n,
        random_state=rf_model.random_state,
        max_depth=rf_model.max_depth,
        min_samples_split=rf_model.min_samples_split,
        min_samples_leaf=rf_model.min_samples_leaf,
        max_features=rf_model.max_features
    )

    # ti imposta gli alberi selezionati nel nuovo modello
    rf_pruned.estimators_ = [rf_model.estimators_[i] for i in best_trees_indices]

    # Impostiamo gli attributi necessari per il funzionamento del modello
    rf_pruned.classes_ = rf_model.classes_
    rf_pruned.n_classes_ = rf_model.n_classes_
    rf_pruned.n_outputs_ = rf_model.n_outputs_
    rf_pruned.n_features_in_ = rf_model.n_features_in_

    return rf_pruned


def merge_and_prune_forests(forests, X_test, y_test, n):
    """
    Funzione per deserializzare, unire e potare più Random Forest.

    Parametri:
    - serialized_forests: Lista di modelli RandomForestClassifier serializzati (in formato byte).
    - X_test: Dati di test (features).
    - y_test: Etichette di test (target).
    - n: Numero di alberi da selezionare dopo la potatura.

    Restituisce:
    - rf_pruned: Random Forest potata con i primi n alberi migliori.
    """
    # Deserializza tutti i modelli
    # forests = [joblib.load(io.BytesIO(forest_bytes)) for forest_bytes in serialized_forests]

    # Unisci tutti gli alberi di tutte le foreste
    all_trees = []
    for forest in forests:
        all_trees.extend(forest.estimators_)

    # Crea una nuova Random Forest con tutti gli alberi
    merged_forest = RandomForestClassifier(
        n_estimators=len(all_trees),
        random_state=forests[0].random_state,  # Usa il random_state della prima foresta
        max_depth=forests[0].max_depth,
        min_samples_split=forests[0].min_samples_split,
        min_samples_leaf=forests[0].min_samples_leaf,
        max_features=forests[0].max_features
    )

    # Imposta gli alberi uniti nel nuovo modello
    merged_forest.estimators_ = all_trees

    # Imposta gli attributi necessari per il funzionamento del modello
    merged_forest.classes_ = forests[0].classes_
    merged_forest.n_classes_ = forests[0].n_classes_
    merged_forest.n_outputs_ = forests[0].n_outputs_
    merged_forest.n_features_in_ = forests[0].n_features_in_

    # Applica la funzione di potatura alla foresta unita
    pruned_forest = prune_random_forest(merged_forest, X_test, y_test, n)

    return pruned_forest


def our_aggregate(results: List[Tuple[ClientProxy, FitRes]], context: Context) -> NDArrays:
    """Aggregate the trained random forest models from multiple clients."""
    # Extract parameters from each client's results
    client_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
    
    # Create models from parameters
    models = []
    for params in client_params:
        model = get_model(n_trees=10, max_depth=3)
        model = set_model_params(model, params)
        models.append(model)

    # Initialize FederatedDataset for test data
    global fds
    if fds is None:
        from flwr_datasets.partitioner import IidPartitioner
        partitioner = IidPartitioner(num_partitions=10)  # Adjust number as needed
        fds = FederatedDataset(
            dataset="n3p7un/KitsuneSystemAttackData_osScanDataset",
            partitioners={"train": partitioner},
        )

    # Load centralized test dataset
    dataset = fds.load_partition(0, "train").with_format('numpy')  # Using first partition as test
    
    # Get feature column names (exclude label and Unnamed: 0)
    feature_columns = [col for col in dataset.column_names if col not in ["label", "Unnamed: 0"]]
    # Extract features as a list of 1D arrays and stack them into 2D
    feature_arrays = [dataset[col] for col in feature_columns]
    X_test = np.column_stack(feature_arrays)
    y_test = dataset["label"]

    # Get number of trees from config and calculate global trees
    n_trees = int(context.run_config["n_trees"])
    n_trees_global = int(np.ceil(n_trees / 2))
    
    # Merge and prune forests
    global_tree = merge_and_prune_forests(models, X_test, y_test, n_trees_global)
    
    # Get parameters from global tree
    return get_model_params(global_tree)



class CustomStrat(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn = None,
        on_fit_config_fn = None,
        initial_parameters = None,
        context: Context,  # Add context parameter
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=initial_parameters,
        )
        self.context = context  # Store the context
        self.results_to_save = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate parameters using our custom method
        aggregated_ndarrays = our_aggregate(results, self.context)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Save the global model for this round
        try:
            model = get_model(n_trees=10, max_depth=3)
            model = set_model_params(model, aggregated_ndarrays)
            dump(model, f'global_tree{server_round}.joblib')
        except Exception as e:
            print(f"Warning: Could not save global model: {e}")

        return parameters_aggregated, {}
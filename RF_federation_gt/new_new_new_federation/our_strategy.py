from flwr.common import FitRes, Parameters, parameters_to_ndarrays, NDArrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Any, Optional, Union

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



import io
import joblib

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


def our_aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
    # Gets model params for each node --> format of get_model_params() output
    client_params = [fit_res.parameters for (fit_res, _) in results]
    #client_params = [params for params, _ in results]
    model = get_model(n_trees=10, max_depth=3)
    models = [set_model_params(model, params) for params in client_params]

    # Load centralized test dataset from fds partition
    test = fds.load_partition('test')[0].with_format('numpy')
    # Get feature column names (exclude label and Unnamed: 0)
    feature_columns = [col for col in dataset.column_names if col not in ["label", "Unnamed: 0"]]
    # Extract features as a list of 1D arrays and stack them into 2D
    feature_arrays = [dataset[col] for col in feature_columns]
    X_test = np.column_stack(feature_arrays)
    y_test = dataset["label"]

    n_trees = context.run_config["n_trees"]
    n_trees_global = np.ceil(n_trees / 2)
    global_tree = merge_and_prune_forests(models, X_test, y_test, n_trees_global)
    global_params = get_model_params(global_tree)

    return global_params



class CustomStrat(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}

    # To modify
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        aggregated_ndarrays = our_aggregate(results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        model = get_model(n_trees=10,max_depth=3)
        global_model = set_model_params(model, parameters_aggregated)
        dump(model, f'global_tree{server_round}.joblib')

        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated
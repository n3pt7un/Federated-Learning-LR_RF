from flwr.common import FitRes, Parameters, parameters_to_ndarrays, Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import json
import numpy as np
import logging
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CustomFedAvg")


# Helper function to convert NumPy types to Python native types
def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}
        
        # Create directory for model checkpoints if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)

    def configure_fit(
        self, server_round: int, parameters, client_manager
    ):
        """Configure the next round of training."""
        config = {}
        
        # Add round information to the configuration
        config["round"] = server_round
        logger.info(f"Configuring round {server_round} with {config}")
        
        # Get clients and their fit_ins from base class
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Update configurations with round information
        for _, fit_ins in client_instructions:
            fit_ins.config.update(config)
            
        return client_instructions

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate received model updates and metrics, save global model checkpoint."""
        
        # Log information about this round
        if results:
            logger.info(f"Round {server_round}: Aggregating updates from {len(results)} clients")
        
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} clients failed")

        # Call the default aggregate_fit method from FedAvg
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        # If we couldn't aggregate parameters (e.g., because no client successfully sent parameters)
        if parameters_aggregated is None:
            logger.error(f"Round {server_round}: No parameters could be aggregated!")
            return None, {}

        try:
            # Convert parameters to ndarrays
            ndarrays = parameters_to_ndarrays(parameters_aggregated)

            # Log the shapes of the aggregated parameters
            shapes = [arr.shape for arr in ndarrays]
            logger.info(f"Shapes of aggregated parameters: {shapes}")

            # Save each parameter array separately instead of trying to save as one array
            checkpoint_path = f'checkpoints/global_model_round_{server_round}.pkl'
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(ndarrays, f)
            
            # Also save as the latest model
            with open('global_model_latest.pkl', 'wb') as f:
                pickle.dump(ndarrays, f)
            
            # Save individual arrays for backup as well
            for i, array in enumerate(ndarrays):
                np.save(f'checkpoints/global_model_round_{server_round}_param_{i}.npy', array)
                
            logger.info(f"Global model for round {server_round} saved successfully.")
            
            # Add to metrics that the model was saved
            metrics_aggregated["model_saved"] = True
            
        except Exception as e:
            logger.error(f"Error while saving global model: {e}", exc_info=True)
            metrics_aggregated["model_saved"] = False

        # Return the expected outputs for `aggregate_fit`
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Evaluate global model, then save metrics to local JSON."""
        # Call the default behaviour from FedAvg
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_result is None:
            logger.warning(f"Round {server_round}: No evaluation results could be aggregated!")
            return None, {}
            
        loss, metrics = aggregated_result
        
        # Convert loss and metrics to Python native types for serialization
        loss = float(loss) if loss is not None else None
        metrics = convert_to_serializable(metrics)
        
        # Log detailed metrics information
        logger.info(f"Round {server_round} - Loss: {loss:.4f}")
        
        # Log each metric individually with detailed information
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                logger.info(f"Round {server_round} - {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"Round {server_round} - {metric_name}: {metric_value}")
            
        # Store metrics as dictionary
        my_results = {"round": server_round, "loss": loss, **metrics}
        
        # Insert into local dictionary
        self.results_to_save[str(server_round)] = my_results  # Convert to string for JSON compatibility
        
        try:
            # Save metrics as json
            with open("results.json", "w") as json_file:
                json.dump(self.results_to_save, json_file, indent=4)
            logger.info(f"Saved metrics for round {server_round} to results.json")
            
            # Additionally save as CSV for easy plotting
            try:
                import pandas as pd
                
                # Convert results dictionary to DataFrame
                df = pd.DataFrame.from_dict(self.results_to_save, orient='index')
                
                # Save to CSV
                df.to_csv("results.csv", index_label="round")
                logger.info(f"Saved metrics to results.csv for easier analysis")
            except ImportError:
                logger.warning("Pandas not installed, skipping CSV export")
            except Exception as e:
                logger.error(f"Error saving CSV: {e}")
                
        except Exception as e:
            logger.error(f"Error saving metrics to JSON: {e}")

        # Return the expected outputs for `evaluate`
        return loss, metrics
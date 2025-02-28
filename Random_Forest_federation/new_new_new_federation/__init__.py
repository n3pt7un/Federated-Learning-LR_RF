"""RF_federation_gt: Federated Random Forest learning with Flower."""

from flwr.common.logger import log
from logging import INFO

# Set logging level
log(INFO, "Federated Random Forest with tree merging and pruning")

# These imports are required for the Flower app to work correctly
from new_new_new_federation.server_app import app as server_app
from new_new_new_federation.client_app import app as client_app

__all__ = ["server_app", "client_app"]
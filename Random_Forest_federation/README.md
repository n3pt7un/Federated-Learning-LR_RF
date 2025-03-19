# Federated Random Forest with Flower

This project implements a federated Random Forest classifier using Flower (flwr). It allows multiple clients to collaboratively train a Random Forest model without sharing their raw data, using a custom aggregation strategy to merge and prune decision trees.

## Features

- Federated training of Random Forest classifiers
- Custom aggregation strategy for merging and pruning trees
- Warm-start mechanism for incremental tree growth
- Tree selection based on performance metrics
- Centralized evaluation on test dataset
- Model serialization and deserialization

## Installation

1. Make sure you have Python 3.8+ installed.
2. Install the required packages:

```bash
pip install flwr scikit-learn numpy pandas
```

## Project Structure

```
RF_federation_gt/
│
├── new_new_new_federation/
│   ├── __init__.py
│   ├── client_app.py     # Client-side implementation
│   ├── our_strategy.py   # Custom federated aggregation strategy
│   ├── server_app.py     # Server-side implementation
│   └── task.py           # Utility functions for data loading and model handling
│
├── main.ipynb            # Example notebook for testing
└── README.md
```

## How It Works

### Federated Learning Process

1. **Initialization**: The server initializes a Random Forest model with a specified number of trees.
2. **Client Training**: Each client receives the global model parameters, trains the model on their local data, and returns the updated model parameters.
3. **Tree Aggregation**: The server combines trees from all client models.
4. **Tree Pruning**: The merged forest is pruned to select the best-performing trees based on evaluation metrics.
5. **Communication**: Only model parameters (serialized models) are communicated between clients and server, not the raw data.

### Custom Random Forest Aggregation

The custom aggregation strategy works as follows:

1. Deserialize each client's model parameters into a RandomForestClassifier.
2. Collect all trees from all client models into a single merged forest.
3. Evaluate the performance of each tree on a test dataset.
4. Select the top-performing trees to form the pruned global model.
5. Serialize the global model for distribution to clients in the next round.

## Usage

### Running with the Simulation Engine

In the `RF_federation_gt` directory, use `flwr run` to run a local simulation:

```bash
flwr run new_new_new_federation
```

### Running with the Deployment Engine

For distributed deployment:

1. Start the server:
```bash
python -m new_new_new_federation.server_app
```

2. Start multiple clients:
```bash
python -m new_new_new_federation.client_app
```

## Configuration

The system can be configured with various parameters:

- `n_trees`: Number of trees in the Random Forest
- `max_depth`: Maximum depth of each tree
- `num-server-rounds`: Number of federated learning rounds
- `warm_start`: Enable/disable incremental tree growth
- `target_n_trees`: Target number of trees in the final pruned model

## Model Saving

After each round, the global model is saved to a file named `global_tree{round_number}.joblib` for later use or evaluation.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
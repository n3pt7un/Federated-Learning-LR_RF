# new-new-federation: A Flower / sklearn app

This project implements federated learning for classification using scikit-learn models with the Flower framework. The implementation has been significantly optimized to handle class imbalance and improve performance.

## Key Features and Optimizations

### Class Imbalance Handling
- **Balanced Sampling**: Implemented both undersampling and oversampling techniques
  - Undersampling: Reduces majority class examples to create balanced datasets
  - Oversampling: Increases minority class examples through replication
- **Stratified Splits**: Ensures training/test splits maintain class proportions
- **Sampling Ratio Control**: Configurable ratio parameter to fine-tune balance level
- **Balanced Evaluation**: Metrics calculation accounts for class imbalance

### Model Improvements
- **Better Initialization**: Replaced zero-weight initialization with random normal initialization
  - Prevents symmetry issues and dead neurons
  - Improves convergence speed and final performance
- **Learning Rate Scheduling**: Implements adaptive learning rates that decrease over rounds
  - Starting with higher rates for faster initial convergence
  - Gradually decreasing to fine-tune at later stages
- **Increased Training**: More local epochs (20) and federated rounds (10)
- **Advanced Optimization Settings**: Configured adaptive learning rates, tolerance parameters
- **Warm Start Optimization**: Enhanced efficiency with proper warm start implementation

### Data Processing Optimizations
- **Feature Standardization**: All features are standardized using StandardScaler
  - Consistent scaling across all clients
  - Improved numerical stability and convergence
- **Memory Optimization**: Using float32 instead of float64 for reduced memory usage
- **Multi-level Caching System**:
  - In-memory caching of processed data partitions
  - Persistent disk caching for repeated runs
  - Global scaler persistence for consistency

### Performance Enhancements
- **Logging Optimization**: Reduced verbose logging for faster execution
- **Parallel Processing**: Leveraging multi-core processing with n_jobs=-1
- **Convergence Improvements**: Better handling of model parameters between rounds
- **Early Stopping Capability**: Added capability to stop when performance plateaus

### Visualization Tools
- **Comprehensive Analysis Scripts**: Added tools to visualize training progress
- **Multiple Metric Tracking**: Loss, accuracy, precision, recall, F1, error rates
- **Performance Summary**: Automatic generation of improvement statistics
- **Training Dashboard**: Combined visualization of all key metrics

## Project Setup

### Install dependencies and project

```bash
pip install -e .
```

### Run with the Simulation Engine

In the `new-new-federation` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

### Visualize Results

After training completes, visualize your results with:

```bash
# On Windows
visualize.bat

# On Unix/Linux
python install_viz_deps.py
python visualize_results.py
```

Visualization outputs will be saved to the `plots` directory.

## Configuration Options

Key parameters can be configured in `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 10      # Number of federated rounds
penalty = "l2"              # Regularization type
local-epochs = 15           # Per-client training iterations
loss = 'hinge'           
num-clients = 25            # Number of federated clients
balance-method = "undersample" # "undersample", "oversample", or "none"
sampling-ratio = 2.0        # Balance ratio for class handling
```

7800 train set batch size, 1900 test set batch size. Repreated across 25 nodes.

Approximate training time(Mac Mini M4 16Gb ram): 58.40s - 10 rounds

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)

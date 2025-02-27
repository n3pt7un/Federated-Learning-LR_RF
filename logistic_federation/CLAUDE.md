# CLAUDE.md for Federated Logistic Regression Project

## Build/Run Commands
- Install: `pip install -e .`
- Run simulation: `flwr run .`
- Run tests: `pytest`
- Lint: `flake8 new_new_federation/`
- Type checking: `mypy new_new_federation/`

## Project Structure
- `server_app.py`: Server-side logic for federated learning
- `client_app.py`: Client implementation using NumPyClient
- `task.py`: Data loading and model parameter handling
- `custom_strategy.py`: Extends FedAvg for model checkpointing

## Code Style Guidelines
- **Imports**: Group in order: stdlib, third-party, local
- **Typing**: Use type hints for all function parameters and returns
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Error handling**: Use try/except with specific exceptions
- **Comments**: Docstrings for all public functions and classes
- **Formatting**: Max line length 88 characters
- **Global model saving**: Use consistent naming convention for checkpoints
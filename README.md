# Federated Learning with Flower

This repository contains implementations of Federated Learning using Flower (flwr) for different Machine Learning algorithms. The goal is to enable multiple clients to collaborate in model training without directly sharing their data, preserving privacy. Each method has its own customized aggregation strategy.

## Repository Structure

The repository is divided into three main directories, each implementing a different Machine Learning algorithm in a federated setting. Each folder includes its own README with more details on the specific implementation:

### 1. **Federated Logistic Regression**
- **Directory:** `new-new-federation-logistic/`
- **Description:** Implements a federated logistic regression model using Flower and scikit-learn.
- **Aggregation:** Model weights are averaged across clients.
- **Execution:**
  - Local simulation:  
    ```bash
    flwr run .
    ```
  - Distributed deployment following the Flower documentation.

### 2. **Federated SVM with Stochastic Gradient Descent (SGD)**
- **Directory:** `new-new-federation-svm/`
- **Description:** Implements a federated SVM model trained with Stochastic Gradient Descent (SGD).
- **Aggregation:** Gradients are aggregated by the server to update the global model.
- **Execution:**
  - Local simulation:  
    ```bash
    flwr run .
    ```
  - Distributed deployment following the Flower documentation.

### 3. **Federated Random Forest**
- **Directory:** `RF_federation_gt/`
- **Description:** Implements a federated Random Forest where clients train their own Random Forest models locally, and the server aggregates the trees into a single global forest.
- **Aggregation:** Trees are merged and pruned based on performance metrics.
- **Execution:**
  - Local simulation:  
    ```bash
    flwr run new_new_new_federation
    ```
  - Distributed deployment:
    - Start the server:  
      ```bash
      python -m new_new_new_federation.server_app
      ```
    - Start the clients:  
      ```bash
      python -m new_new_new_federation.client_app
      ```

 
## Useful Resources
- [Flower Official Website](https://flower.ai)  
- [Flower Documentation](https://flower.ai/docs)  
- [Flower GitHub](https://github.com/adap/flower)  
- [Flower Community Slack](https://flower.ai/slack)  

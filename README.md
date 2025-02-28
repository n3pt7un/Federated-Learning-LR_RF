# Federated Learning with Flower

Questa repository contiene implementazioni di Federated Learning utilizzando Flower (flwr) per diversi algoritmi di Machine Learning. L'obiettivo è consentire a più client di collaborare nell'addestramento di modelli senza condividere direttamente i loro dati, preservando la privacy. Ogni metodo ha la sua strategia di aggregazione personalizzata.

## Struttura della Repository

La repository è suddivisa in tre directory principali, ognuna delle quali implementa un diverso algoritmo di Machine Learning in un contesto federato. Ogni cartella include il proprio README con maggiori dettagli sulla specifica implementazione:

### 1. **Federated Random Forest**
- **Directory:** `RF_federation_gt/`
- **Descrizione:** Implementa un Random Forest federato, in cui i client addestrano i propri modelli di Random Forest localmente e il server aggrega gli alberi in un'unica foresta globale.
- **Aggregazione:** Gli alberi vengono uniti e potati in base a metriche di performance.
- **Esecuzione:**
  - Simulazione locale:  
    ```bash
    flwr run new_new_new_federation
    ```
  - Deployment distribuito:
    - Avviare il server:  
      ```bash
      python -m new_new_new_federation.server_app
      ```
    - Avviare i client:  
      ```bash
      python -m new_new_new_federation.client_app
      ```

### 2. **Federated Logistic Regression**
- **Directory:** `new-new-federation-logistic/`
- **Descrizione:** Implementa una regressione logistica federata utilizzando Flower e scikit-learn.
- **Aggregazione:** I pesi dei modelli vengono mediati tra i client.
- **Esecuzione:**
  - Simulazione locale:  
    ```bash
    flwr run .
    ```
  - Deployment distribuito seguendo la documentazione di Flower.

### 3. **Federated SVM con Stochastic Gradient Descent (SGD)**
- **Directory:** `new-new-federation-svm/`
- **Descrizione:** Implementa un modello SVM federato addestrato con Stochastic Gradient Descent (SGD).
- **Aggregazione:** I gradienti vengono aggregati dal server per aggiornare il modello globale.
- **Esecuzione:**
  - Simulazione locale:  
    ```bash
    flwr run .
    ```
  - Deployment distribuito seguendo la documentazione di Flower.

## Requisiti

Assicurati di avere Python 3.8+ installato e installa i pacchetti richiesti con:
```bash
pip install -e .

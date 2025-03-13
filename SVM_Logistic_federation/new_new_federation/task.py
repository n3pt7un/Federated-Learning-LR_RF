"""new-new-federation: A Flower / sklearn app."""

import numpy as np
import logging
import os
import pickle
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

# Configure logging with less verbose output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Federation")

# Global variables
fds = None  # Cache FederatedDataset
scaler = StandardScaler()  # Create a global scaler for feature normalization
partition_cache = {}  # Cache for processed data partitions
scaler_fitted = False  # Track if the scaler has been fitted

# Create a directory for persistent cache
os.makedirs("data_cache", exist_ok=True)

def load_data(partition_id: int, num_partitions: int, balance_method='undersample', sampling_ratio=1.0):
    """Load partition data with optional balancing of classes.
    
    Args:
        partition_id: ID of the partition to load
        num_partitions: Total number of partitions
        balance_method: Method to balance classes ('undersample', 'oversample', or None)
        sampling_ratio: For undersampling - ratio of negative:positive samples (e.g., 1.0 = equal)
                       For oversampling - multiplier for minority class (e.g., 3.0 = triple)
    """
    # Check memory cache first
    cache_key = f"{partition_id}_{balance_method}_{sampling_ratio}"
    if cache_key in partition_cache:
        logger.info(f"Using in-memory cached data for partition {partition_id}")
        return partition_cache[cache_key]
    
    # Check disk cache next
    cache_file = f"data_cache/partition_{cache_key}.pkl"
    if os.path.exists(cache_file):
        try:
            logger.info(f"Loading cached data from disk for partition {partition_id}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                # Store in memory cache too
                partition_cache[cache_key] = data
                return data
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
    
    logger.info(f"Processing partition {partition_id}/{num_partitions}")
    
    global fds, scaler, scaler_fitted
    if fds is None:
        logger.info("Initializing FederatedDataset")
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="n3p7un/KitsuneSystemAttackData_osScanDataset",
            partitioners={"train": partitioner},
        )
    
    dataset = fds.load_partition(partition_id, "train").with_format("numpy")
    
    # Get feature column names (exclude label and Unnamed: 0)
    feature_columns = [col for col in dataset.column_names if col not in ["label", "Unnamed: 0"]]
    
    # Extract features into a 2D array more efficiently - use float32 for memory efficiency
    X = np.array([dataset[col] for col in feature_columns], dtype=np.float32).T
    y = dataset["label"]
    
    # Consolidated logging for dataset information
    logger.info(f"Partition {partition_id}: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Apply feature standardization - only fit on first call
    global_scaler_file = "data_cache/global_scaler.pkl"
    
    if os.path.exists(global_scaler_file) and not scaler_fitted:
        # Load previously fitted scaler
        try:
            with open(global_scaler_file, 'rb') as f:
                scaler = pickle.load(f)
                scaler_fitted = True
                logger.info("Loaded pre-fitted scaler from disk")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")
    
    if not scaler_fitted:
        X = scaler.fit_transform(X)
        scaler_fitted = True
        logger.info("Fitted new StandardScaler to features")
        
        # Save the scaler for future runs
        try:
            with open(global_scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
        except Exception as e:
            logger.warning(f"Failed to save scaler: {e}")
    else:
        X = scaler.transform(X)
    
    # Apply class balancing if needed
    if balance_method:
        # Separate samples by class
        X_neg = X[y == 0]
        y_neg = y[y == 0]
        X_pos = X[y == 1] 
        y_pos = y[y == 1]
        
        if balance_method == 'undersample':
            # Undersample the majority class (0)
            target_neg_count = int(len(X_pos) * sampling_ratio)
            if len(X_neg) > target_neg_count:
                indices = np.random.choice(len(X_neg), size=target_neg_count, replace=False)
                X_neg_sampled = X_neg[indices]
                y_neg_sampled = y_neg[indices]
            else:
                X_neg_sampled = X_neg
                y_neg_sampled = y_neg
                
            # Combine the balanced dataset
            X = np.vstack((X_neg_sampled, X_pos))
            y = np.concatenate((y_neg_sampled, y_pos))
            
        elif balance_method == 'oversample':
            # Oversample the minority class (1)
            target_pos_count = int(len(X_pos) * sampling_ratio)
            indices = np.random.choice(len(X_pos), size=target_pos_count, replace=True)
            X_pos_oversampled = X_pos[indices]
            y_pos_oversampled = y_pos[indices]
            
            # Combine the balanced dataset
            X = np.vstack((X_neg, X_pos_oversampled))
            y = np.concatenate((y_neg, y_pos_oversampled))
            
        # Shuffle the balanced dataset
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        logger.info(f"Applied {balance_method}, new balance: {np.mean(y):.2f} positive ratio")
    
    # Use stratification to maintain class balance in train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Bundle the result
    result = (X_train, X_test, y_train, y_test)
    
    # Store in memory cache
    partition_cache[cache_key] = result
    
    # Store on disk for future runs
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Saved processed data to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save to disk cache: {e}")
    
    return result


def get_model(penalty: str, local_epochs: int, loss: str):
    """Create and return a classifier model.
    
    Uses SGDClassifier for 'hinge' loss (SVM) and LogisticRegression for 'log_loss'.
    
    Args:
        penalty: Regularization type ('l1', 'l2', 'elasticnet', or None)
        local_epochs: Number of training epochs/iterations
        loss: Loss function ('hinge' for SVM, 'log_loss' for LogReg)
    """
    logger.info(f"Creating model with loss={loss}, penalty={penalty}, local_epochs={local_epochs}")
    
    if loss == 'hinge':
        return SGDClassifier(
            loss=loss,
            penalty=penalty,
            max_iter=local_epochs,
            warm_start=True,
            learning_rate='adaptive',  # Add adaptive learning rate
            eta0=0.01,  # Initial learning rate
            tol=1e-4,   # Convergence tolerance
            n_jobs=-1,  # Use all available cores
            random_state=42,  # For reproducibility
        )
    else:  # default to LogisticRegression for 'log_loss'
        return LogisticRegression(
            penalty=penalty,
            max_iter=local_epochs*5,  # Increase iterations for better convergence
            warm_start=True,
            solver='saga',  # Efficient solver for L1/L2 penalties
            C=1.0,         # Inverse of regularization strength
            tol=1e-4,      # Convergence tolerance
            n_jobs=-1,     # Use all available cores
            random_state=42,  # For reproducibility
        )


def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    """Set initial parameters for a model.
    
    Uses random initialization for better convergence instead of zeros.
    """
    n_classes = 2  # Dataset has 2 classes
    n_features = 115  # Number of features in dataset
    
    logger.info("Setting initial model parameters with random initialization")
    model.classes_ = np.array([i for i in range(n_classes)])

    if isinstance(model, SGDClassifier):
        # Use random initialization instead of zeros
        model.coef_ = np.random.normal(0, 0.01, (1, n_features))
        if model.fit_intercept:
            model.intercept_ = np.random.normal(0, 0.01, (1))
        logger.info(f"Initialized SGDClassifier with random weights: mean={np.mean(model.coef_):.4f}, std={np.std(model.coef_):.4f}")
    else:  # LogisticRegression
        # Use random initialization instead of zeros
        model.coef_ = np.random.normal(0, 0.01, (1, n_features))
        if model.fit_intercept:
            model.intercept_ = np.random.normal(0, 0.01, (1))
        logger.info(f"Initialized LogisticRegression with random weights: mean={np.mean(model.coef_):.4f}, std={np.std(model.coef_):.4f}")
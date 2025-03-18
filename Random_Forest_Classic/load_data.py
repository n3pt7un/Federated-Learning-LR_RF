from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
import pickle
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Random_Forest")

# Create a directory for persistent cache
os.makedirs("data_cache", exist_ok=True)

# Load dataset
dataset = load_dataset("n3p7un/KitsuneSystemAttackData_osScanDataset")

def preprocess_data(balance_method='undersample', sampling_ratio=1.0):
    """Preprocess the dataset with optional balancing of classes.
    
    Args:
        balance_method: Method to balance classes ('undersample', 'oversample', or None)
        sampling_ratio: For undersampling - ratio of negative:positive samples (e.g., 1.0 = equal)
                       For oversampling - multiplier for minority class (e.g., 3.0 = triple)
    """
    # Check disk cache first
    cache_file = f"data_cache/processed_data_{balance_method}_{sampling_ratio}.pkl"
    scaler_file = "data_cache/rf_scaler.pkl"
    
    if os.path.exists(cache_file):
        try:
            logger.info(f"Loading cached data from disk")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
    
    logger.info("Processing dataset")
    
    # Get the train split
    train_data = dataset["train"].with_format("numpy")
    
    # Get feature column names (exclude label and Unnamed: 0)
    feature_columns = [col for col in train_data.column_names if col not in ["label", "Unnamed: 0"]]
    
    # Extract features into a 2D array - use float32 for memory efficiency
    X = np.array([train_data[col] for col in feature_columns], dtype=np.float32).T
    y = train_data["label"]
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Apply feature standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    logger.info("Fitted StandardScaler to features")
    
    # Save the scaler for future use
    try:
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("Saved scaler to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save scaler: {e}")
    
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
    
    # Store on disk for future runs
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Saved processed data to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save to disk cache: {e}")
    
    return result

def load_preprocessed_data(balance_method='undersample', sampling_ratio=1.0):
    """Load the preprocessed data for model training.
    
    Args:
        balance_method: Method used for balancing ('undersample', 'oversample', or None)
        sampling_ratio: The sampling ratio used during preprocessing
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return preprocess_data(balance_method, sampling_ratio)

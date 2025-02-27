from datasets import Dataset, DatasetDict
import pandas as pd

# Load your dataset (example: CSV)
df = pd.read_csv('../Federated-Random-Forest/OS%20Scan-dataset_labels.csv')

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Perform train-test split (80% train, 20% test)
split_dataset = dataset.train_test_split(test_size=0.2)

# Access train and test datasets
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# Optionally, save to disk
train_dataset.to_csv('train_OSscan.csv')
test_dataset.to_csv('test_OSscan.csv')
from datasets import Dataset, DatasetDict
import pandas as pd

df = pd.read_csv('../Federated-Random-Forest/OS%20Scan-dataset_labels.csv')
dataset = Dataset.from_pandas(df)
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']
train_dataset.to_csv('train_OSscan.csv')
test_dataset.to_csv('test_OSscan.csv')
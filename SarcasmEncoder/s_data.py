import torch
import pandas as pd
from torch.utils.data import Dataset

class SarcasmDataset(Dataset):
    def __init__(self, data, data_encodings):
        self.data = data
        self.data_encodings = data_encodings
        self.labels = data['is_sarcastic'].values

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row['is_sarcastic'])

        return {
            'input_ids': self.data_encodings['input_ids'][idx],
            'attention_mask': self.data_encodings['attention_mask'][idx],
            'special_tokens_mask': self.data_encodings['special_tokens_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
      
    def __len__(self):
        return len(self.labels)

class ProcessSarcasm():
    def __init__(self, tokenizer, train_index_path, validation_index_path, test_index_path):
        self.tokenizer = tokenizer
        self.train_index_path = train_index_path
        self.validation_index_path = validation_index_path
        self.test_index_path = test_index_path

    def run(self):
        device = torch.device('cuda')
        cols = ['tweet', 'is_sarcastic']

        # Reads the data from csv into a DataFrame
        self.train_data = pd.read_csv(self.train_index_path) 
        train = pd.DataFrame(self.train_data, columns=cols).dropna()

        self.validation_data = pd.read_csv(self.validation_index_path)
        validation = pd.DataFrame(self.validation_data, columns=cols).dropna()

        self.test_data = pd.read_csv(self.test_index_path)
        test = pd.DataFrame(self.test_data, columns=cols).dropna()
        
        # Tokenizes train data
        self.train_encodings = self.tokenizer.batch_encode_plus(
            train['tweet'].values,
            max_length = 128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_special_tokens_mask=True
        )
        self.val_encodings = self.tokenizer.batch_encode_plus(
            validation['tweet'].values,
            max_length = 128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_special_tokens_mask=True
        )
        self.test_encodings = self.tokenizer.batch_encode_plus(
            test['tweet'].values,
            max_length = 128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_special_tokens_mask=True
        )
        
        train_dataset = SarcasmDataset(train, self.train_encodings)
        val_dataset = SarcasmDataset(validation, self.val_encodings)
        test_dataset = SarcasmDataset(test, self.test_encodings)
        
        return (train_dataset, val_dataset, test_dataset)
    
    def get_train_shape(self):
        return (self.train_data.shape[0], 128)
      
    def get_train_encodings(self):
        return self.train_encodings
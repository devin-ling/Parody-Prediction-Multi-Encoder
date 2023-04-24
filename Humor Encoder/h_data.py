import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class HumorDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        label = int(row['humor'])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def process_h_data(train_path, dev_path, test_path):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", do_lower_case=True)

    train_data = pd.read_csv(train_path)
    dev_data = pd.read_csv(dev_path)
    test_data = pd.read_csv(test_path)

    train_dataset = HumorDataset(train_data, tokenizer)
    dev_dataset = HumorDataset(dev_data, tokenizer)
    test_dataset = HumorDataset(test_data, tokenizer)

    return train_dataset, dev_dataset, test_dataset
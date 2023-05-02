import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


class ParodyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
class ProcessParody():
    def __init__(self, raw_data_path, train_index_path, validation_index_path, test_index_path):
        self.raw_data_path = raw_data_path
        self.train_index_path = train_index_path
        self.validation_index_path = validation_index_path
        self.test_index_path = test_index_path

    def run(self):
        raw_data = pd.read_csv(self.raw_data_path, lineterminator='\n')
        train_data = pd.read_csv(self.train_index_path, lineterminator='\n')
        validation_data = pd.read_csv(self.validation_index_path, lineterminator='\n')
        test_data = pd.read_csv(self.test_index_path, lineterminator='\n')

        data = pd.DataFrame(raw_data, columns=['tweet_id', 'tweet_pp', 'label\r'])
        
        train = pd.DataFrame(train_data, columns=['tweet_pp', 'label\r'])
        validation = pd.DataFrame(validation_data, columns=['tweet_pp', 'label\r'])
        test = pd.DataFrame(test_data, columns=['tweet_pp', 'label\r'])

        device = torch.device('cuda')

        train_text_values = train_data['tweet_pp'].values
        validation_text_values = validation_data['tweet_pp'].values
        test_text_values = test_data['tweet_pp'].values
        
        train_labels = train_data['label\r'].values
        validation_labels = validation_data['label\r'].values
        test_labels = test_data['label\r'].values

        tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', do_lower_case=True)
        train_encodings = tokenizer(train_text_values.tolist(), max_length=85, truncation=True, padding=True)  # tried with length 52
        val_encodings = tokenizer(validation_text_values.tolist(), max_length=85, truncation=True, padding=True)
        test_encodings = tokenizer(test_text_values.tolist(), max_length=85, truncation=True, padding=True)

        train_dataset = ParodyDataset(train_encodings, train_labels.tolist())
        val_dataset = ParodyDataset(val_encodings, validation_labels.tolist())
        test_dataset = ParodyDataset(test_encodings, test_labels.tolist())
        
        return (train_dataset, val_dataset, test_dataset)
    
p_raw_data_path = '/content/drive/My Drive/data/parody_data_acl20/data_all.csv'
p_train_path = '/content/drive/My Drive/data/parody_data_acl20/split_train.csv'
p_val_path = '/content/drive/My Drive/data/parody_data_acl20/split_val.csv'
p_test_path = '/content/drive/My Drive/data/parody_data_acl20/split_test.csv'

process = ProcessParody(p_raw_data_path, p_train_path, p_val_path, p_test_path)
res_data = process.run()
train_dataset = res_data[0]  # processed train dataset
val_dataset = res_data[1]  # processed validation dataset
test_dataset = res_data[2]  # processed test dataset
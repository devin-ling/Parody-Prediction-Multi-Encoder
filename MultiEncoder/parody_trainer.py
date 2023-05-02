import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertForSequenceClassification, AdamW, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, AdamW, RobertaForSequenceClassification
from transformers import BertTokenizer, RobertaTokenizer
from transformers import AutoModel, AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from model import Combo_Model_Attention
from model import Combo_Model_Other
from process_data import Process

class parody_trainer():

    def run(self):
        os.environ['WANDB_MODE'] = 'offline'
        os.environ["WANDB_DISABLED"] = "true"

        raw_data_path = '/content/drive/My Drive/data/parody_data_acl20/data_split1.csv'
        dev_path = '/content/drive/My Drive/data/parody_data_acl20/dev.txt'
        test_path = '/content/drive/My Drive/data/parody_data_acl20/test.txt'
        train_path = '/content/drive/My Drive/data/parody_data_acl20/train.txt'
        
        process = Process(raw_data_path, train_path, dev_path, test_path)
        res_data = process.run()
        train_dataset = res_data[0]  # processed train dataset
        val_dataset = res_data[1]  # processed validation dataset
        test_dataset = res_data[2]  # processed test dataset

        # To keep the reproducity, we set a random seed as below
        torch.manual_seed(100)
        random.seed(100)
        np.random.seed(100)
        torch.cuda.manual_seed_all(100)


        # TODO: Choose A TOKENIZER
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", do_lower_case=True)

        # TODO: CHOOSE A MODEL
        # 1. BERTweet Model
        # model = AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base', num_labels=2, output_attentions=False, output_hidden_states=False)

        # 2. Multi-Semantic-Encoder Model (Attention)
        model = Combo_Model_Attention()

        # 3. Multi-Semantic-Encoder Model (Other Approach, need manual config)
        # model = Combo_Model_Other()

        # 4. RoBERTA Model
        # model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, output_attentions=False, output_hidden_states=False)

        # 5. BERT Model
        # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)

        model.to('cuda')
        training_args = TrainingArguments("test_trainer", per_device_train_batch_size=16, per_device_eval_batch_size=64,
                                          evaluation_strategy="epoch", num_train_epochs=2.0)

        # About 22G GPU memories are needed                                 
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=tokenizer
        )

        # Train
        trainer.train()

        model.eval()
        torch.save(model,"/content/drive/My Drive/data/attention_model.pt")

        return

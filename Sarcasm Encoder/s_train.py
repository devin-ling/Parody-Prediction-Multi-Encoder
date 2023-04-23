import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

class SarcasmTrainer:
    
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", do_lower_case=True)

    def train_sarcasm_encoder(self, pretrain_epochs=5, finetune_epochs=5, batch_size=16, learning_rate=2e-5):
        # Pretraining
        model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base")
        model.to('cuda')

        pretrain_args = TrainingArguments(
            "sarcasm_pretrain",
            per_device_train_batch_size=batch_size,
            num_train_epochs=pretrain_epochs,
            learning_rate=learning_rate,
        )
        pretrain_trainer = Trainer(
            model=model,
            args=pretrain_args,
            train_dataset=self.train_data,
            tokenizer=self.tokenizer,
        )
        pretrain_trainer.train()

        # Save the pretrained model
        model.save_pretrained("pretrained_sarcasm_bertweet")

        # Finetuning
        model = AutoModelForSequenceClassification.from_pretrained("pretrained_sarcasm_bertweet", num_labels=2)
        model.to('cuda')

        finetune_args = TrainingArguments(
            "sarcasm_finetune",
            per_device_train_batch_size=batch_size,
            num_train_epochs=finetune_epochs,
            learning_rate=learning_rate,
        )
        finetune_trainer = Trainer(
            model=model,
            args=finetune_args,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            tokenizer=self.tokenizer,
        )
        finetune_trainer.train()

        # Save the finetuned model
        save_path = "/content/drive/MyDrive/s_pre"
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        return
    
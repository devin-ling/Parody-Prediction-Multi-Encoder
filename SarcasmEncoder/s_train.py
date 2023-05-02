import torch
from tqdm import tqdm
from s_data import ProcessSarcasm
from transformers import TrainingArguments, Trainer
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer

class SarcasmTrainer():
    def __init__(self, model, tokenizer, positive_datasets, classification_datasets, output_dir):
        self.model = model

        self.positive_train = positive_datasets[0]
        self.positive_val = positive_datasets[1]
        self.positive_test = positive_datasets[2]

        self.classification_train = classification_datasets[0]
        self.classification_val = classification_datasets[1]
        self.classification_test = classification_datasets[2]

        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def adaptive_pretrain(self):
        process_sarcasm = ProcessSarcasm(self.tokenizer, self.positive_train, self.positive_val, self.positive_test)
        train_dataset, val_dataset, test_dataset = process_sarcasm.run()

        # Selects masked tokens at a probability of 0.15, and updates token_ids
        rand = torch.rand(process_sarcasm.get_train_shape())
        train_encodings = process_sarcasm.get_train_encodings()
        mask_arr = (rand < 0.15) * (train_encodings['special_tokens_mask'] != 1)
        selected_masks = [torch.flatten(mask_arr[i].nonzero()).tolist() for i in range(mask_arr.shape[0])]
        for i in range(mask_arr.shape[0]):
            train_encodings['input_ids'][i, selected_masks[i]] = sarcasm_tokenizer.convert_tokens_to_ids('<mask>')
        
        # Sets the device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

        dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.model.train()
        optim = AdamW(self.model.parameters(), 2e-5)

        epochs = 2
        for epoch in range(epochs):
            loop = tqdm(dataloader, leave=True)
            for batch in loop:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optim.step()

                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

    def classification_train(self):
        process_sarcasm = ProcessSarcasm(self.tokenizer, self.classification_train, self.classification_val, self.classification_test)
        train_dataset, val_dataset, test_dataset = process_sarcasm.run()

        # Sets the device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
    
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=128,
            evaluation_strategy="epoch",
            num_train_epochs=2,
            learning_rate=3e-5,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        return trainer

    def evaluate(self, trainer):
        results = trainer.evaluate()
        return results

    def save(self, trainer):
        trainer.save_model(self.output_dir)

sarcasm_pretrain_model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2, output_attentions=False, output_hidden_states=False)
sarcasm_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
positive_datasets = ('/content/drive/My Drive/data/iSarcasmEval/positive_train_split.csv', '/content/drive/My Drive/data/iSarcasmEval/positive_val_split.csv', '/content/drive/My Drive/data/iSarcasmEval/positive_test_split.csv')
classification_datasets = ('/content/drive/My Drive/data/iSarcasmEval/classification_train_split.csv', '/content/drive/My Drive/data/iSarcasmEval/classification_val_split.csv', '/content/drive/My Drive/data/iSarcasmEval/classification_test_split.csv')
sarcasm_trainer = SarcasmTrainer(sarcasm_pretrain_model, sarcasm_tokenizer, positive_datasets, classification_datasets, output_dir='/content/drive/My Drive/data/s_test_pre')
sarcasm_trainer.adaptive_pretrain()
classification_trainer = sarcasm_trainer.classification_train()
results = sarcasm_trainer.evaluate(classification_trainer)
print(results)
sarcasm_trainer.save(classification_trainer)
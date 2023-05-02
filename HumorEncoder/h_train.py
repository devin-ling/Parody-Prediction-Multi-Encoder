from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from h_data import process_h_data

class HumorTrainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, tokenizer, output_dir):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def train(self):
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
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        return trainer

    def evaluate(self, trainer):
        results = trainer.evaluate()
        return results

    def save(self, trainer):
        trainer.save_model(self.output_dir)


# opportunity to actually test humor encoder 
train_path = "/content/drive/My Drive/humor-dataset/train.csv"
dev_path = "/content/drive/My Drive/humor-dataset/dev.csv"
test_path = "/content/drive/My Drive/humor-dataset/test.csv"

train_dataset, val_dataset, test_dataset = process_h_data(train_path, dev_path, test_path)
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2, output_attentions=False, output_hidden_states=False).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", do_lower_case=True)

humor_trainer = HumorTrainer(model, train_dataset, val_dataset, test_dataset, tokenizer, output_dir='/content/drive/My Drive/h_pre')
trainer = humor_trainer.train()
# results = humor_trainer.evaluate(trainer)
# print(results)
humor_trainer.save(trainer)
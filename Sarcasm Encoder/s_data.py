import csv

class ISarcasmDataset:
    def __init__(self, train_path, test_path):
        self.train_data = self._process_file(train_path)
        self.test_data = self._process_file(test_path)

    def _process_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = []
            for row in reader:
                tweet_id, tweet_text, sarcasm_label, sarcasm_type = row
                example = {
                    "tweet_id": tweet_id,
                    "tweet_text": tweet_text,
                    "sarcasm_label": int(sarcasm_label),
                }
                if sarcasm_type:
                    example["sarcasm_type"] = sarcasm_type
                data.append(example)
            return data

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data
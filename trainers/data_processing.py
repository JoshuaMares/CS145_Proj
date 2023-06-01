from sklearn.model_selection import train_test_split

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import DataLoader, SequentialSampler


from transformers import (
    AutoTokenizer,
)




class dataProcessor:
    """Processor for Dummy Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self):
        """Initialization."""
        self.data_dir = 'datasets/final.csv'
        self.df = pd.read_csv(self.data_dir)

        self.X = self.df['info_sentence']
        self.y = self.df['meta']



        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

   

    def get_train_examples(self):
        return self.X_train, self.y_train

    def get_dev_examples(self):
        return self.X_dev, self.y_dev

    def get_test_examples(self):
        return self.X_test, self.y_test


class cardDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_length):
        self.texts, self.labels = examples

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        batch_encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        input_ids = torch.Tensor(batch_encoding["input_ids"]).long()
        attention_mask = torch.Tensor(batch_encoding["attention_mask"]).long()
        if "token_type_ids" not in batch_encoding:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = torch.Tensor(batch_encoding["token_type_ids"]).long()

        label = torch.Tensor([label]).long()[0]

        return input_ids, attention_mask, token_type_ids, label
    




if __name__ == "__main__":

    processor = dataProcessor()

    X_train, y_train = processor.get_train_examples()

    print(X_train)
    print(y_train)



    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")



    examples = processor.get_dev_examples()
    dataset = cardDataset(examples, tokenizer,
                        max_seq_length=32)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    for step, batch in enumerate(dataloader):
        for each in batch:
            assert each.size()[0] == 1, "Batch not loading correctly! Some error!"
        break
    print ("Dummy Dataset loading correctly.")


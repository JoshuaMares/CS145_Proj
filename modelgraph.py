import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
import torch
import matplotlib.pyplot as plt
import numpy as np

# load data
df = pd.read_csv('card_info_clean.csv')  # replace with your csv file
df_sampled = df.sample(n=50, random_state=0)  # sample 1000 data points
df_sampled.info()
texts = df_sampled['info_sentence'].tolist()
labels = df_sampled['meta'].tolist()

# Split into training and validation before converting to tensors
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# load tokenizer and model
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

# tokenize data for training set
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
train_encodings['labels'] = torch.tensor(train_labels)

# tokenize data for validation set
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
val_encodings['labels'] = torch.tensor(val_labels)

# print keys of the encodings dictionary
print(train_encodings.keys())

# print the 'input_ids' tensor
print(train_encodings['input_ids'])

# print the 'attention_mask' tensor
print(train_encodings['attention_mask'])

# if the labels were included in the encodings, print the 'labels' tensor
if 'labels' in train_encodings:
    print(train_encodings['labels'])

# Convert encodings to a Dataset format
class CardInfoDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = CardInfoDataset(train_encodings)
val_dataset = CardInfoDataset(val_encodings)

# define the training args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# load model with the right number of labels
model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=len(set(labels)))

# create the trainer
iteration = 0
train_accs = []
def compute_metrics(pred):
    print("function called")
    global iteration
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).mean().item()
    train_accs.append(accuracy)
    print(f"Iteration: {iteration}, Accuracy: {accuracy}")
    iteration += 1
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# train the model
trainer.train()

print("train_accs", train_accs)

# plot the accuracy vs. training iteration graph
iterations = list(range(1, len(train_accs) + 1))
plt.plot(iterations, train_accs, label="Training Accuracy")
plt.xlabel("Training Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Training Iteration")
plt.legend()
plt.show()

# Save the plot to a file
output_file = 'accuracy_plot.png'  # Specify the desired output file path and name
plt.savefig(output_file)

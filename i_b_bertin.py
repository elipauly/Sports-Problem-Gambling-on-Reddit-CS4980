#from https://www.geeksforgeeks.org/machine-learning/toxic-comment-classification-using-bert/
import numpy as np
import pandas as pd

#data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

#to avoid warnings
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("/Users/pauly/VSCode/26spring/Sports-Problem-Gambling-on-Reddit-CS4980/bert_test.csv")
print(data.head())

# Visualizing the class distribution of the 'label' column
column_labels = data.columns[1:2]
label_counts = data[column_labels].sum()


# Create subsets based on pg and clean comments
train_problem = data[data[column_labels].sum(axis=1) > 0]
train_clean = data[data[column_labels].sum(axis=1) == 0]

#splot into training, testing, and validation sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'], data[column_labels].values, test_size=0.2, random_state=42)

#validation set
test_texts, val_texts, test_labels, val_labels = train_test_split(
    test_texts, test_labels, test_size=0.5, random_state=42)

#tokenization
def tokenize_and_encode(tokenizer, text, labels, max_length=512):
    encoded = tokenizer(
        list(text),
        padding=True,              # ensures all sequences same length
        truncation=True,           # cuts off long sequences
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels
#initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

input_ids, attention_masks, labels = tokenize_and_encode(
    tokenizer, train_texts, train_labels)
test_input_ids, test_attention_masks, test_labels = tokenize_and_encode(
    tokenizer, test_texts, test_labels)
val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(
    tokenizer, val_texts, val_labels)

print("Training comments:", train_texts)
print("Input Ids shape:", input_ids.shape)
print("Attention Masks shape:", attention_masks.shape)
print("Labels shape:", labels.shape)

#creating pytorch dataloaders
batch_size = 16
train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   

test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#checking the train_loader data
print("Batch size: ", train_loader.batch_size)
Batch = next(iter(train_loader))
print("Batch input ids shape: ", Batch[0].shape)
print("Batch attention masks shape: ", Batch[1].shape)
print("Batch labels shape: ", Batch[2].shape)

#adamw optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

#model training
def train_model(model, train_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0

        #disable gradient computation during validation
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [
                    t.to(device) for t in batch]
                outputs = model(
                    input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
            print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}')

train_model(model, train_loader, optimizer, device, num_epochs=13)

#model evaluation
def evaluate_model(model, test_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            predicted_labels.append(preds.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    true_labels = np.concatenate(true_labels, axis=0)
    predicted_labels = np.concatenate(predicted_labels, axis=0)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

evaluate_model(model, test_loader, device)


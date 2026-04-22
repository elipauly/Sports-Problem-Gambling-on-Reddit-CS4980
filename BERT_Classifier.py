# This is our official classifier model. It uses the BERT architecture to classify comments as either "problem gambling" or "clean". The model is trained on a dataset of Reddit comments that have been labeled as either "problem gambling" or "clean". The model is then used to predict the labels of new comments.

# Before running this, follow BERT_Visualization.py to make sure training data is (reasonably) balanced and happy.
# Also install the necessary libraries (transformers, torch, sklearn, etc.) if you haven't already (listed in requirements.txt)

# Structure:
# 1. Imports / Data loading
# 2. Data preprocessing & tokenization
# 3. Model Training
# 4. Model Evaluation
# 5. Prediction

##############
# 1. Imports / Data loading
import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

import warnings
warnings.filterwarnings('ignore')

##############
# LOAD THE CORRECT CSV FILE 
data = pd.read_csv("BERT_Training_Labelled_Sampled.csv")
predict_df = pd.read_csv("bert_sportsbetting.csv")
predict_df2 = pd.read_csv("bert_sportsbook.csv")
print("data.head(): ", data.head())

##############
# 2. Data preprocessing & tokenization

label_column = data.columns[1]
labels = data[label_column].values.astype(int)
texts = data['text'].values

def set_seed(seed=67):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(67)

#splot into training, testing, and validation sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels,
    test_size=0.3,
    stratify=labels,
    random_state=67
)

#validation set
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=67
)

#tokenization
def tokenize_and_encode(tokenizer, text, labels, max_length=512):
    encoded = tokenizer(
        list(text),
        padding="max_length",            
        truncation=True,           
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    labels = torch.tensor(labels).long().view(-1)

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


##############
# 3. Model Training / Building

#training
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss / max(1, len(val_loader)):.4f}')
        torch.cuda.empty_cache()

train_model(model, train_loader, val_loader, optimizer, device, num_epochs=8)


##############
# 4. Model Evaluation

def evaluate_model(model, test_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            predicted_labels.append(preds.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    true_labels = np.concatenate(true_labels, axis=0)
    predicted_labels = np.concatenate(predicted_labels, axis=0)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)

    all_probs = np.array(all_probs)
    pos_probs = all_probs[:, 1]

    roc_auc = roc_auc_score(all_labels, pos_probs)
    pr_auc = average_precision_score(all_labels, pos_probs)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

evaluate_model(model, test_loader, device)

##############
# SAVING THE MODEL, TOKENIZER, AND LOADING IT BACK IN FOR PREDICTION
output_dir = "Saved_BERT_Model_8Epochs"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

model_name = "Saved_BERT_Model_8Epochs"
Bert_Tokenizer = BertTokenizer.from_pretrained(model_name)
Bert_Model = BertForSequenceClassification.from_pretrained(model_name).to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
##############
# 5. Prediction

#chatgpt gave me this, predict in batches for large dataset
def predict_batch(texts, model, tokenizer, device, batch_size=64):
    model.eval()
    
    all_probs = []
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        encodings = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    label_map = {0: "Clean", 1: "Problem Gambling"}

    results = []
    for prob, pred in zip(all_probs, all_preds):
        results.append({
            "predicted_label": label_map[pred],
            "probabilities": dict(zip(label_map.values(), prob))
        })

    return results

predict_texts = predict_df['text'].astype(str).tolist()

results = predict_batch(predict_texts, model, tokenizer, device)

predict_df['predicted_label'] = [r['predicted_label'] for r in results]
predict_df['prob_clean'] = [r['probabilities']['Clean'] for r in results]
predict_df['prob_pg'] = [r['probabilities']['Problem Gambling'] for r in results]

predict_df.to_csv("bert_sportsbetting_predictions_8epochs.csv", index=False)


####
predict_texts = predict_df2['text'].astype(str).tolist()

results = predict_batch(predict_texts, model, tokenizer, device)

predict_df2['predicted_label'] = [r['predicted_label'] for r in results]
predict_df2['prob_clean'] = [r['probabilities']['Clean'] for r in results]
predict_df2['prob_pg'] = [r['probabilities']['Problem Gambling'] for r in results]

predict_df2.to_csv("bert_sportsbook_predictions_8epochs.csv", index=False)
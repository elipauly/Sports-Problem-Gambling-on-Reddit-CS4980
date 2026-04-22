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

predict_df = pd.read_csv("bert_sportsbook.csv")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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

predict_df.to_csv("bert_sportsbook_predictions.csv", index=False)
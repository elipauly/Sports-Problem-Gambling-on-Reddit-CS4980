import pandas as pd

nathan = pd.read_csv('nathan.csv')
tristan = pd.read_csv('tristan.csv')
pauly = pd.read_csv('pauly.csv')
nathan2 = pd.read_csv('nathan_pt2.csv')
tristan2 = pd.read_csv('tristan_pt2.csv')
pauly2 = pd.read_csv('pauly_pt2.csv')

dfs = [nathan, tristan, pauly, nathan2, tristan2, pauly2]

combined_samples = pd.concat(dfs, ignore_index=True)
combined_samples.to_csv("BERT_Training_Labelled_Full.csv", index=False)

print(len(combined_samples))
print(combined_samples.head())
print(combined_samples['label'].value_counts())
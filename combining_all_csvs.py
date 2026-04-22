import pandas as pd

nathan = pd.read_csv('nathan.csv')
tristan = pd.read_csv('tristan.csv')
pauly = pd.read_csv('pauly.csv')
nathan2 = pd.read_csv('nathan_pt2.csv')
tristan2 = pd.read_csv('tristan_pt2.csv')
pauly2 = pd.read_csv('pauly_pt2.csv')

dfs = [nathan, tristan, pauly, nathan2, tristan2, pauly2]

combined_samples = pd.concat(dfs, ignore_index=True)
#combined_samples['text'] = combined_samples['text'].str.replace('"', '', regex=False)

#print(combined_samples['label'].isna().sum())
#print(tristan[tristan.isnull().any(axis=1)])


combined_samples['label'] = combined_samples['label'].astype(int)
combined_samples.to_csv("BERT_Training_Labelled_Full.csv", index=False)


#print(len(combined_samples))
#print(combined_samples.head())
#print(combined_samples['label'].value_counts())

#Undersampling the majority class to balance the dataset
#print(combined_samples['label'].value_counts().min())

#count of minority = 222, so we will sample 888 from the majority to have a 20-80 split balance
undersampled_majority = combined_samples[combined_samples['label'] == 0].sample(n=888, random_state=42)

# Combine minority class with downsampled majority class
df_sampled = pd.concat([undersampled_majority, combined_samples[combined_samples['label'] == 1]])
df_sampled.to_csv("BERT_Training_Labelled_Sampled.csv", index=False)
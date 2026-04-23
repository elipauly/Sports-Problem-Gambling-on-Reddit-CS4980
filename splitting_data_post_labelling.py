import pandas as pd

df_betting = pd.read_csv('bert_sportsbetting_predictions_8epochs.csv')
df_book = pd.read_csv('bert_sportsbook_predictions_8epochs.csv')
filtered_betting = df_betting[df_betting['predicted_label'] == "Problem Gambling"]
filtered_book = df_book[df_book['predicted_label'] == "Problem Gambling"]

dfs = [filtered_betting, filtered_book]

combined_samples = pd.concat(dfs, ignore_index=True)
combined_samples.to_csv("BERT_Predicted_Positives.csv", index=False)

print(combined_samples.shape[0])

combined_samples['path_behaviors'] = ''
combined_samples['life_problems'] = ''
combined_samples['cognitive_distortions'] = ''
combined_samples['helpful_comments'] = ''
combined_samples['evil_comments'] = ''
combined_samples['notes'] = ''

tristan = combined_samples.iloc[:67, :]  # First 287 rows
pauly = combined_samples.iloc[67:137, :]
nathan = combined_samples.iloc[137:, :]

tristan.to_csv("tristan_positives.csv", index=False)
pauly.to_csv("pauly_positives.csv", index=False)  
nathan.to_csv("nathan_positives.csv", index=False)
import pandas as pd

df = pd.read_csv('bert_sportsbetting_predictions_8epochs.csv')
count = (df['predicted_label'] == "Problem Gambling").sum()

# Method 2: Get a full breakdown of all values in a column
print(95+128)
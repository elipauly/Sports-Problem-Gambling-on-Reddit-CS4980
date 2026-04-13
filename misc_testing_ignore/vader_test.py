#this attatches a sentiment score to each post in the json file and saves it to a new json file. It uses the VADER sentiment analysis tool to calculate the sentiment scores.

import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# 1. Load JSON Data
with open('sportsbetting_14days.json', 'r') as f:
    data = json.load(f)

# 2. Process Data
# Assuming 'data' is a list of objects, each with a 'text' field
for entry in data:
    text = entry['body']
    vs = analyzer.polarity_scores(text)
    
    # Add sentiment scores to the entry
    entry['sentiment'] = vs
    
    # Optional: Categorize based on compound score
    if vs['compound'] >= 0.05:
        entry['sentiment_label'] = 'positive'
    elif vs['compound'] <= -0.05:
        entry['sentiment_label'] = 'negative'
    else:
        entry['sentiment_label'] = 'neutral'

# 3. Save Results to New JSON File
with open('sentiment_results.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Sentiment analysis complete. Results saved to 'sentiment_results.json'.")
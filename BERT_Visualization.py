# largely from https://www.geeksforgeeks.org/machine-learning/toxic-comment-classification-using-bert/
# run this before BERT_Classifier.py to visualize the test data distribution.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LABELLED TRAINING DATA.csv GOES HERE !
data = pd.read_csv("BERT_Training_Labelled_Sampled.csv")
print(data.head())

# Visualizing the class distribution of the 'label' column
column_labels = data.columns[1:2]
label_counts = data[column_labels].sum()

# Create subsets based on problem gambling (pg) and clean comments
train_problem = data[data[column_labels].sum(axis=1) > 0]
train_clean = data[data[column_labels].sum(axis=1) == 0]

# Number of pg and clean comments
num_problem = len(train_problem)
num_clean = len(train_clean)

# Create a DataFrame for visualization
plot_data = pd.DataFrame(
    {'Category': ['Problem', 'Clean'], 'Count': [num_problem, num_clean]})

# Create a black background for the plot
plt.figure(figsize=(7, 5))

# Horizontal bar plot
ax = sns.barplot(x='Count', y='Category', data=plot_data, palette='viridis')

# Add labels and title to the plot
plt.xlabel('Number of Comments')
plt.ylabel('Category')
plt.title('Distribution of Problem Gambling and Clean Comments')

# Set ticks' color to white
ax.tick_params()

plt.show()
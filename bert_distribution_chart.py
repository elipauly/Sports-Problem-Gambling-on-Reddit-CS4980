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
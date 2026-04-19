import pandas as pd

#method: random bootstrap sampling 200 from gamblingaddiction to learn from problem gambling language. from a skim, this one is more applicable to us than problemgamblign 
#400 are random bootstrap sampled from sportsbook and sportsbetting to learn from those subreddits.


book = pd.read_csv('bert_sportsbook.csv')
betting = pd.read_csv('bert_sportsbetting.csv')
problem = pd.read_csv('./misc_testing_ignore/bert_gamblingaddiction.csv')

row_count = len(book)
row_count2 = len(betting)
row_count3 = len(problem)
print(f"Total rows: {row_count}")
print(f"Total rows: {row_count2}")
print(f"Total rows: {row_count3}")


df1 = book.sample(n=200, replace=True)
df2 = betting.sample(n=200, replace=True)
df3 = problem.sample(n=200, replace=True)

dfs = [df1, df2, df3]

# 1. Take a sample from each and combine them
combined_samples = pd.concat(dfs, ignore_index=True)
combined_samples.to_csv("BERT_Training_Unlabelled_Full.csv", index=False)


shuffled = combined_samples.sample(frac=1)

nathan = shuffled.iloc[:287, :]  # First 287 rows
tristan = shuffled.iloc[287:467, :]  # Next 225 rows, first 512 columns
pauly = shuffled.iloc[467:, :]  # 


nathan.to_csv("nathan.csv", index=False)
tristan.to_csv("tristan.csv", index=False)
pauly.to_csv("pauly.csv", index=False)

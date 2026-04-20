import pandas as pd

book = pd.read_csv('bert_sportsbook.csv')
betting = pd.read_csv('bert_sportsbetting.csv')
problem = pd.read_csv('./misc_testing_ignore/bert_gamblingaddiction.csv')


df1_1 = book.sample(n=200, replace=True)
df2_1 = betting.sample(n=200, replace=True)
df3_1 = problem.sample(n=200, replace=True)

dfs_1 = [df1_1, df2_1, df3_1]

combined_samples_1 = pd.concat(dfs_1, ignore_index=True)
combined_samples_1.to_csv("BERT_Training_Unlabelled_Full_pt2.csv", index=False)


shuffled = combined_samples_1.sample(frac=1)

nathan_pt2 = shuffled.iloc[:200, :]  # First 287 rows
tristan_pt2 = shuffled.iloc[200:400, :]  # Next 225 rows, first 512 columns
pauly_pt2 = shuffled.iloc[400:, :]  # 

nathan_pt2.to_csv("nathan_pt2.csv", index=False)
tristan_pt2.to_csv("tristan_pt2.csv", index=False)
pauly_pt2.to_csv("pauly_pt2.csv", index=False)
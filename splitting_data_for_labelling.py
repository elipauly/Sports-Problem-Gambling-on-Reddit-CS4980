import pandas as pd

#method: sampling 200 from problemgambling to learn from problem gambling language. 
#100 of the instances are keyword based, i searched for comments containing "bet", "gamble", "sportsbook", "sportsbetting" and then sampled 100 from those. the other 100 are randomly sampled from the problem gambling subreddit without keyword filtering, to get a more representative sample of the language used in that subreddit.
#300 are from sportsbook and sportsbetting to learn from those subreddits.


df = pd.read_csv('bert_sportsbook.csv')
df2 = pd.read_csv('bert_sportsbetting.csv')

row_count = len(df)
row_count2 = len(df2)
print(f"Total rows: {row_count}")
print(f"Total rows: {row_count2}")


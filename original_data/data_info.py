import pandas as pd

# Load from a file or string
dfga = pd.read_json('original_data/gamblingaddiction_15days.json')
dfbook = pd.read_json('original_data/sportsbook_15days.json')
dfbet = pd.read_json('original_data/sportsbetting_14days.json')

print("dfga: ", dfbet.shape[0])
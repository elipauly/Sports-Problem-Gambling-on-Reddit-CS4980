import pandas as pd

# Load the file and select only one column (e.g., 'column_name')
df = pd.read_json('data.json')[['column_name']]
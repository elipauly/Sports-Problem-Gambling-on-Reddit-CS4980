#this cleans a json file to only include items where body has a length of 512 characters or less.

import json

with open('sportsbetting_14days.json', 'r') as f:
    data = json.load(f)
    threshold = 512
    count = 0

filtered_data = [
    item for item in data 
    if len(item.get("body", "")) <= threshold
]

#write the filtered data to a new json file
#with open('token_limit_sportsbetting.json', 'w') as f:
#    json.dump(filtered_data, f, indent=4)

print(f"Filtered {len(data) - len(filtered_data)} items.")
print(f"Remaining items: {len(filtered_data)}.")
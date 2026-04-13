import json
import pandas as pd
import re

INPUT_FILE = "token_limit_problemgambling.json"
OUTPUT_FILE = "bert_problemgambling.csv"

#TEXT CLEANING#
def clean_text(text):
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove "Images:" artifacts
    text = re.sub(r"Images:\s*", "", text)   
    # Remove HTML entities
    text = re.sub(r"&amp;", "&", text)  
    # Remove extra whitespace/newlines/tabs
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


#TEXT EXTRACTION#
def extract_text(entry):
    if entry["dataType"] == "post":
        title = entry.get("title", "")
        body = entry.get("body", "")
        text = f"{title} {body}"
    else:  # comment
        text = entry.get("body", "")
    
    return clean_text(text)


#LABEL HANDLING#
def extract_label(entry):
    #Problem gambling indicator (if available)
    if "pg_indicator" in entry:
        return int(entry["pg_indicator"])


#MAIN PIPELINE#
def preprocess():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    processed = []

    for entry in data:
        text = extract_text(entry)

        # Skip empty or junk
        if not text or len(text) < 20: # arbitrary threshold for minimum length to filter spammy/low-quality content
            continue

        label = extract_label(entry)

        processed.append({
            "text": text,
            "label": label,
            "type": entry["dataType"],  # optional, useful for analysis
            "subreddit": "r/ProblemGamling"
        })

    df = pd.DataFrame(processed)

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    # Optional: balance inspection
    print("\nClass distribution:")
    print(df["label"].value_counts(normalize=True))

    print("\nCounts by type:")
    print(df["type"].value_counts())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess()
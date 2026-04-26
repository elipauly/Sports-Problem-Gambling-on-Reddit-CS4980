import json
from os import read
import re
import pandas as pd

# PREDICTION_FILE = "bert_sportsbetting_predictions_8epochs.csv"
# RAW_JSON_FILE = "original_data/sportsbetting_14days.json"
# OUT_FILE = "sportsbetting_nested_comments.json"

PREDICTION_FILE = "bert_sportsbook_predictions_8epochs.csv"
RAW_JSON_FILE = "original_data/sportsbook_15days.json"
OUT_FILE = "sportsbook_nested_comments.json"

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

def getTargetPostTexts():
  predictions = pd.read_csv(PREDICTION_FILE)
  pg_posts = predictions
  texts = pg_posts.loc[(pg_posts['prob_pg'] > 0.5), ["text"]]
  return texts["text"].values.tolist()

def getChildComments(postId, samples):
  def isChild(sample):
    return sample["dataType"] == "comment" and sample["parentId"] == postId
  
  def simplify(comment):
    return {k:comment.get(k) for k in ('id','username','body',"enabling", "url")}
  # recursively nest subComments or just use parentId to get only top-level comments
  def helper(comment):
    comment["children"] = getChildComments(comment["id"], samples)
    return comment

  return list(map(helper, map(simplify, filter(isChild, samples))))

def nestCommentsInFile(texts):
  with open(RAW_JSON_FILE, "r") as inFile:
    with open(OUT_FILE, "w") as outFile:
      samples = json.load(inFile)
      output = []
      for text in texts:
        # TODO: need to combine title and body the same way as was done in processing so that the texts match exactly
        mainPost = next((p for p in samples if p["dataType"] == "post" and extract_text(p) == text), None)
        if (mainPost):
          postId = mainPost["id"]
          comments = getChildComments(postId, samples)
          print(len(comments))
          output.append({"postId": postId, "url": mainPost["url"], "title": mainPost["title"], "body": mainPost["body"], "problemGambling": None, "comments": comments})
      json.dump(output, outFile, indent=4)


texts = getTargetPostTexts()
nestCommentsInFile(texts)
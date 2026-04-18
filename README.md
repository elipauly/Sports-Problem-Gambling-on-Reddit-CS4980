🐣🐣 

## "Pauly what the heck is going on here ?" ##
The 2 main files to build our models are **BERT_Visualizer.py** and **BERT_Classifier.py** There are instructions in each file.  
This is a binary classifier. If there exists any labelled data that is not a 1 or 0 my lovely model will break and I will be sad. :(.  

All data is pulled from **bert_sportsbetting.csv** and **bert_sportsbook.csv** from their respective subreddits. Take note of the processing changes from their original data, found in /original_data folder. Some data cuts for token length and spam reduction, the code which made these changes is in /preprocess_json_for_classifying.py.  
  
File organization is a bit of a mess. For that I apologize but you're smart you'll figure it out.

## "What should I do?" ##
Another great question, you each have lovely datasets to manually label, **tristan.csv** and **nathan.csv**, *strictly following our annotation guide*. BERT relies on quality labelling. Better to ask than make an icky model. I want a pretty model.  

## short guide as im not certain how familiar we all are with classifier models ##
Labelled data: the data which You, my friend, will spend a lot of time manually adding a 1 or 0  
Training set: built from labelled data, but undersampling the majority class. Having a *somewhat* balanced training set is very necessary for our data.  
0: Clean (negative class)  
1: Problem Gambling (positive class)
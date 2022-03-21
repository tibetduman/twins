When there is a window of opportunity, smart entrepreneurs all around the world see the same opportunity. For that reason, there is almost never one company who tackles the same problem. There are many others.

This python script solely focusses on this problem to identify similar companies given a description.

DEPENDENCIES:
In this python script we make use of:
-Multiprocessing
-Time
-Scipy
-Pandas
-Sentence_transformers

Therefore to use this script please make sure to install these python models; some, like time, modules are already built in vanilla Python, whereas others need some installation. Our recommendation is installing them by a package managment system like pip.

SETUP:
The setup is quite simple:
* Pull the python script called twins.py
* Save the desired dataset as dataset.xls (in Microsoft Excel format, Note: CSV is not desired as short and long descriptions tend to have commas in them which mess up the formatting, converting from CSV to Excel is simple with third party websites, an example is https://cloudconvert.com/csv-to-xls)
* Inside twins.py change input_sentence to be the desired sentence
* Inside twins.py change category to be the desired category to search companies in
* Execute the python script (in terminal it would look like 'python3 twins.py'

If you would like to play around with the parameters feel free to do so, the ones you should change without any trouble are:
- BEST_K (the desired amount of most relevant companies)
- SENTENCE_LIMIT (limit on how many sentences to use BERT on)
- LOWEST_SENTENCE_LENGTH (lower bound on the length of a description for a sentence)
- WORDS_TO_IGNORE (the words to ignore when calculating token based similarity score)
- DESCRIPTION_LIMIT (limit on number of words to check for similarities in heuristics)
- model (#NLP model to use)



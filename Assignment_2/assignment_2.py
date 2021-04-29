# -*- coding: utf-8 -*-
"""assignment_2.ipynb
# Install requirements
"""
"""# Imports"""

import spacy, nltk
nlp = spacy.load('en_core_web_sm')
import pandas as pd
import conll
from sklearn.metrics import classification_report

"""# 1) Evaluate spaCy NER on CoNLL 2003 data (provided)

import_dataset(path):
  * Input: the path of the dataset to load
  * Output: two lists:
    1. text_dataset: contains the lists of sentences of the dataset as text
    2. dataset: contais the pair (token, name entity) for each token, divided in sentences (one list for each sentence)
  * Implementation: it reads the dataset using conll function, for each sentence it extracts the tokens as text or the tuple (token, name entity)
"""

def import_dataset(path):
  data = conll.read_corpus_conll(path)
  text_dataset = []
  dataset = []
  for t in data:
    sentence = []
    txt = ""
    for t2 in t:
      sentence.append((t2[0].split()[0], t2[0].split()[3]))
      txt += str(t2[0].split()[0]) + " "
    dataset.append(sentence)
    text_dataset.append([txt])
  return text_dataset, dataset

"""convert_type(ent_type):
  * Input: named entity from spaCy
  * Output: the name entity converted in the dataset format
  * Implementation: assign a specific named entity from the dataset format to each named entity from spaCy
"""

def convert_type(ent_type):
  if(ent_type in ["ORG"]):
    return "ORG"
  if(ent_type in ["GPE", "LOC"]):
    return "LOC"
  if(ent_type in ["LANGUAGE", "WORK_OF_ART", "FAC", "ORDINAL", "TIME", "NORP", "EVENT", "LAW", "CARDINAL", "PRODUCT", "DATE", "QUANTITY", "MONEY", "PERCENT"]):
    return "MISC"
  if(ent_type in ["PERSON"]):
    return "PER"
  return ""

"""convert_spacy(token, parent=None):
  * Input: the token to convert, the parent of the token to use in the third exercise
  * Output: the tags converted in form ```iob-type``` as in the dataset
  * Implementation:
    * if parent is None it returns just the concatenation between the ```IOB``` tag and the named entity tag
    * if parent is set it returns the named entity from the parent if possible


"""

def convert_spacy(token, parent=None):
  if(parent == None): # exercise 1 usage
    if(token.ent_iob_ == "O"):
      return "O"
    else:
      return f"{token.ent_iob_}-{convert_type(token.ent_type_)}"
  else: # exercise 3 usage
    if(token.ent_iob_ == "O"):
      if(parent.ent_type_ != ""):
        return f"I-{convert_type(parent.ent_type_)}"
      else:
        return "O"
    else:
      if(parent.ent_type_ != ""):
        return f"{token.ent_iob_}-{convert_type(parent.ent_type_)}"
      else:
        return f"{token.ent_iob_}-{convert_type(token.ent_type_)}"

"""reconstruct_output(doc, comp=False):
  * Input:
    * Doc object from spaCy
    * comp (compound) flag to set on the third exercise
    * ancestors: flag to use to reach the first ancestor with dependency different from "compound"
  * Output: list of sentences, each sentence contains the token "reconstructed" as in the dataset
  * Implementation:
    * given a token it uses whitespace to check if the token is part of a word in the dataset, if yes it concatenates the tokens with the same tag, otherwise the single token is used.
    * if comp is set to True, the tokens with compound dependency will have the same tag as their parents, moreover if also ancestors is set, the parent passed to convert_spacy will be the first ancestor with a dependency different from "compound"
"""

def reconstruct_output(doc, comp=False, ancestors=False):
  output = []
  current_token = ""
  current_tag = ""
  first = True
  for token in doc:
    if(first):
        current_tag = convert_spacy(token)
        if((comp) and (token.dep_ == "compound")):
          parent = token.head
          while((parent.dep_ == "compound") and (ancestors)):
            parent = parent.head
          current_tag = convert_spacy(token, parent)
        first = False
    if(not token.whitespace_):
      current_token += token.text
    else:
      current_token += token.text
      output.append((current_token, current_tag))
      first = True
      current_token = ""
      current_tag = ""
  if(not first):
    output.append((current_token, current_tag))

  return output

"""process_dataset(dataset_text, expand):
  * Input: the dataset as lists of sentences, expand is a flag used in the third exercise as well as the ancestors flag
  * Output: the predicted named entities
  * Implementation: it processes each sentence using nlp and it calls reconstruct_output to format it as in the dataset
"""

def process_dataset(dataset_text, expand, ancestors):
  pred = []
  for sentence in dataset_text:
    spacy_output = nlp(sentence[0])
    pred.append(reconstruct_output(spacy_output, expand, ancestors))
  return pred

"""get_accuracy(dataset_text, dataset_refs, expand = False):
  * Input:
    * dataset_text: the dataset as lists of sentences (text)
    * dataset_refs: the true named entities from the dataset
    * expand: whether to use the expanded version (ex3) or not
    * ancestors: whether to retrace the tree to find the first parent with dependency different from compound (ex3)
  * Output:
    * the scikit classification report of spaCy NER on the specified dataset (using the setting on convert_type function)
    * the predictions
  * Implementation: process the dataset and compute the report

"""

def get_accuracy(dataset_text, dataset_refs, expand = False, ancestors = False):
  pred = process_dataset(dataset_text, expand, ancestors)
  predicted = []

  for sentence in pred:
    for token in sentence:
      predicted.append(token[1])

  true_labels = []
  for sentence in dataset_refs:
    for token in sentence:
      true_labels.append(token[1])

  report = classification_report(true_labels, predicted)

  return report, pred

"""# -) Execution"""

dev_path = '/content/dataset/dev.txt'
train_path = '/content/dataset/train.txt'
test_path = '/content/dataset/test.txt'

"""Extract the datasets as:
 * *_txt: list of sentences as text
 * *_refs: the true named entities from each dataset
"""

dev_txt, dev_refs = import_dataset(dev_path)
train_txt, train_refs = import_dataset(train_path)
test_txt, test_refs = import_dataset(test_path)

"""1.1) Compute the token level accuracy for the test set"""

report_test, pred = get_accuracy(test_txt, test_refs)
print(report_test)

"""1.2) Compute the chunk level accuracy for the test set using the evaluate function provided by conll.py"""

results = conll.evaluate(test_refs, pred)
pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)

"""# -) Experiment
Here I was curious about using already tokenized text from the dataset (overriding spaCy tokenizer).
Despite spaCy's documentation reports that the performance should decrease (due to the fact that the tokenization methods may be different) the perfomance remains similar.
"""

from spacy.tokens import Doc

# function to replace spaCy tokenizer
def get_tokens(sentence):
  return Doc(nlp.vocab, sentence)

nlp.tokenizer = get_tokens

data = conll.read_corpus_conll(test_path)
pred = []

for s in data:
  sentence = []
  for token in s:
    sentence.append(token[0].split()[0])
  doc = nlp(sentence)
  pred.append(reconstruct_output(doc))

predicted = []
for sentence in pred:
  for token in sentence:
    predicted.append(token[1])

true_labels = []
for sentence in test_refs:
  for token in sentence:
    true_labels.append(token[1])

report = classification_report(true_labels, predicted)
print(report)

results = conll.evaluate(test_refs, pred)
pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)

"""# 2) Grouping of Entities.
Write a function to group recognized named entities using noun_chunks method of spaCy. Analyze the groups in terms of most frequent combinations (i.e. NER types that go together).

group_eintities(sentence):
  * Input: the sentence to process
  * Output: named entities grouped based on noun_chunk
  * Implementation:
    * first a set containing all the sentence entities is created
    * for each noun_chunk its entities are checked if they belong to the main entity set, if yes they will be part of the chunk group
    * the entities added are removed from the main set
    * in the end if the set is not empty, each remaining entity is added to a different new chunk (entities that were not in any chunk)
"""

# I checked whether all the entities of the sentence (doc.ents) are inside chunk.ents.
# there might be new entities inside chunk.ents, they will be discarded, so just the main entities from the sentence will be considered.

def group_entities(sentence):
  doc = nlp(sentence)
  groups = []
  entities = set()

  for ent in doc.ents:
    entities.add(ent)

  for chunk in doc.noun_chunks:
    group = []
    for span in chunk.ents:
      if span in entities:
        group.append(span.root.ent_type_)
        entities.remove(span)
    if(len(group) != 0):
      groups.append(group)

  for ent in entities:
    groups.append([ent.root.ent_type_])

  return groups

"""get_frequencies(dataset):
  * Input: the dataset where counting the combinations of entities
  * Output: a dict containing the frequencies for each combination
  * Implementation:
    * process each sentence of the dataset and groups its entities using group_entities
    * for each group create a tuple and increase the count of that group (combination) on the dict
"""

def get_frequencies(dataset):
  freq = dict()
  for sentence in dataset:
    groups = group_entities(sentence[0])
    for group in groups:
      group = tuple(group)
      if(group in freq):
        freq[group] += 1
      else:
        freq[group] = 1
  return freq

"""**Get the frequencies of the test set**
Print the dictionary containing the frequencies
"""

nlp = spacy.load('en_core_web_sm') # reset the tokenizer if 1.experiment has been run
freq = get_frequencies(test_txt)
print(freq)

"""# 3) One of the possible post-processing steps is to fix segmentation errors.
Write a function that extends the entity span to cover the full noun-compounds. Make use of compound dependency relation.

For this point I reused the get_accuracy function of the first point.
In this case the expand flag is set to True, this means that to the tokens with compound dependence will be assigned the tag of their parents (if possible).
"""

nlp = spacy.load('en_core_web_sm') # reset the tokenizer if 1.experiment has been run
report_test, pred = get_accuracy(test_txt, test_refs, expand=True, ancestors=False)
print(report_test)

results = conll.evaluate(test_refs, pred)
pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)

"""As we can see, using this method, the performance slightly decreases.
***

In order to try to improve the previous results I decided to also retrace the parent tree until an ancestor with dependence different from ```compound``` is found, then this ancestor will be used to set the tags to the token with ```compound``` dependency.
Below the run taking in account also the ancestors of the token
"""

nlp = spacy.load('en_core_web_sm') # reset the tokenizer if 1.experiment has been run
report_test, pred = get_accuracy(test_txt, test_refs, expand=True, ancestors=True)
print(report_test)

results = conll.evaluate(test_refs, pred)
pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)

"""This method further reduced the performance. One explanation for the decrease could be the fact that moving away from the token can lead to a decrease in the accuracy of the tag chosen for that particular token, this would means that the ```compound``` dependency is not that useful away from the token.
Other techniques may be useful to choose a better tag for the token.
"""

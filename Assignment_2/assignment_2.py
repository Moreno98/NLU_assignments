# -*- coding: utf-8 -*-
"""assignment_2.ipynb
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
  if(ent_type in ["ORG", "WORK_OF_ART"]):
    return "ORG"
  if(ent_type in ["GPE", "LOC"]):
    return "LOC"
  if(ent_type in ["LANGUAGE", "NORP", "EVENT", "LAW", "PRODUCT", "MONEY"]):
    return "MISC"
  if(ent_type in ["PERSON", "FAC"]):
    return "PER"
  return ""

"""convert_spacy(token, parent=None, parent_iob=None):
  * Input:
    * Token: the token to convert
    * Parent: the parent of the token to use to group the named entity
    * Parent_iob: this tag is used to know if there is a parent in the tree which already has the B tag, if yes the I tag will be assigned to the token, B otherwise.
  * Output: the tags converted in form ```iob-type``` as in the dataset
  * Implementation:
    * if parent is None it returns just the concatenation between the ```IOB``` tag and the named entity tag
    * if parent is set it returns the named entity from the parent if possible


"""

def convert_spacy(token, parent=None, parent_iob=None): # parent_iob maybe different from the current parent.ent_iob_
  if(parent == None): # exercise 1 usage
    if(convert_type(token.ent_type_) == ""):
      return "O"
    else:
      return f"{token.ent_iob_}-{convert_type(token.ent_type_)}"
  else: # exercise 3 usage
    iob = "I"
    if(parent_iob == "I"):
      iob = "B"
    if(parent.ent_type_ != ""):
      if(convert_type(parent.ent_type_) == ""):
        return "O"
      return f"{iob}-{convert_type(parent.ent_type_)}"
    else:
      if(token.ent_iob_ == "O"):
        return "O"
      else:
        if(convert_type(token.ent_type_) == ""):
          return "O"
        return f"{token.ent_iob_}-{convert_type(token.ent_type_)}"

"""reconstruct_output(doc, comp=False, ancestors = True):
  * Input:
    * Doc object from spaCy
    * comp (compound) flag to set on the third exercise
    * ancestors: this parameter is used in the final experiment of the third point, where I try just the direct parent of the token and not the entire tree
  * Output: list of sentences, each sentence contains the token "reconstructed" as in the dataset
  * Implementation:
    * given a token it uses whitespace to check if the token is part of a word in the dataset, if yes it concatenates the tokens with the same tag, otherwise the single token is used.
    * if comp is set to True, the tokens with compound dependency will have the same tag as the first parent with a dependency different from "compound" (if possible)
"""

def reconstruct_output(doc, comp=False, ancestors = True):
  output = []
  current_token = ""
  current_tag = ""
  first = True
  for token in doc:
    if(first):
        current_tag = convert_spacy(token)
        if((comp) and (token.dep_ == "compound")):
          parent = token.head
          parent_iob = token.head.ent_iob_
          while((parent.dep_ == "compound") and (ancestors)):
            if(parent_iob != "B"):
              parent_iob = parent.head.ent_iob_
            parent = parent.head
          current_tag = convert_spacy(token, parent, parent_iob)
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

"""process_dataset(dataset_text, expand, ancestors):
  * Input: the dataset as lists of sentences, expand is a flag used in the third exercise to expand the named entities as well as the ancestors
  * Output: the predicted named entities
  * Implementation: it processes each sentence using nlp and it calls reconstruct_output to format it as in the dataset
"""

def process_dataset(dataset_text, expand, ancestors):
  pred = []
  for sentence in dataset_text:
    spacy_output = nlp(sentence[0])
    pred.append(reconstruct_output(spacy_output, expand, ancestors))
  return pred

"""get_accuracy(dataset_text, dataset_refs, expand = False, ancestors = True):
  * Input:
    * dataset_text: the dataset as lists of sentences (text)
    * dataset_refs: the true named entities from the dataset
    * expand: whether to use the expanded version (ex3) or not
    * ancestors: usually True, just used in the final experiment (ex3)
  * Output:
    * the scikit classification report of spaCy NER on the specified dataset (using the setting on convert_type function)
    * the predictions
  * Implementation: process the dataset and compute the report

"""

def get_accuracy(dataset_text, dataset_refs, expand = False, ancestors = True):
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
freq_ordered = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
print("Frequencies:")
for group in freq_ordered:
  print(f"{group} -> {freq_ordered[group]}")

"""# 3) One of the possible post-processing steps is to fix segmentation errors.
Write a function that extends the entity span to cover the full noun-compounds. Make use of compound dependency relation.

For this point I reused the get_accuracy function of the first point.
In this case the expand flag is set to True, this means that to the tokens with compound dependence will be assigned the tag of their parents (if possible).
"""

nlp = spacy.load('en_core_web_sm') # reset the tokenizer if 1.experiment has been run
report_test, pred = get_accuracy(test_txt, test_refs, expand=True)
print(report_test)

results = conll.evaluate(test_refs, pred)
pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)

"""As we can see, using this method, the performance slightly decreases.
***

As final experiment, for curiosity, I try to use just the direct parent of the node and not the whole ancestors.
"""

nlp = spacy.load('en_core_web_sm') # reset the tokenizer if 1.experiment has been run
report_test, pred = get_accuracy(test_txt, test_refs, expand=True, ancestors = False)
print(report_test)

results = conll.evaluate(test_refs, pred)
pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)

"""As we can see the perfomance are similar with a slightly increase at chunk level, however this method does not have much sense since the IOB tag and named entity tag will be chosen without considering the whole span."""

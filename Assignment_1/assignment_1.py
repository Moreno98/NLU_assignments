# -*- coding: utf-8 -*-
"""Assignment_1.ipynb

**This is the code for the first assignment of the NLU course**
Here are some basic explanations of the workflow of the functions, please refer to the report (README.md) for further explanations.
"""

"""# Import spaCy and load the pipeline"""

import spacy
nlp = spacy.load("en_core_web_sm")

"""# A simple function which parses the input sentence

This function will be used in multiple points of the assignment
"""

def process_sentence(sentence):
  return nlp(sentence)

"""# 1) Extract a path of dependency relations from the ROOT to a token


*   path_to_root(sentence)
    * Input: the sentence to parse
    * Output: dict with tokens as keys and list of dipendencies (from ROOT to token) as values
    * Implementation:
      *   Process the sentence
      *   For each token extracts its dependency path

*   dependency_path(token)
    * Input: a token from Doc spaCy object
    * Output: the dependency path from ROOT to the token as a list
    * Implementation:
      * starting from the dependency of the token itself, retrace the dependency tree until finding the ROOT (token.head == token)
      * return the reverse of the dependency list in order to output ROOT -> token path


"""

def path_to_root(sentence):
  doc = process_sentence(sentence)
  d = {}
  for token in doc:
    d[token] = dependency_path(token)
  return d

def dependency_path(token):
  path = [token.dep_] # save the first dependency (the one of the token)
  exit = False
  # until the root has not been found
  while(not exit):
    # if this token is the root then exit
    if(token.head == token):
      exit = True
    else:
      # otherwise append the dependency of the head
      path.append(token.head.dep_)
      # start again from the head of the token
      token = token.head
  return list(reversed(path))

"""# 2) Extract subtree of the dependents given a token

*   get_subtree(token)
    * Input: a Token object from spaCy
    * Output: the list of the depentents from the token (its subtree)
    * Implementation: creation of a list containing the tokens inside its subtree

*   get_subtrees(sentence)
    * Input: the sentence
    * Output: dict
      * keys: the tokens of the list
      * values: for each token the list of its dependents (the subtree of the token)
    * Implementation:
      *   Process the sentence
      *   For each token extracts its subtree using get_subtree
"""

def get_subtree(token):
  return [dipendent for dipendent in token.subtree]

def get_subtrees(sentence):
  doc = process_sentence(sentence)
  d = {}
  for token in doc:
    d[token] = get_subtree(token)
  return d

"""# 3) Check if a given list of tokens (segment of a sentence) forms a subtree
*   check_subtree(sentence, subsequence)
    * Input: The sentence to parse and the ordered list of words to check
    * Output: True if the list forms a subtree of the sentence, False otherwise
    * Implementation:
      * Extract every subtree of the sentence using get_subtrees
      * Compare the input sequence with each subtree, if one matches the input sequence return True, False otherwise
"""

def check_subtree2(sentence, sequence):
  subtrees = get_subtrees(sentence)
  for token in subtrees:
    subtree = subtrees[token]
    if(len(subtree) == len(sequence)):
      if([t.text for t in subtree] == sequence):
        return True
  return False

"""# 4) Identify head of a span, given its tokens
* head_span(span)
  * Input: a string containing a span of words
  * Output: the word which is the head of the span
  * Implementation:
    * parse the span to have a Doc object
    * from the Doc object, take the Span of the entire sequence and return its root
"""

def head_span(span):
  doc = process_sentence(span)
  return doc[:].root.text

"""# 5) Extract sentence subject, direct object and indirect object spans
* extract(sentence)
  * Input: the sentence
  * Output: a dict with keys:
    * ``Subject``
    * ``Direct object``
    * ``Indirect object``
  and the corresponding list of words as values, for the subject it returns a list for each subtree depending on the subject

  * Implementation:
    * For each token of the Doc object check if the token is:
      * ``nsubj`` which stand for Nominal Subject
      * ``nsubjpass`` which stand for Nominal Subject Passive
      * ``csubj`` which stand for Clausal Subject
      * ``csubjpass`` which stand for Clausal Subject Passive
      * ``expl`` which stand for Expletive Subject
      * ``dobj`` which stand for Direct Object
      * ``dative`` which stand for Indirect Object
    * For each of these tokens save its subtree
    * For each different type of subject of the sentence a list is created with each subtree, these lists are then returned in the final dict
"""

def extract(sentence):
  doc = process_sentence(sentence)
  d = {}
  d["nsubj"], d["nsubjpass"], d["dobj"], d["dative"], d["csubj"], d["csubjpass"], d["expl"] = [], [], [], [], [], [], []
  for token in doc:
    if((token.dep_ == "nsubj") or
       (token.dep_ == "nsubjpass") or
       (token.dep_ == "csubj") or
       (token.dep_ == "csubjpass") or
       (token.dep_ == "expl") or
       (token.dep_ == "dobj") or
       (token.dep_ == "dative")):
      for t in token.subtree:
        d[token.dep_].append(t.text)
  subject = []
  for key in d:
    if(key in ["nsubj", "nsubjpass", "csubj", "csubjpass", "expl"]):
      if(len(d[key]) != 0):
        subject.append(d[key])

  return {"Subject": subject if len(subject)!=0 else None,
          "Direct object": d["dobj"] if len(d["dobj"])!=0 else None,
          "Indirect object": d["dative"] if len(d["dative"])!=0 else None}

"""# Execution
Script which calls and executes the previous functions.
"""

# list of sentences used to test the functions and the underlying ideas:

# sentence = "Luca has been killed by a car"
# sentence = "What she said is interesting"
# sentence = "That his theory was flawed soon became obvious"
# sentence = "What I need is a long holiday"
# sentence = "To become an opera singer takes years of training"
# sentence = "Being the chairman is a huge responsibility"
# sentence = "There is a fly in my soup"
# sentence = "That Fred is a funny comedian"
# sentence = "There is a woman in the bus who is called Diana"
# sentence = "There is a toy airplane on the grass in the backyard."
# sentence = "There is a red house over iyonder"
# sentence = "What she said makes sense"
# sentence = "He gave me a nice gift for Christmas."
# sentence = "I saw the man"

sentence = "I watched a movie with Sisko."

print(f"Sentence: {sentence}\n")

print("1) ------ ROOT to token -------")
d = path_to_root(sentence)
for token in d:
  print(f"Token: {token} | path : {d[token]}")

print("\n2) --------- Subtrees ---------")
subtrees = get_subtrees(sentence)
for token in subtrees:
  print(f"Token: {token}")
  print(f"--> Subtree: {subtrees[token]}")

print("\n3) --------- Check subtree ---------")
seq = ["I"]
seq1 = ["a", "movie", "with", "Sisko"]
seq2 = ["movie", "a", "with", "Sisko"]
print(f"Sequence to check: {seq}")
print(f"Is it a subtree? {check_subtree2(sentence, seq)}")
print(f"Sequence to check: {seq1}")
print(f"Is it a subtree? {check_subtree2(sentence, seq1)}")
print(f"Sequence to check: {seq2}")
print(f"Is it a subtree? {check_subtree2(sentence, seq2)}")

print("\n4) --------- Head of span ----------")
span = "I man world"
print(f"Span: {span}")
print(f"Head: {head_span(span)}")

print("\n5) --------- Subject, Object, Indirect object ----------")
extracted = extract(sentence)
for value in extracted:
  print(value, extracted[value])

sentence = "Luca has been killed by a car."
print(f"\nExample with: {sentence}")
print("In this case 'Luca' is a Nominal Subject Passive")
extracted = extract(sentence)
for value in extracted:
  print(value, extracted[value])

sentence = "There is a woman in the bus who is called Diana."
print(f"\nExample with: {sentence}")
print("In this case 'who' is a Nominal Subject Passive and 'There' is a Expletive Subject")
extracted = extract(sentence)
for value in extracted:
  print(value, extracted[value])

"""# Optional part

**Imports**
"""

import nltk
from nltk.parse.transitionparser import TransitionParser
from nltk.parse.transitionparser import Configuration
from sklearn.neural_network import MLPClassifier
nltk.download('dependency_treebank')
from nltk.parse import DependencyEvaluator
import tempfile
from nltk.corpus import dependency_treebank
from sklearn.datasets import load_svmlight_file
import pickle
from nltk.parse import DependencyEvaluator
from os import remove
from sklearn.ensemble import RandomForestClassifier

"""TransitionParser_MLP(TransitionParser):
I extend the TransitionParser from nltk.parse.transitionparser to change the model used during trainining and predicting.
I choose a MLP classifier from sklearn and since the neural netwroks have a fluctuation on the accuracy, I take the avearge accuracy of 10 runs.
Then I test the new model against the default one (SVM) from the default TransitionParser
"""

class TransitionParser_MLP(TransitionParser):

  def __init__(self, alg_option):
    TransitionParser.__init__(self, alg_option)

  def train(self, depgraphs, modelfile, verbose=True):
      try:
          input_file = tempfile.NamedTemporaryFile(
              prefix="transition_parse.train", dir=tempfile.gettempdir(), delete=False
          )

          if self._algorithm == self.ARC_STANDARD:
              self._create_training_examples_arc_std(depgraphs, input_file)
          else:
              self._create_training_examples_arc_eager(depgraphs, input_file)

          input_file.close()

          x_train, y_train = load_svmlight_file(input_file.name)

          model = MLPClassifier()

          model.fit(x_train, y_train)
          # Save the model to file name (as pickle)
          pickle.dump(model, open(modelfile, "wb"))
      finally:
          remove(input_file.name)

# compare the performances
# Since we don't have dependency labels (as in the lab) I take in account just the uas (unlabeled)

times = 10

p = 0
for i in range(times):
  tp = TransitionParser_MLP('arc-standard')
  tp.train(dependency_treebank.parsed_sents()[:200], 'tp.model')
  parses = tp.parse(dependency_treebank.parsed_sents()[-20:], 'tp.model')
  de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-20:])

  _, uas = de.eval()
  p += uas

print(f"[arc-standard] Average accuracy of MLP: {p/times}")

tp = TransitionParser('arc-standard')
tp.train(dependency_treebank.parsed_sents()[:200], 'tp.model')
parses = tp.parse(dependency_treebank.parsed_sents()[-20:], 'tp.model')

de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-20:])
_, uas = de.eval()

print(f"[arc-standard] Accuracy of the default model (SVC): {uas}")

p = 0
for i in range(times):
  tp = TransitionParser_MLP('arc-eager')
  tp.train(dependency_treebank.parsed_sents()[:200], 'tp.model')
  parses = tp.parse(dependency_treebank.parsed_sents()[-20:], 'tp.model')

  de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-20:])
  _, uas = de.eval()
  p += uas

print(f"[arc-eager] Average accuracy of MLP: {p/times}")

tp = TransitionParser('arc-eager')
tp.train(dependency_treebank.parsed_sents()[:200], 'tp.model')
parses = tp.parse(dependency_treebank.parsed_sents()[-20:], 'tp.model')

de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-20:])
_, uas = de.eval()

print(f"[arc-eager] Accuracy of the default model (SVC): {uas}")

"""Here I try the random forest as other model with the results which are below."""

class TransitionParser_RF(TransitionParser):

  def __init__(self, alg_option):
    TransitionParser.__init__(self, alg_option)

  def train(self, depgraphs, modelfile, verbose=True):
      try:
          input_file = tempfile.NamedTemporaryFile(
              prefix="transition_parse.train", dir=tempfile.gettempdir(), delete=False
          )

          if self._algorithm == self.ARC_STANDARD:
              self._create_training_examples_arc_std(depgraphs, input_file)
          else:
              self._create_training_examples_arc_eager(depgraphs, input_file)

          input_file.close()

          x_train, y_train = load_svmlight_file(input_file.name)

          model = clf = RandomForestClassifier()

          model.fit(x_train, y_train)
          # Save the model to file name (as pickle)
          pickle.dump(model, open(modelfile, "wb"))
      finally:
          remove(input_file.name)

times = 10

p = 0
for i in range(times):
  tp = TransitionParser_RF('arc-standard')
  tp.train(dependency_treebank.parsed_sents()[:200], 'tp.model')
  parses = tp.parse(dependency_treebank.parsed_sents()[-20:], 'tp.model')
  de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-20:])

  _, uas = de.eval()
  p += uas

print(f"[arc-standard] Average accuracy of the random forests: {p/times}")

p = 0
for i in range(times):
  tp = TransitionParser_RF('arc-eager')
  tp.train(dependency_treebank.parsed_sents()[:200], 'tp.model')
  parses = tp.parse(dependency_treebank.parsed_sents()[-20:], 'tp.model')

  de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-20:])
  _, uas = de.eval()
  p += uas

print(f"[arc-eager] Average accuracy of the random forests: {p/times}")

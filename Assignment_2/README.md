# Assignment 2
This is the colab notebook of the assignment: [![Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HHmU13WIT6RA55Mmqkc4uRU7-Hk0Et11?usp=sharing)  
## 1) Evaluate spaCy NER on CoNLL 2003 data (provided)
In this point of the assignmnet I made use of the conll script provided by the Professor for importing the dataset and to compute the chunk level evaluation.  
***
First of all I created a function to import the dataset specified as input.  
This function extracts the required information from the dataset: the sentences as strings (so just the tokens converted) and the tokens with their named entities to be used in the evaluation process.
* import_dataset(path):  
  * Input: the path of the dataset to load
  * Output: two lists:
    * text_dataset: contains the lists of sentences of the dataset as text
    * dataset: contais the pair (token, name entity) for each token, divided in sentences (one list for each sentence)
  * Implementation: 
    * it reads the dataset using conll function
    * for each sentence it extracts the tokens as text or the tuple (token, name entity) 

> Note: I removed the -DOCSTART- token from the dataset, however this token is not affecting the predictions since it is always correctly classified.

```python
def import_dataset(path):
  data = conll.read_corpus_conll(path)
  text_dataset = []
  dataset = []
  for t in data:
    sentence = []
    txt = ""
    for t2 in t:
      if(t2[0].split()[0] != "-DOCSTART-"):
        sentence.append((t2[0].split()[0], t2[0].split()[3]))
        txt += str(t2[0].split()[0]) + " "
    dataset.append(sentence)
    text_dataset.append([txt])
  return text_dataset, dataset
```
---

The conll2003 dataset has a different format with respect to the spaCy named entities therefore there is the need to convert one format to the other.  
The format which is best suited for conversion is the spaCy format since it is more detailed (its classes are a subset of the conll2003 classes), 
for example spaCy has two different classes for representing locations: GPE (geopolitical entities) and LOC, both can be converted in the dataset class LOC. The other way around 
is not possible as we do not know which spaCy class (GPE or LOC) the dataset class (LOC) refers to.   
I decided the following map for the conversion:
  * ```ORG``` and ```WORK_OF_ART``` to ```ORG```
  * ```GPE``` and ```LOC``` to ```LOC```
  * ```PERSON``` and ```FAC``` to ```PER```
  * ```LANGUAGE```, ```NORP```, ```EVENT```, ```LAW```, ```PRODUCT```, ```MONEY``` to ```MISC``` (miscellaneous)  
  * the remaining will not have a tag

> Note 1: The accuracy on the dataset is highly dependent on the previous decisions

> Note 2: Initially I divided the tags in a different way, then, after some testing, I noticed that the MISC accuracy had very low performance with respect to the other tags, so I started digging about why. I found that some tags, which previously I was converting as MISC, were strongly decreasing the accuracy, so I removed them. As final experiment I tried various combination of the tags and finally I ended up with the current configuration which, obviosuly, may not be the best.

The following function takes in input the named entity type from spaCy and it returns its conversion. 

* convert_type(ent_type):
  * Input: named entity from spaCy
  * Output: the named entity converted in the dataset format
  * Implementation: assign a specific named entity from the dataset format to each named entity from spaCy

```python
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
```
---
Another difference from the dataset format is that spaCy separates the ```IOB``` tags from the named entity tags, so I created a function which concatenates these tags to get 
the same format as the dataset (for instance ```I-ORG``` with ```I``` as ```IOB``` and ```ORG``` as named entity).  
Moreover this function has been updated for the third point of the assignment. When the parent node is not set, the function behaves as previously indicated, instead when it is set, the parent parameter is the parent of a ```compound``` dependency token, in this case the token will receive the same type as the parent to try to fix the segmentation error (as asked in the last point). Here the ```IOB``` tag is important, infact we need to know the position of the tag to undestand which iob tag has to be assigned to the token. The idea here is to find out if other parents have the ```B``` tag, if yes then the token will have the ```I``` tag, otherwise I assume that the token iteself is at the beginning of the span, so the ```B``` tag is assigned. This is implemented using the ```parent_iob``` paremeter which will contain the B tag if one of the ancestors was at the beggining of the span, otherwise it will contain an I or O tag.  
If the parent does not have a named entity tag the token will receive its representation if it exists otherwise "O".

> Note: if the convert_type function returns "", then the predicted tag will be just "O".

* convert_spacy(token, parent=None, parent_iob=None):
  * Input: 
    * Token: the token to convert
    * Parent: the parent of the token to use to group the named entity
    * Parent_iob: this tag is used to know if there is a parent in the tree which already has the B tag, if yes the I tag will be assigned to the token, B otherwise.
  * Output: the tags converted in form ```iob-type``` as in the dataset
  * Implementation: 
    * if parent is None it returns just the concatenation between the ```IOB``` tag and the named entity tag
    * if parent is set it returns the named entity from the parent if possible

```python
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
```
---
SpaCy tokenizes the text in a different way with respect to the dataset, so this function will reconstruct the spaCy's output to match the dataset one. I made use of the whitespace flag in order to group the tokens which were together in the dataset.  
If comp is set it uses convert_spacy with the parent setting, therefore this function checks whether the token has a ```compound``` dependency or not.  
Moreover, after some though, it occured to me that it may be possible to have multiple parents with dependency ```compound```, so I updated the function to check all the ancestors.  
As final experiment, for curiosity, I tried to use just the direct parent of the node and not the whole ancestors, so I added the parameter ancestors for this purpose.

> Note: To expand the named entity I took in account the fact that between the token and the head there could be another entity (overlapping of entities), I tested this and in test.txt it seems this is not the case, so I just check if the compound token has the same entity of the parent or an empty antity, if this is the case its tag is updated (this final change increases the performance of about 4%).

* reconstruct_output(doc, comp=False, ancestors = True):
  * Input: 
    * Doc object from spaCy 
    * comp (compound) flag to set on the third exercise
    * ancestors: this parameter is used in the final experiment of the third point, where I try just the direct parent of the token and not the entire tree
  * Output: list of sentences, each sentence contains the token "reconstructed" as in the dataset
  * Implementation: 
    * given a token it uses whitespace to check if the token is part of a word in the dataset, if yes it concatenates the tokens with the same tag, otherwise the single token is used.  
    * if comp is set to True, the tokens with compound dependency will have the same tag as the first parent with a dependency different from "compound" (if possible) 

```python
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
          if(token.ent_type_ in ["", parent.ent_type_]): # change entity only if the token does not belong to another entity
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
```
---
The following function just processes the dataset and returns the predicted named entities.

* process_dataset(dataset_text, expand, ancestors):
  * Input: the dataset as lists of sentences, expand is a flag used in the third exercise to expand the named entities as well as the ancestors
  * Output: the predicted named entities
  * Implementation: it processes each sentence using nlp and it calls reconstruct_output to format it as in the dataset 
  
```python
def process_dataset(dataset_text, expand, ancestors):
  pred = []
  for sentence in dataset_text:
    spacy_output = nlp(sentence[0])
    pred.append(reconstruct_output(spacy_output, expand, ancestors))
  return pred
```
---
This function processes the dataset using ```process_dataset```, then it extracts the predictions and true_labels to compute the classification report from ```scikit-learn```.

* get_accuracy(dataset_text, dataset_refs, expand = False, ancestors = True):
  * Input: 
    * dataset_text: the dataset as lists of sentences (text)
    * dataset_refs: the true named entities from the dataset
    * expand: whether to use the expanded version (ex3) or not
    * ancestors: usually True, just used in the final experiment (ex3)
  * Output:
    * the scikit classification report of spaCy NER on the specified dataset (using the setting on convert_type function)
    * the predictions
  * Implementation: process the dataset and compute the report
  
```python
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
  ```
  ---
  ### Execution
  For the execution part I call the previous functions, print the report and compute the chunk level accuracy using conll.evaluate function from the Professor script.
  
  #### 1.1) Token level evaluation:
  ```
                precision    recall  f1-score   support

         B-LOC       0.77      0.68      0.72      1668
        B-MISC       0.58      0.55      0.57       702
         B-ORG       0.51      0.31      0.38      1661
         B-PER       0.79      0.63      0.70      1617
         I-LOC       0.57      0.53      0.55       257
        I-MISC       0.26      0.36      0.30       216
         I-ORG       0.41      0.52      0.46       835
         I-PER       0.82      0.79      0.80      1156
             O       0.95      0.97      0.96     38323

      accuracy                           0.90     46435
     macro avg       0.63      0.59      0.61     46435
  weighted avg       0.90      0.90      0.90     46435
  ```
  #### 1.2) Chunk level evaluation:
  
  |     |   p	  |   r	  |  f	|    s|
  | -- | -- | -- | -- | -- |
  |MISC	|0.576	|0.546	|0.560|	702|
  |ORG	|0.460	|0.277	|0.346|	1661|
  |PER	|0.760	|0.610	|0.677|	1617|
  |LOC	|0.755	|0.667	|0.708|	1668|
  |total|	0.663	|0.521	|0.583|	5648|
  
  ### Experiment
  I was curious about using already tokenized text from the dataset (overriding spaCy tokenizer).  
  SpaCy's documentation reports that the performance should decrease (due to the fact that the tokenization methods may be different), in this case the perfomance slightly decreases, so spaCy's documentation is right.
  Here the results:
  #### Token level evaluation:
  ```
              precision    recall  f1-score   support

         B-LOC       0.78      0.70      0.74      1668
        B-MISC       0.58      0.55      0.56       702
         B-ORG       0.50      0.30      0.38      1661
         B-PER       0.77      0.61      0.68      1617
         I-LOC       0.60      0.62      0.61       257
        I-MISC       0.27      0.37      0.31       216
         I-ORG       0.41      0.52      0.46       835
         I-PER       0.80      0.76      0.78      1156
             O       0.95      0.97      0.96     38323

      accuracy                           0.90     46435
     macro avg       0.63      0.60      0.61     46435
  weighted avg       0.90      0.90      0.90     46435
  ```
  #### Chunk level evaluation:
  
  |     |   p	  | r	  | f	    |  s|
  | -- | -- | -- | -- | -- |
  |MISC	|0.571	|0.541|	0.556	|702|
  |ORG	|0.444	|0.273|	0.338	|1661|
  |PER	|0.748	|0.592|	0.661	|1617|
  |LOC	|0.766	|0.695|	0.729	|1668|
  |total|	0.659	|0.522|	0.583	|5648|

***
## 2) Grouping of Entities
In order to group the named entities based on the chunks I iterate the dataset over the sentences and for each of them I group the entities inside the same chunk.  
The following function groups the entities of a given sentence. I make use of ```noun_chunk``` to know the chunks inside the sentence and the ```chunk.ents``` to find which entities are inside the chunks. I checked whether all the entities of the sentence (```doc.ents```) are inside ```chunk.ents```, so this method can be used. Moreover I found that there might be new entities inside ```chunk.ents``` (w.r.t. ```doc.ents```), they will be discarded, so just the main entities from the sentence (```doc.ents```) will be considered.

* group_eintities(sentence):
  * Input: the sentence to process
  * Output: named entities grouped based on noun_chunk
  * Implementation:
    * first a set containing all the sentence entities is created
    * for each noun_chunk its entities are checked if they belong to the main entity set, if yes they will be part of the chunk group
    * the entities added are removed from the main set
    * in the end if the set is not empty, each remaining entity is added to a different new chunk (entities that were not in any chunk)

```python
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
```
---

This function computes the frequencies of a given dataset, it uses ```group_entities``` to group the entities of each sentence, then each group is converted in tuple in order to be used as key in the ```freq``` dictionary; this dictionary is used to count the frequency of each combination.

> Note: I take in account the order of the entities, because, in my opinion, different ordering could have different meaning in the main sentence so in that case they have to be considered different.

* get_frequencies(dataset):
  * Input: the dataset where counting the combinations of entities
  * Output: a dict containing the frequencies for each combination
  * Implementation:
    * process each sentence of the dataset and groups its entities using group_entities
    * for each group create a tuple and increase the count of that group (combination) on the dict
```python
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
```
---
### Execution
I simply run ```get_frequencies``` on the test set and print, ordered, the dictionary of the frequencies.
***
## 3) One of the possible post-processing steps is to fix segmentation errors.  
**Write a function that extends the entity span to cover the full noun-compounds. Make use of compound dependency relation.**

For this part I reused the ```get_accuracy``` function but, this time, with the ```expand``` flag set to True. In this setting the tokens with ```compound``` dependency will receive the same tag of their parents.  
Initially I just took the first parent of the token, but after some thoughts it occured to me that multiple ancestors on the tree may have the ```compound``` dependency, so I updated the function to retrace the tree to take in account also the ancestors (the entire entity) as I explained previously on the ```convert_spacy``` explanation.

> Note 1: To expand the named entity I took in account the fact that between the token and the head there could be another entity (overlapping of entities), I tested this and in test.txt it seems this is not the case, so I just check if the compound token has the same entity of the parent or an empty antity, if this is the case its tag is updated (this final change increases the performance of about 4%).  

> Note 2: the expansion of the entity can be chosen in a smarter way, for example if we have a situation like ["B-ORG", "O", "O"] with the third token compound of the first, in the current setting, the output would be ["B-ORG", "O", "I-ORG"] (if the second token is not a compound). This setting can be updated to expand also the second token to have ["B-ORG", "I-ORG", "I-ORG"] which may lead to better performance.

These are the results:
### Token level accuracy
```
              precision    recall  f1-score   support

       B-LOC       0.77      0.69      0.73      1668
      B-MISC       0.57      0.57      0.57       702
       B-ORG       0.44      0.35      0.39      1661
       B-PER       0.64      0.64      0.64      1617
       I-LOC       0.47      0.50      0.48       257
      I-MISC       0.30      0.35      0.32       216
       I-ORG       0.43      0.40      0.41       835
       I-PER       0.81      0.74      0.77      1156
           O       0.95      0.97      0.96     38323

    accuracy                           0.89     46435
   macro avg       0.60      0.58      0.59     46435
weighted avg       0.89      0.89      0.89     46435
```
### Chunk level accuracy

|     |  p	  | r	   |   f	| s  |
| -- | -- | -- | -- | -- |
|MISC	|0.559	|0.566 |0.562	|702 |
|ORG	|0.344	|0.278 |0.307	|1661|
|PER	|0.573	|0.583 |0.578	|1617|
|LOC	|0.730	|0.673 |0.700	|1668|
|total|	0.558	|0.517 |0.537	|5648|

As we can see, using this method, the performance slightly decreases.
***
As final experiment, for curiosity, I tried not to retrace the tree (as I initially thought) in order to end on the direct parent of the token, this method seems better then the previous one with these results:
### Token level accuracy
```
              precision    recall  f1-score   support

       B-LOC       0.77      0.69      0.72      1668
      B-MISC       0.57      0.58      0.58       702
       B-ORG       0.43      0.35      0.38      1661
       B-PER       0.66      0.64      0.65      1617
       I-LOC       0.48      0.50      0.49       257
      I-MISC       0.30      0.34      0.32       216
       I-ORG       0.44      0.39      0.42       835
       I-PER       0.81      0.74      0.78      1156
           O       0.95      0.97      0.96     38323

    accuracy                           0.90     46435
   macro avg       0.60      0.58      0.59     46435
weighted avg       0.89      0.90      0.89     46435
```

### Chunk level accuracy
|     |  p	  |  r	  |  f	  | s  |
| -- | -- | -- | -- | -- |
|MISC	|0.558	|0.573	|0.565	|702 |
|ORG	|0.334	|0.276	|0.302	|1661|
|PER	|0.593	|0.583	|0.588	|1617|
|LOC	|0.729	|0.673	|0.700	|1668|
|total|	0.560	|0.518	|0.538	|5648|

As we can see the perfomance are similar with a slightly increase at chunk level, however this method does not have much sense since the ```IOB``` tag and named entity tag will be chosen without considering the whole entity. 
***
> I noticed too late that my idea about assigning the IOB tag to the compound tokens has a bug, the IOB tag assigned is always "I" since we already have a "B" tag somewhere; to fix this the .i attribute can be used to understand the position of the tokens inside the entity to assign the correct IOB tag accordingly, then, in this situation, we should also change the previous "B" tag to "I" to maintain the correct matching in the entity.

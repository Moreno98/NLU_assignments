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

```python
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
```
---

The conll2003 dataset has a different format with respect to the spaCy named entities therefore there is the need to convert one format to the other.  
The format which is best suited for conversion is the spaCy format since it is more detailed (its classes are a subset of the conll2003 classes), 
for example spaCy has two different classes for representing locations: GPE (geopolitical entities) and LOC, both can be converted in the dataset class LOC. The other way around 
is not possible as we do not know which spaCy class (GPE or LOC) the dataset class (LOC) refers to.   
I decided the following map for the conversion:
  * ```ORG``` to ```ORG```
  * ```GPE``` and ```LOC``` to ```LOC```
  * ```PERSON``` to ```PER```
  * all the others to ```MISC``` (miscellaneous)  

> Note: The accuracy on the dataset is highly dependent on the previous decision

The following function takes in input the named entity type from spaCy and it returns its conversion. 

* convert_type(ent_type):
  * Input: named entity from spaCy
  * Output: the named entity converted in the dataset format
  * Implementation: assign a specific named entity from the dataset format to each named entity from spaCy

```python
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
```
---
Another difference from the dataset format is that spaCy separates the ```IOB``` tags from the named entity tags, so I created a function which concatenates these tags to get 
the same format as the dataset (for instance ```I-ORG``` with ```I``` as ```IOB``` and ```ORG``` as named entity).  
Moreover this function has been adapted for the third point of the assignment. When the parent node is not set, the function behaves as previously indicated, instead when it is set, 
the parent is the parent of a ```compound``` dependency token, in this case the token will receive the same type as the parent to try to fix the segmentation error (as asked in the last point).

> Note: When parent is set, if the token has an IOB ```O``` I decided to use the named entity tag from the parent along side to a ```I``` IOB tag (I assumed that the child token is 
Inside the span, which is not always the case).  
If the parent does not have a named entity tag then just the ```O``` tag is returned.

* convert_spacy(token, parent=None):
  * Input: the token to convert, the parent of the token to use in the third exercise
  * Output: the tags converted in form ```iob-type``` as in the dataset
  * Implementation: 
    * if parent is None it returns just the concatenation between the ```IOB``` tag and the named entity tag
    * if parent is set it returns the named entity from the parent if possible

```python
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
```
---
SpaCy tokenizes the text in a different way with respect to the dataset, so this function will reconstruct the spaCy's output to match the dataset one. I made use of the whitespace flag in order to group the tokens which were together in the dataset.  
If comp is set it uses convert_spacy with the parent setting, therefore this function checks whether the token has a ```compound``` dependence or not.  
Moreover, after some though, it occured to me that it may be possible to have even the parent with dependency ```compound```, so I decided to add a parameter ```ancestors```, if set the function will use as parent the first ancestor with a dependency different from ```compound```.

* reconstruct_output(doc, comp=False):
  * Input: 
    * Doc object from spaCy 
    * comp (compound) flag to set on the third exercise
    * ancestors: flag to use to reach the first ancestor with dependency different from "compound"
  * Output: list of sentences, each sentence contains the token "reconstructed" as in the dataset
  * Implementation: 
    * given a token it uses whitespace to check if the token is part of a word in the dataset, if yes it concatenates the tokens with the same tag, otherwise the single token is used.  
    * if comp is set to True, the tokens with compound dependency will have the same tag as their parents, moreover if also ancestors is set, the parent passed to convert_spacy will be the first ancestor with a dependency different from "compound"

```python
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
```
---
The following function just processes the dataset and returns the predicted named entities.

* process_dataset(dataset_text, expand):
  * Input: the dataset as lists of sentences, expand is a flag used in the third exercise as well as the ancestors flag
  * Output: the predicted named entities
  * Implementation: it processes each sentence using nlp and it calls ```reconstruct_output``` to format it as in the dataset 
  
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

* get_accuracy(dataset_text, dataset_refs, expand = False):
  * Input: 
    * dataset_text: the dataset as lists of sentences (text)
    * dataset_refs: the true named entities from the dataset
    * expand: whether to use the expanded version (ex3) or not
    * ancestors: whether to retrace the tree to find the first parent with dependency different from compound
  * Output:
    * the scikit classification report of spaCy NER on the specified dataset (using the setting on convert_type function)
    * the predictions
  * Implementation: process the dataset and compute the report
  
```python
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
  ```
  ---
  ### Execution
  For the execution part I call the previous functions, print the report and compute the chunk level accuracy using conll.evaluate function from the Professor script.
  
  #### 1.1) Token level evaluation:
  ```
       precision    recall  f1-score   support

         B-LOC       0.77      0.68      0.72      1668
        B-MISC       0.10      0.57      0.17       702
         B-ORG       0.52      0.31      0.38      1661
         B-PER       0.80      0.63      0.70      1617
         I-LOC       0.57      0.53      0.55       257
        I-MISC       0.05      0.40      0.09       216
         I-ORG       0.42      0.51      0.46       835
         I-PER       0.84      0.79      0.81      1156
             O       0.95      0.86      0.90     38554

      accuracy                           0.81     46666
     macro avg       0.56      0.59      0.53     46666
  weighted avg       0.89      0.81      0.84     46666
  ```
  #### 1.2) Chunk level evaluation:
  
  |     |  p	 |    r|	    f|	 s  |
  | -- | -- | -- | -- | -- |
  |MISC	|0.100 |0.554|	0.169|	702|
  |PER	|0.774 |0.609|	0.681|	1617|
  |LOC	|0.755 |0.667|	0.708|	1668|
  |ORG	|0.464 |0.276|	0.346|	1661|  
  |total|	0.385|0.521|	0.443|	5648|
  
  ### Experiment
  I was curious about using already tokenized text from the dataset (overriding spaCy tokenizer).  
  Despite spaCy's documentation reports that the performance should decrease (due to the fact that the tokenization methods may be different) the perfomance remains similar.  
  Here the results:
  #### Token level evaluation:
  ```
    precision    recall  f1-score   support

         B-LOC       0.78      0.70      0.74      1668
        B-MISC       0.11      0.56      0.18       702
         B-ORG       0.50      0.30      0.38      1661
         B-PER       0.79      0.61      0.69      1617
         I-LOC       0.60      0.62      0.61       257
        I-MISC       0.05      0.40      0.09       216
         I-ORG       0.42      0.52      0.46       835
         I-PER       0.82      0.76      0.78      1156
             O       0.94      0.86      0.90     38554

      accuracy                           0.81     46666
     macro avg       0.56      0.59      0.54     46666
  weighted avg       0.89      0.81      0.84     46666
  ```
  #### Chunk level evaluation:
  
  |    | p	|  r	|  f	|  s |
  | -- | -- | -- | -- | -- |
  |MISC|	0.105|	0.550|	0.177|	702|
  |PER |	0.761|	0.590|	0.665|	1617|
  |LOC |	0.766|	0.695|	0.729|	1668|
  |ORG |	0.448|	0.272|	0.339|	1661|
  |total|	0.397|	0.523|	0.451|	5648|

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
I simply run ```get_frequencies``` on the test set and print the dictionary of the frequencies.
***
## 3) One of the possible post-processing steps is to fix segmentation errors.  
**Write a function that extends the entity span to cover the full noun-compounds. Make use of compound dependency relation.**

For this part I reuse the ```get_accuracy``` function but, this time, with the ```expand``` flag set to True. In this setting the tokens with ```compound``` dependency will receive the same tag of their parents (please refer to the ```convert_spacy``` function for further details).  
These are the results:
### Token level accuracy
```
precision    recall  f1-score   support

       B-LOC       0.77      0.67      0.72      1668
      B-MISC       0.10      0.57      0.17       702
       B-ORG       0.51      0.30      0.38      1661
       B-PER       0.79      0.63      0.70      1617
       I-LOC       0.48      0.53      0.50       257
      I-MISC       0.05      0.41      0.09       216
       I-ORG       0.40      0.52      0.45       835
       I-PER       0.71      0.79      0.75      1156
           O       0.95      0.85      0.90     38554

    accuracy                           0.80     46666
   macro avg       0.53      0.59      0.52     46666
weighted avg       0.88      0.80      0.84     46666
```
### Chunk level accuracy

|     |p	  | r	    |f	    |s   |
| -- | -- | -- | -- | -- |
|MISC	|0.098	|0.553	|0.167|	702|
|PER	|0.669	|0.607	|0.637| 1617|
|LOC	|0.739	|0.662	|0.699|	1668|
|ORG	|0.445	|0.273	|0.338|	1661|
|total|	0.370	|0.518	|0.431|	5648|

As we can see, using this method, the performance slightly decreases; to try to improve it a different setting on ```convert_spacy``` can be tried, for instance I tried to replace the ```IOB``` tag assigned if the token has ```O``` as IOB from ```I``` to ```B```, the perfomance remains the same with an increase on the ```B-*``` tags and a decrease on the ```I-*``` tags (obvously). Another experiment maybe to choose this tag in a smarter way.

---
In order to try to improve the previous results I decided to also retrace the parent tree until an ancestor with dependency different from ```compound``` is found, then this ancestor will be used to set the tags to the token with ```compound``` dependency.  
These are the new results:
### Token level accuracy
```
precision    recall  f1-score   support

       B-LOC       0.77      0.66      0.71      1668
      B-MISC       0.10      0.56      0.17       702
       B-ORG       0.51      0.30      0.37      1661
       B-PER       0.77      0.63      0.69      1617
       I-LOC       0.46      0.52      0.49       257
      I-MISC       0.05      0.40      0.09       216
       I-ORG       0.40      0.51      0.45       835
       I-PER       0.67      0.79      0.72      1156
           O       0.95      0.85      0.90     38554

    accuracy                           0.80     46666
   macro avg       0.52      0.58      0.51     46666
weighted avg       0.88      0.80      0.83     46666
```
### Chunk level accuracy
|     | p	   |  r	    |   f	  | s |
| -- | -- | -- | -- | -- |
|ORG	| 0.436|	0.263	|0.328	|1661|
|LOC	| 0.742|	0.653	|0.695	|1668|
|MISC	| 0.098|	0.550	|0.167	|702|
|PER	| 0.667|	0.606	|0.635	|1617|
|total| 0.368|	0.512	|0.428	|5648|

This method further reduced the performance. One explanation for the decrease could be the fact that moving away from the token can lead to a decrease in the accuracy of the tag chosen for that particular token, this would means that the ```compound``` dependency is not that useful away from the token.  
Other techniques may be useful to choose a better tag for the token.  

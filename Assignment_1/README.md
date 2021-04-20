# Assignment 1

This is the colab notebook of the assignment: [![Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Cj62PqJy_SbmL0Tw_pcLECfzy3TdzJAK?usp=sharing)  
Note: I assume that each input sentence to each function is one (a string containing one sentence), 
if the string contains multiple sentences the code can be easily updated adding ```doc.sents``` in order to extract the object containing the Spans of each sentence.

## 1) Extract a path of dependency relations from the ROOT to a token

For the first point of the assignment I create two functions: path_to_root and dependency_path, the main function is the former.  

* path_to_root(sentence): this function is used to parse the sentence and extract the dependency paths of each token.

  * Input: the sentence to parse
  * Output: dict with tokens as keys and list of dipendencies (from ROOT to token) as values
  * Implementation:
    * Process the sentence
    * For each token extracts its dependency path

```
def path_to_root(sentence):
  doc = process_sentence(sentence)
  d = {}
  for token in doc:
    d[token] = dependency_path(token)
  return d
```

* dependency_path(token): this function extracts the path, from the ROOT to the input token, of the dependencies

  * Input: a token from Doc spaCy object
  * Output: the dependency path from ROOT to the token as a list
  * Implementation:
    * starting from the dependency of the token itself, retrace the dependency tree until finding the ROOT (token.head == token)
    * return the reverse of the dependency list in order to output ROOT -> token path

```
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
```

## 2) Extract subtree of the dependents given a token  
For this point of the assignment there are two functions: get_subtree and get_subtrees, the last one is the main function which takens in input the sentence and output the subtree for each token.

*   get_subtree(token): given a token it ectracts its subtree
    * Input: a Token object from spaCy
    * Output: the list of the depentents from the token (its subtree)
    * Implementation: creation of a list containing the tokens inside its subtree
   
```
def get_subtree(token):
  return [dipendent for dipendent in token.subtree]
```

*   get_subtrees(sentence): given a sentence, it outputs the subtree of each token as a dict
    * Input: the sentence
    * Output: dict
      * keys: the tokens of the list
      * values: for each token the list of its dependents (the subtree of the token)
    * Implementation:
      *   Process the sentence
      *   For each token extracts its subtree using get_subtree
 
 ```
def get_subtrees(sentence):
  doc = process_sentence(sentence)
  d = {}
  for token in doc:
    d[token] = get_subtree(token)
  return d 
```

## 3) Check if a given list of tokens (segment of a sentence) forms a subtree

The function for this part takes in input the sentence and an ordered list of words, it checks if the list forms a subtree of the sentence, if yes it returns True, False otherwise.  
This function takes in account the order of the words, for example for the sentence ```"I watched a movie with Sisko."```, if the sequence is ```["a", "movie", "with", "Sisko"]``` it returns ```True```, 
if it is ```["movie", "a", "with", "Sisko"]``` it returns ```False```.  
If one want to not take in account the order, we can use sets, like ```if(set([t.text for t in subtree]) == set(sequence)):```, in this case we are comparing the sets without duplicates.
Here a note is a must: using sets means that if a subtree has duplicated words but the input list not, the condition above will result True, 
in fact the created sets, which delete the duplicates, will be equal, but the input list does not contain the second word of the subtree. For this reason I wanted to check also the ordering and the duplicates.

*   check_subtree(sentence, subsequence)
    * Input: The sentence to parse and the ordered list of words to check
    * Output: True if the list forms a subtree of the sentence, False otherwise
    * Implementation:
      * Extract every subtree of the sentence using get_subtrees
      * Compare the input sequence with each subtree, if one matches the input sequence return True, False otherwise

```
def check_subtree2(sentence, sequence):
  subtrees = get_subtrees(sentence)
  for token in subtrees:
    subtree = subtrees[token]
    if(len(subtree) == len(sequence)):
      if([t.text for t in subtree] == sequence):
        return True
  return False
```

## 4) Identify head of a span, given its tokens

This function takes in input a span as a string (not necessary a meaningful sentence), for instance "I man world" and it returns the root of this span, which in this case is man. This function assumes that the input span is one, namely not multiple spans.

* head_span(span)
  * Input: a string containing a span of words
  * Output: the word which is the head of the span
  * Implementation: 
    * parse the span to have a Doc object
    * from the Doc object, take the Span of the entire sequence and return its root

```
def head_span(span):
  doc = process_sentence(span)
  return doc[:].root.text
```

There is another way which is to use ```doc.sents``` to extract all the spans of the sequence (for example if we have multiple sentences), in fact ```.sents``` returns an object containing 
the spans of the input document, then using ```doc.sents[0].root.text``` we can extract the root of the first span (which is the root of the input span).

## 5) Extract sentence subject, direct object and indirect object spans

This point has the goal to extract the subject, direct object and indirect object of a sentence.  
Since the objects are specified (direct and indirect) but not the subject I decided to extract all the types of subject, which spaCy provides, in a given sentence. In fact in sentence we could have 
a passive nominal subject, so returning just the nominal subject is not enough. For instance in the sentence ```"Luca has been killed by a car"``` the nominal subject is ```None``` but the subject of the sentence is Luca which is the Nominal Subject Passive.  
Moreover I thought about returning just one subject (the most import one), but after some testing I reached the conclusion that there are too many possibilities to return just a subject. For example the sentence ```"There is a woman in the bus who is called Diana"``` 
has two types of subject: ```Nominal passive subject "who"``` and ```Expletive subject "There"```, so I decided to return both in two separated lists inside the list "Subject" of the output dict.  
The below function takes in input a sentence and it returns a dict with the ```Subject```, ```Direct Object``` and ```Indirect Object``` as keys and the corrisponding span as values (with a list of span for the subject as explained previously).  

* extract(sentence)
  * Input: the sentence
  * Output: a dict with keys:
    * ``Subject``
    * ``Direct object`` 
    * ``Indirect object``  
  and the corresponding lists of words as values, for the subject it returns a list for each subtree depending on the subject.

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
    
```   
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
```

## -) Execution
For the execution part I used some sentences (created by me or taken from the web).  
This part just calls the functions above to show the results.

## Optional part
For the optional part I tried to extract some new features but I could not find new ones, so I decided to change the model of the parser and to test it against the default one.  
I extended the default ```TransitionParser``` class in order to override the method ```train``` of the class; in this way I can change the model. I choose the MLP classifier which is a 
multi layer perceptron from the scikit learn library. Since the neural networks can have a fluctuation on the accuracy I trained and evaluated the model 10 times and I took the average.

```
class My_TransitionParser(TransitionParser):

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
```
For the dependency dataset I used the ```dependency_treebank``` from ```nltk.corpus``` as we did in the lab.  
I compared ```My_transition_parser``` with the default one using both ```arc-standard``` and ```arc-eager``` algorithms.  
The following table summarizes the results:

| Parser | arc-standard | arc-eager |
| :---: | :---: | :---: |
| TransitionParser | 0.82 | 0.82 |
| TransitionParser_MLP | 0.70 | 0.71 |

As we can see, using both algorithms, we have a better performance with the default model, which is a SVC (SVM) from scikit learn.  
Then I decided to try also the random forsets from scikit learn, after changing the class as previosuly I run the evaluation part with the same setting (an average of the accuracies after 10 runs), reaching these results:

| Parser | arc-standard | arc-eager |
| :---: | :---: | :---: |
| TransitionParser | 0.82 | 0.82 |
| TransitionParser_RF | 0.77 | 0.78 |

Even this time the default model is performing better, however the random forests are performing better than the MLP.


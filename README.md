# NLU_assignments
## Assignment 1
The repo contains a colab notebook with the code for the assignment  

### 1) Extract a path of dependency relations from the ROOT to a token

For the first point of the assignment I create two functions: path_to_root and dependency_path, the main function is the former.  

* path_to_root(sentence)

  * Input: the sentence to parse
  * Output: dict with tokens as keys and list of dipendencies (from ROOT to token) as values
  * Implementation:
    * Process the sentence
    * For each token extracts its dependency path

* dependency_path(token)

  * Input: a token from Doc spaCy object
  * Output: the dependency path from ROOT to the token as a list
  * Implementation:
    * starting from the dependency of the token itself, retrace the dependency tree until finding the ROOT (token.head == token)
    * return the reverse of the dependency list in order to output the ROOT -> token path

### 2) Extract subtree of the dependents given a token

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
 
### 3) Check if a given list of tokens (segment of a sentence) forms a subtree
*   check_subtree(sentence, subsequence)
    * Input: The sentence to parse and the ordered list of words to check
    * Output: True if the list forms a subtree of the sentence, False otherwise
    * Implementation:
      * Extract every subtree of the sentence using get_subtrees
      * Compare the input sequence with each subtree, if one matches the input sequence return True, False otherwise
### 4) Identify head of a span, given its tokens
* head_span(span)
  * Input: a string containing a span of words
  * Output: the head of the span
  * Implementation: 
    * parse the span to have a Doc object
    * from the Doc object, take the Span of the entire sequence and return its root

### 5) Extract sentence subject, direct object and indirect object spans
* extract(sentence)
  * Input: the sentence
  * Output: a dict with keys:
    * ``Nominal subject``
    * ``Nominal subject passive``
    * ``Clausal subject``
    * ``Clausal subject passive``
    * ``Expletive subject``
    * ``Direct object`` 
    * ``Indirect object``  
  and the corresponding subtrees as values

  * Implementation: 
    * For each token of the Doc object check if the token is:
      * ``nsubj`` which stand for Nominal Subject
      * ``nsubjpass`` which stand for Nominal Subject Passive
      * ``csubj`` which stand for Clausal Subject
      * ``csubjpass`` which stand for Clausal Subject Passive
      * ``expl`` which stand for Expletive Subject
      * ``dobj`` which stand for Direct Object
      * ``dative`` which stand for Indirect Object
    * For each of these tokens save its subtrees

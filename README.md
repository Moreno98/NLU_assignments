# NLU_assignments

* This repo contains two folders for the assignments of the NLU course.  
* In each folder there is a readme.md file which is the report for that particular assignment.  
* The code has been done using colab notebooks which are linked in the reports.
* The data folder contains the conll2003 dataset and the conll.py script, both provided by the Professor.

# Assignment 1 
## Requirements
The requirements of this assignment are:
* spaCy library
* en_core_web_sm pipeline from spaCy
* nltk
* sklearn

## How to run

Install the requirements in your system (if not present):
```
pip3 install -U spacy
python3 -m spacy download en_core_web_sm
pip3 install -U nltk
pip3 install -U scikit-learn
```
> Note: the optional part of the assignment can take some time due to the training, you may want to comment that part, in any case I recommend you to run the colab notebook directly.  

To run the code:
```
git clone https://github.com/Moreno98/NLU_assignments
cd NLU_assignments/Assignment_1 
python3 assignment_1.py
```
# Assignment 2
## Requirements (same as assignment 1)
The requirements of this assignment are:
* spaCy library
* en_core_web_sm pipeline from spaCy
* nltk
* sklearn

## How to run

Install the requirements in your system (if not present):
```
pip3 install -U spacy
python3 -m spacy download en_core_web_sm
pip3 install -U nltk
pip3 install -U scikit-learn
```
> Note: I suggest (as in the first assignment) to run the code directly on colab.  

To run the code:
```
git clone https://github.com/Moreno98/NLU_assignments
cd NLU_assignments/Assignment_2 
python3 assignment_2.py
```

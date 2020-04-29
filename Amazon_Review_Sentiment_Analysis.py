#!/usr/bin/env python
# coding: utf-8

# In[61]:


pip install autocorrect


# In[82]:


import numpy as np
import pandas as pd
import seaborn as sns
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from IPython.display import display
import timeit
from collections import defaultdict
import math
import random
from matplotlib import pyplot as plt
import matplotlib.dates as md
get_ipython().run_line_magic('matplotlib', 'inline')
import operator 

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import sentiment
#from autocorrect import spell 
import pandas as pd 
from urllib import request
from nltk.tokenize import WordPunctTokenizer
#download(autpcorrect)
import sys
get_ipython().system('{sys.executable} -m pip install autorrect')


# In[ ]:





# data = pd.read_csv("Python Scripts/Amazon_Unlocked_Mobile.csv", delimiter = ",")

# In[27]:


data.head()


# In[28]:


data.describe()


# In[29]:


data.count()


# In[6]:


# Check if there is any null value
data.isnull().values.any()


# In[7]:


# Count the number of rows with the null values
data.isnull().values.sum()


# In[14]:


"""Pre- Processing : 
#We will apply tokenization to remove stopwords, converting text to lower case, removing stop words, 
#removing non-alphaneumeric characters, synonyms, negation, POS tagging.  

Tokenization: 
Stop-words:
POS Tagging:
Stemming:
"""


# In[38]:


reviews = data['Reviews']
reviews.head()


# In[48]:


products = data['Brand Name']
nltk.download('stopwords')
nltk.download('punkt')


# In[83]:


def get_tokens(data, stem = False, negation = False):
    stemmer = PorterStemmer()
    # We filter out stop words before the processing dataset for analysis
    stop = set(stopwords.words('english'))
    reviews = []    
    i = 1
    
    for review in data["Reviews"]:
        tokenized_review = []      

        # convert the reviews to lower case
        review = str(review).lower() 
        
        """In Tokenizatin, we remove the stopwords, perform stemming of words, 
        case-folding, removing characters that are not alphanumeric and breaking at 
        whitespace. Remove every character except A-Z, a-z,space 
        and punctuation """
        
        review = re.sub(r'[^A-Za-z /.]','',review) 
        
        # Negation needs punctuation separated by white space.
        review = review.replace(".", " .")   
        
        # Breaking the text into tokens
        tokens = word_tokenize(review)
        
        for token in tokens:
        # Remove single characters and stop words
            if (len(token)>1 or token == ".") and token not in stop: 
                if stem:
                    tokenized_review.append(stemmer.stem(get_synonym(token)))            
                else:
                    tokenized_review.append(get_synonym(token))
        
        if negation:
            tokenized_review = sentiment.util.mark_negation(tokenized_review)   
        
        # Now we can get rid of punctuation and also let's fix some spellings:
        tokenized_review = [correction(x) for x in tokenized_review if x != "." ]
           
        if i%100 == 0:
            print('progress: ', (i/len(data["Reviews"]))*100, "%")
        i = i + 1
        
    return reviews


def part_of_speech_tagging(tokenized_reviews):
    tokenized_pos = []
    
    for review in tokenized_reviews:
        tokenized_pos.append(nltk.pos_tag(review))
    
    return tokenized_pos
        
    
def get_frequency_of_words(tokens):    
    term_freqs = defaultdict(int)    
    
    for token in tokens:
        term_freqs[token] += 1 
            
    return term_freqs

"""A DTM is basically a matrix, with documents designated by rows and
words by columns, that the elements are the counts or the weights (usually by tf-idf). 
"""
def get_term_document_matrix(tokenized_reviews):
    tdm = []
    
    for tokens in tokenized_reviews:
        tdm.append(get_frequency(tokens))
    
    return tdm

def normalize_term_document_matrix(tdm):    
    tdm_normalized = []
        
    for review in tdm:
        den = 0
        review_normalized = defaultdict(int)
        
        for k,v in review.items():
            den += v**2
        den = math.sqrt(den)
    
        for k,v in review.items():
            review_normalized[k] = v/den
        
        tdm_normalized.append(review_normalized)
        
    return tdm_normalized

def get_all_terms(tokenized_reviews):
    all_terms = []
    
    for tokens in tokenized_reviews:
        for token in tokens:
            all_terms.append(token)
            
    return(set(all_terms))
    
def get_all_terms_dft(tokenized_reviews, all_terms):
    terms_dft = defaultdict(int)  
    
    for term in all_terms: 
        for review in tokenized_reviews:
            if term in review:
                terms_dft[term] += 1
                
    return terms_dft

"""Another normalized TDM was constructed this time using TF*IDF weightings for each product name term.
Its purpose was to determine which potential terms could be considered as standardized product names. 
The higher the IDF value the more important to be a potential part of the standardized name
since the most commons words such as “unlocked”, “black” or “dual-core” should be avoided 
as they have low IDF scores"""


def get_tf_idf_transform(tokenized_reviews, tdm, n_reviews):
    tf_idf = []        
    all_terms = get_all_terms(tokenized_reviews)    
    terms_dft = get_all_terms_dft(tokenized_reviews, all_terms)
    
    for review in tdm:
        review_tf_idf = defaultdict(int)
        for k,v in review.items():
            review_tf_idf[k] = v * math.log(n_reviews / terms_dft[k], 2)
        
        tf_idf.append(review_tf_idf)     
    
    return tf_idf

"""defaultdict: is dictionary in python which is an unordered collection of data values that are used to store data
values like a map. Unlike other Data Types that hold only single value as an element,
the Dictionary holds key:value pair. In Dictionary, the key must be unique and immutable."""

def get_idf_transform(tokenized_reviews, tdm, n_reviews):
    idf = []    
    terms_dft = defaultdict(int)    
    
    all_terms = get_all_terms(tokenized_reviews)
    
    for term in all_terms: 
        for review in tokenized_reviews:
            if term in review:
                terms_dft[term] += 1
    
    for review in tdm:
        review_idf = defaultdict(int)
        for k,v in review.items():
            review_idf[k] = math.log(n_reviews / terms_dft[k], 2)
        
        idf.append(review_idf)     
    return idf


def correction(x):
    ok_words = ["microsd"]
    
    if x.find("_NEG") == -1 and x not in ok_words: # Don't correct if they are negated words or exceptions
        return x
    else:
        return x

def get_synonym(word):
    synonyms = [["camera","video", "display"], 
               ["phone", "cellphone", "smartphone", "phones"],
               ["setting", "settings"],
               ["feature", "features"],
               ["pictures", "photos"],
               ["speakers", "speaker"]]
    synonyms_parent = ["camera", "phone", "settings", "features", "photos", "speakers"]
    
    for i in range(len(synonyms)):
        if word in synonyms[i]:
            return synonyms_parent[i]
    
    return word


def get_similarity_matrix(similarity, tokenized_reviews):
    similarity_matrix = []
    all_terms = get_all_terms(tokenized_reviews)
    
    for review in similarity:
        similarity_matrix_row = []
        
        for term in all_terms:
            similarity_matrix_row.append(review[term])  
            similarity_matrix.append(similarity_matrix_row)      
        
        return similarity_matrix


"""
A vector space model was created based on a normalized (by euclidean distance) Term-Document-Matrix 
via bags-of-words for both product names as well as reviews in preparation for clustering purposes.
For the first to standardize product names and for the latter to filter reviews.
"""

def get_synonym(word):
    synonyms = [["camera","video", "display"], 
                ["phone", "cellphone", "smartphone", "phones"],
               ["setting", "settings"],
               ["feature", "features"],
               ["pictures", "photos"],
               ["speakers", "speaker"]]
    synonyms_parent = ["camera", "phone", "settings", "features", "photos", "speakers"]
    
    for i in range(len(synonyms)):
        if word in synonyms[i]:
            return synonyms_parent[i]
    
    return word


# In[ ]:


lookup_review = 1
for val in df[df.id_new_col == lookup_review]["Review"]: print(val)
display(tokenized_reviews[lookup_review])
display(tokenized_pos[lookup_review])
display(tdm[lookup_review])
display(tf_idf[lookup_review])


# In[84]:


tic=timeit.default_timer()

tokenized_reviews = get_tokens(data, stem = False, negation = False)
tokenized_pos = get_parts_of_speech(tokenized_reviews)
tdm = get_term_document_matrix(tokenized_reviews)
vsm = normalize_term_document_matrix(tdm)
tf_idf = get_tf_idf_transform(tokenized_reviews, tdm, n_reviews)
toc=timeit.default_timer()
print("minutes: ", (toc - tic)/60)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





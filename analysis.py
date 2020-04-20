# Cross-Industry Standard Process of Data Mining - (CRISP-DM)
# A. Bussiness Understanding                 
#     1. Formulate research/analysis question(s)
    
#            Question I
#                 Which demographic groups respond best to which offer type?
#                         Analyzing the response rate of each gender and age groups.
#             Question II
#                 Who is the typical Starbucks rewards mobile app user?
#                         Analysis of descriptive statistics of all variables in the dataset.
#             Question III
#                 Who will response to an offer?
#                         Build a model that predicts whether or not someone will respond to an offer.




# B. Data Understanding
#     1. Seek for relevant datasets
#     2. Read/Download relevant datasets
    
    
#   import modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer  

bcolor = sns.color_palette()[0]

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)
    
# C. Data preparation
#     1. Examine and understand the datasets

def exam_understand(dataset):
    """The exam_understand function print out the head, information (dtype, non-missing number, column name),
        shape and descriptive statistics for numerical features of dataset
    
    Args:
        dataset (DataFrame)
        
    Outputs:
        
    
    """
    
    sp = {"sep":"\n\n", "end":"\n\n"}
    
    print(dataset.head(),dataset.info(),dataset.shape,dataset.decribe(), **sp)
    
exam_understand(portfolio) 
exam_understand(profile) 
exam_understand(transcript) 
transcript['event'].value_counts()

#     2. Clean and join datasets
#     3. Features extraction and engineering
#     4. Exploratory and explanatory data analysis
#     5. Data visualization




# D. Modeling
#     1. Supervised learning
# E. Deployment
#     1. Summary report
#     2. Conclusion(s)

    
    





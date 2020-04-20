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

#     2a. Clean datasets
#     3. Features extraction and engineering

def clean_transact(datta):
    """The clean_transact function will do data cleaning and wrangling of the transaction data 
     
     Input:
         data (DataFrame) : The dataset for cleaning
         
         
     Output:
         
         clean_dat (DataFrame) : cleaned dataset
    
    
    """
    
    # make copy of the data
    data = datta.copy()
    
    # extract the values from the rewards if the exist on the value column's dictionary
    #  if don't exist replace with 99999
    data['offer_id_rewards'] = data['value'].apply(lambda x: list(x.values())[1] if len(list(x.values())) > 1 else 99999)
    

    # extract the 'offer ids','offer_id', and 'amount' keywords from the value column's dictionary and convert to string dtype
    data['offer_id_keys'] = data['value'].apply(lambda x: list(x.keys())[0]).astype(str)
    
    # extract the offer values of the value column's dictionary
    # these are the offer ids and rewards amounts respectively where the exist
    data['offer_id_values'] = data['value'].apply(lambda x: list(x.values())[0])
    
    
    # extract offer ids if there is an offer or replace with no_offer is there is not offer made to user
    data["offer_id"] = data[["offer_id_values", "offer_id_keys"]].apply(lambda x: x[0] if x[1] !="amount" else "no_offer" , axis=1)
    
    
    # extract the offer amount is there was an offer or replace with 9999 if not offer was made to user
    data["reward_amt"] = data[["offer_id_values", "offer_id_keys"]].apply(lambda x: x[0] if x[1] =="amount" else 9999 , axis=1)
    
    
    # drop columns not essential for further analysis
    data.drop(columns=["value", "offer_id_keys", "offer_id_keys"], inplace=True)
    
    return data


def profile_cleaner(dat):
    """The profile_cleaner function will do data cleaning and wrangling of the profile data 
     
     Input:
         dat (DataFrame) : The dataset for cleaning
         
         
     Output:
         
         profile2 (DataFrame) : cleaned dataset
         
    
    """
    
    # make copy of the data
    profile2 = dat.copy()
    
    # convert become_member_on column to datetime dtype
    profile2["membershipDate"] = pd.to_datetime(profile2["became_member_on"], format="%Y%m%d")
    
    # get the most recent member membership start date
    ## add 1 day to it, to get a reference date => refDate
    refDate = profile2.membershipDate.max().date() + dt.timedelta(days=1)
    
    # calculate days since joining, using the refDate
    profile2["membershipDays"] = profile2["membershipDate"].apply(lambda x : (refDate - x.date()).days)
    
    # categorise age to get agegroups
    #     - Elder Adults (over 70 years old)
    #     - Baby Boomers (Roughly 50 to 70 years old)
    #     - Generation X (Roughly 35 – 50 years old)
    #     - Millennials, or Generation Y (18 – 34 years old)
    #     - Generation Z, or iGeneration (Teens & younger)
    
    profile2["agegroups"] = pd.cut(profile2.age, bins=[18,34,50,70, profile2.age.max()],
                                   labels=[ "millennials", "genX", "babyboomer","elderlyadult"],
                                  right=False)
    
    
    # drop become_member_on column 
    profile2.drop(columns=["became_member_on"], inplace=True)
    
    return profile2
    
    
def clean_portfolio(datapp):
    """The clean_portfolio function will do data cleaning and wrangling of the portfolio data 
     
     Input:
         datapp (DataFrame) : The dataset for cleaning
         
         
     Output:
         
         vectdata (DataFrame) : cleaned dataset
    
    
    """
    
    # make copy of the data
    datap = datapp.copy()
    
    # get ids
    bbb = datap.id.tolist()
    
    # create offerType2
    datap['offerType2'] = datap[['offer_type', 'id']].apply(lambda x: x[0]+str(bbb.index(x[1])), axis=1)
    
    
    # initialize countervectorizer
    countVectorizer = CountVectorizer()
    
    # tranform channels column content list to text
    datap['ch'] = datap['channels'].apply(lambda x: " ".join(word for word in x))
    
    # Fit and  Transform the texts
    content_vectorized = countVectorizer.fit_transform(datap['ch'])

    # convert to array
    content_Varray = content_vectorized.toarray()


    # convert to dataframe
    content_df = pd.DataFrame(content_Varray, 
                         columns=countVectorizer.get_feature_names()).add_prefix('Channel_')

    # concat the dataframes
    vectdata = pd.concat([datap, content_df], axis=1)
    

    # drop channels and ch columns
    vectdata.drop(columns=['channels', 'ch'], inplace=True)
    
    return vectdata

#     2b. Join datasets
def data_merger(data1, data22, data33):
    """The data_merger function will merge the datasets
    
    Inputs:
        data1 (DataFrame)
        data22 (DataFrame)
        data33 (DataFrame)
        
    
    Output:
        data4 (DataFrame) : Merged dataset
        
    
    """
    
    # clean raw  datasets
    # clean transcripts dataset
    data1_clean = clean_transact(data1)
    
    # clean profile dataset
    data2 = profile_cleaner(data22)
    
    # clean portfolio dataset
    data3 = clean_portfolio(data33)
    
    # update the id column name to p_id
    # profile had a column named id
    # this will give error on merging the datasets
    data3.rename(columns = {'id':'p_id'}, inplace = True) 
    
    
    # merge cleaned trascript and profile datasets
    data4 = pd.merge(data1_clean, data2, how="outer", left_on ="person", right_on="id")
    
    # merge data4 with cleaned portfolio dataset
    data5 = pd.merge(data4, data3, how="outer", left_on ="offer_id", right_on="p_id")
    
    
    # drop duplicate columns
    data5.drop(columns=["offer_id_values", "id", "p_id"], inplace=True)
    
    return data5


clean_data = data_merger(transcript, profile, portfolio)
clean_data.head()
exam_understand(clean_data)
clean_data.agegroups.value_counts()

#     4. Exploratory and explanatory data analysis
#     5. Data visualization




# D. Modeling
#     1. Supervised learning
# E. Deployment
#     1. Summary report
#     2. Conclusion(s)

    
    





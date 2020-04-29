#!/usr/bin/env python
# coding: utf-8

# # Starbucks Capstone Challenge

# # Analysis PLan 
# 
# Cross-Industry Standard Process of Data Mining - (CRISP-DM)
# 
# A. Bussiness Understanding                 
#     1. Formulate research/analysis question(s)
# B. Data Understanding
#     1. Seek for relevant datasets
#     2. Read/Download relevant datasets
# C. Data preparation
#     1. Examine and understand the datasets
#     2. Clean and join datasets
#     3. Features extraction and engineering
#     4. Exploratory and explanatory data analysis
#     5. Data visualization
# D. Predictive modeling and answers to research question
#     1. Tackle reseach questions
#     2. Supervised learning
# E. Deployment
#     1. Summary report
#     2. Conclusion(s)

# # A. Bussiness Understanding                 
#     1. Formulate research/analysis question(s)
#     
#            Question I
#                 Who is the typical Starbucks rewards mobile app user?
#                         Analysis of descriptive statistics of all variables in the dataset.
#                         
#             Question II
#                  Which demographic group respond best to which offer type?
#                         Analyzing the response rate of each gender and age groups.
# 
#                         
#             Question III
#                 Who will response to an offer?
#                         Build machine learning model that predicts whether or not someone will respond to an offer.

# # B. Data Understanding
#     1. Seek for relevant datasets
#     2. Read/Download relevant datasets

# ### import modules and libraries


import numpy as np
import pandas as pd
import math
import json
import datetime as dt

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MaxAbsScaler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report

from eli5.sklearn import PermutationImportance
bcolor = sns.color_palette()[0]
blued = sns.color_palette("Blues_d")

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# # C. Data preparation
#     1. Examine and understand the datasets
#     2. Clean and join datasets
#     3. Features extraction and engineering
#     4. Exploratory and explanatory data analysis
#     5. Data visualization

# #### 1. Examine and understand the datasets

def exam_understand(dataset):
    """The exam_understand function print out the head, information (dtype, non-missing number, column name),
        shape and descriptive statistics for numerical features of dataset
    
    Args:
        dataset (DataFrame)
        
    Outputs:
        
    
    """
    
    sp = {"sep":"\n\n\n", "end":"\n\n\n"}
    
    print(dataset.describe(),  **sp)
    print(" ", **sp)
    print(dataset.info(), dataset.shape, **sp)


exam_understand(portfolio)
exam_understand(profile)
exam_understand(transcript)

print(portfolio.info())
print(portfolio.head())
print(profile.head())
print(transcript.head())
print(transcript['event'].value_counts())


# ## 2a. Clean
# ## 3. Features extraction and engineering

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
    # these are the offer ids and rewards amounts respectively where they exist
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
    
    # get income groups
    #     Lowest income less than 31,000
    #     Lower-middle income between 31,000  and 41,999
    #     Middle-income between 42,000 and 125,999
    #     Upper-middle income between 126,000 and 187,999
    #     Higher-income more than 188,000 
    
    # sorted bar chart on ordinal categories
    ordering_i = ["Lowest", "Lower-middle", "Middle-income","Upper-middle", "Higher-income"]
    ti = pd.api.types.CategoricalDtype(categories= ordering_i, ordered=True)

    profile2["incomegroups"] = pd.cut(profile2.income, bins=[0,30999,42000,126000, 188000, 1000000],
                                   labels=ordering_i, right=False).astype(ti)
                  
                  
    # categorise age to get agegroups
    #     - Elder Adults (over 70 years old)
    #     - Baby Boomers (Roughly 51 to 70 years old)
    #     - Generation X (Roughly 35 – 50 years old)
    #     - Millennials, or Generation Y (18 – 34 years old)
    #     - Generation Z, or iGeneration (Teens & younger)
                  
    ordering_a = [ "millennials", "genX", "babyboomer","elderlyadult"]
    ta = pd.api.types.CategoricalDtype(categories= ordering_a, ordered=True)
    
    profile2["agegroups"] = pd.cut(profile2.age, bins=[profile2.age.min(),35,51,71, profile2.age.max()+1],
                                   labels=ordering_a, right=False).astype(ta)
    
    
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
    # bbb = datap.id.tolist()
    # create offerType2
    # datap['offerType2'] = datap[['offer_type', 'id']].apply(lambda x: x[0]+str(bbb.index(x[1])), axis=1)
    datap['offer_id_short'] = [str(x).strip()[-4:] for x in datap['id']]
    datap['offerType2'] = datap[['offer_type', 'offer_id_short']].agg('-'.join, axis=1)
    
    
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
    
    #     # clean portfolio dataset
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
print(clean_data.head())
exam_understand(clean_data)
print(clean_data.shape)
print(clean_data.agegroups.value_counts())


# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record

# # 3. Exploratory and explanatory data analysis

# ## 3a.  Missing data analysis

# analyse missing gender
missing_gender = clean_data[clean_data['gender'].isna()]
print(missing_gender.event.value_counts())
print(missing_gender.head())
print(missing_gender.shape, missing_gender.person.nunique())


# analyse missing income
missing_income = clean_data[clean_data['income'].isna()]
print(missing_income.event.value_counts())
print(missing_income.shape, missing_income.person.nunique())

# analyse missing gender among those missing income
missing_income_gender = missing_income[missing_income['gender'].isna()]
print(missing_income_gender.event.value_counts())

missing_income_nodup = missing_income.drop_duplicates(subset=['person'])
print(missing_income_nodup.shape)
print(missing_income_nodup.event.value_counts())

# analyse none missing gender among those missing income
missing_income_gender_npg = missing_income[~missing_income['gender'].isna()]
print(missing_income_gender_npg.shape)

# analyse age amongst those missing gender and income
print(missing_income.age.describe())


nullreward = clean_data[clean_data.reward.isnull()]
print(nullreward.head())
print(nullreward.info())
print(nullreward.event.value_counts())


# ## insights
# - From the analysis above, members missing gender data also had missing income data.
# - Looking carefully at the age for members with missing gender and income data, the mean, max, median, 25% and 75% percentiles are the same, 118 years with standard deviation of 0.0. Therefore,
#      - Apart from the age of 118 years being questionable; all 2172 user had the same age, 118 years.
#      - Also, many of those missing gender and income data were transaction (1303) with few receiving an offer (872) but none of them ever viewed or completed the offer.
#      - I can conclude that the missing entries in the data is not random.
#      - I can conclude that those missing income, age and gender data wouldn't contribute much information toward our analysis.
# - Missing reward, difficulty, duration, offer_type, offerType2, Channel_email, Channel_mobile, Channel_social, Channel_web are associated with features of offer. Therefore, they are missing if user didn't receive an offer. Also, for user with missing data on these variable, the event recorded is only transaction, confirming that those user didn't receive any offer.
#         
# - Based on the above analysis, it seem reasonable to drop missing gender and age data from further analysis

# ## 3b. Working with complete, non-missing dataset


data = clean_data[~clean_data['gender'].isna()]
print(data.event.value_counts())

print(data.info())
print(data.person.nunique())


data = data.fillna({'reward': 99 ,'difficulty':99, 'offer_type':'No_offer', 
                    'offerType2':'No_offer','duration':0, 'Channel_web':0,
                    'Channel_email':0, 'Channel_mobile':0, 'Channel_social':0})

# data = data.fillna(0, inplace = True)

print(data.info())
print(data.head())

print(data[data.event == "transaction"].tail())
print(data.shape, data.person.nunique())
print(data['gender'].value_counts())

def col_encoder(data, x):
    """The col_encoder function maps the long string column values to simple number string
    
    Input:
        data (DataFrame): the dataframe to encode
        x (string): column to recode
        
        
    Output:
        person_encoded (list) : list of encoded number string
        
    """
    
    # instantiate dictionary lookup 
    codedict = dict()
    
    # instantiate numeric counter
    counter = 1
    
    # instantiate a holding list
    col_encoded = []
    
    
    # loop through the data column
    for val in data[x]:
        if val not in codedict:
            codedict[val] = counter
            col_encoded.append(codedict[val])
            counter+=1
        else:
            col_encoded.append(codedict[val]) 
            
       
    return col_encoded

person_encoded = col_encoder(data, "person")
offer_encoded = col_encoder(data, "offer_id")


data.insert(1, 'user_id', person_encoded)
data.insert(3 , 'offer_id2', offer_encoded)

# show header
print(data.head())  


# calcuate number of unique members
print(data.shape, data.person.nunique())
# calcuate events
print(data.event.value_counts())

# Drop duplicates and get one record per user for further analysis
data2 = data.drop_duplicates(subset=['person'])
print(data2.shape)


# # 5. Data visualization
# # Get listing percentage for gender
# Analyzing the distribution of members according to gender.
genderclass_p = data2["gender"].value_counts(normalize=True) * 100
genderclass_count = data2["gender"].value_counts() 
print("\n\n\n\nlisting each gender Percentages :", genderclass_p, "\n\n\n\nlisting each gender Raw counts" , 
      genderclass_count, sep="\n" )


# Plot the gender class distribution
plt.figure(figsize=[10,8])
sns.barplot(x= genderclass_p.index, y = genderclass_p.values, color = bcolor, edgecolor="#2b2b28")
# plt.bar( ddf['index'], ddf.gender,  edgecolor="#2b2b28")
plt.xlabel("Gender")
plt.xticks([0,1,2],['Male','Female', 'Other'])
plt.ylabel("Percentage of members")
plt.title("Starbucks Membership by Gender")
plt.tight_layout()
plt.show()

# # Get listing percentage of events
# Analyzing the distribution of events.
event_p = data["event"].value_counts(normalize=True) * 100
event_counts = data["event"].value_counts() 
print("\n\n\n\nlisting each event Percentages :", event_p, "\n\n\n\nlisting each event Raw counts" , event_counts, sep="\n" )

  
# Plot the event distribution
plt.figure(figsize=[10,8])
sns.barplot(x = event_p.index, y= event_p.values,  color = bcolor,   edgecolor="#2b2b28")
plt.xlabel("Event")
plt.ylabel("Percentage of members")
plt.title("Starbucks Event Distribution")
plt.tight_layout()
plt.show()
    
# get data for offer events
devent = data[data['event']!="transaction"].reset_index(drop=True)

# sorted bar chart on ordinal categories
# this method requires pandas v0.21 or later
ordering_ = ['offer received', 'offer viewed', 'offer completed']
t = pd.CategoricalDtype(categories= ordering_, ordered=True)

devent["event"] = devent["event"].astype(t)
print(devent.event.value_counts())

# # Get listing percentage of offer events
# Analyzing the distribution of offer events.
offerevents_p = devent["event"].value_counts(normalize=True) * 100
offerevents_count = devent["event"].value_counts() 
print("\n\n\n\nOffer events Percentages :", offerevents_p, "\n\n\n\nOffer events Raw counts" , 
      offerevents_count, sep="\n" )
 
# Plot the distribution of offer events
plt.figure(figsize=[10,8])
sns.barplot(x = offerevents_p.index, y= offerevents_p.values,  color = bcolor,   edgecolor="#2b2b28")
plt.xlabel("Event")
plt.ylabel("Percentage of members")
plt.title("Starbucks Offer Events Distribution")
plt.tight_layout()
plt.show()

# # get the counts of offer events across  and within gender
hh = pd.crosstab(data2["event"], data2["gender"], normalize="index", margins = True).fillna(0) * 100
hh2 = pd.crosstab(data2["event"], data2["gender"],  margins = True).fillna(0)
hht = pd.crosstab(data2["event"], data2["gender"], normalize="all").fillna(0) * 100
print("\n\n\n\noffer events across gender Percentages :", hh, " \n\n\n\noffer events across gender Raw counts" , hh2,
        " \n\n\n\nroom_types per bourough Raw counts", hht , sep="\n")

# # Plot the distribution of offer within gender
hh.plot.bar(stacked=True, cmap='Blues_r', figsize=(10,7), edgecolor=["#2b2b28", "#2b2b28", "#2b2b28"])
plt.xticks(rotation=0)
plt.xlabel("offer events within gender")
plt.ylabel("Percent")
plt.xticks([0,1,2],['offer received','No offer', 'All Users'])
plt.title("  Starbucks Offer Events Distribution")
plt.tight_layout()
plt.show()

# RdBu_r, PuBu_r
# # Plot the distribution of offer across gender
hht.plot.bar(stacked=True, cmap='Blues', figsize=(10,7), edgecolor=["#2b2b28", "#2b2b28", "#2b2b28"])
plt.xticks(rotation=0)
plt.xlabel("offer events across gender")
plt.ylabel("Percent")
plt.xticks([0,1],['offer received','No offer'])
plt.title("  Starbucks Offer Events Distribution ")
plt.tight_layout()
plt.show()

print(data2.head())
print(data2.describe())
print(data2.shape)

# Create a histogram of income
plt.figure(figsize=[10,8])
plt.hist(data2['income'], bins=10)
plt.title('Distribution of Income')
plt.ylabel("Count")
plt.xlabel('Income')
plt.show()

# Create a histogram of income among genders or only Males and females
plt.figure(figsize=[11,8])
plt.hist('income', data=data2[data2['gender'] == 'M'], alpha=0.5, label='Male', bins=10)
plt.hist('income', data=data2[data2['gender'] == 'F'], alpha=0.5, label='Female', bins=10)
plt.title('Distribution of income by Gender')
plt.xlabel('income')
plt.ylabel("Count")
plt.legend()
plt.show()

# Create a boxplot of income among genders 
plt.figure(figsize=[11,8])
sns.boxplot(x="gender", y="income", data=data2, color = bcolor)
plt.title('Distribution of income by Gender')
plt.xticks([0,1,2],['Male','Female', 'Other'])
plt.ylabel('income')
plt.xlabel("Gender")
plt.legend()
plt.show()

# Create a histogram of ages
plt.figure(figsize=[10,8])
plt.hist(data2['age'], bins=20)
plt.title('Distribution of Age')
plt.ylabel("Count")
plt.xlabel('Age')
plt.show()

# Create a histogram of ages among genders for only Males and females
plt.figure(figsize=[11,8])
plt.hist('age', data=data2[data2['gender'] == 'M'], alpha=0.5, label='Male', bins=10)
plt.hist('age', data=data2[data2['gender'] == 'F'], alpha=0.5, label='Female', bins=10)
plt.title('Distribution of Age by Gender')
plt.xlabel('Age')
plt.ylabel("Count")
plt.legend()
plt.show()

# Create a boxplot of ages among genders 
plt.figure(figsize=[11,8])
sns.boxplot(x="gender", y="age", data=data2, color = bcolor)
plt.title('Distribution of Age by Gender')
plt.xticks([0,1,2],['Male','Female', 'Other'])
plt.ylabel('Age')
plt.xlabel("Gender")
plt.legend()
plt.show()

# Create a histogram of membership Days
plt.figure(figsize=[10,8])
plt.hist(data2['membershipDays'], bins=20)
plt.title('Distribution of Membership Days')
plt.ylabel("Number of Users")
plt.xlabel('Number of Days')
plt.show()

# Create a histogram of membership Days among genders or only Males and females
plt.figure(figsize=[11,8])
plt.hist('membershipDays', data=data2[data2['gender'] == 'M'], alpha=0.5, label='Male', bins=15)
plt.hist('membershipDays', data=data2[data2['gender'] == 'F'], alpha=0.5, label='Female', bins=15)
plt.title('Distribution of Membership Days')
plt.ylabel("Number of Users")
plt.xlabel('Number of Days')
plt.legend()
plt.show()

# Create a boxplot of membership Days among genders 
plt.figure(figsize=[11,8])
sns.boxplot(x="gender", y="membershipDays", data=data2, color = bcolor)
plt.title('Distribution of Membership Days')
plt.xticks([0,1,2],['Male','Female', 'Other'])
plt.title('Distribution of Membership Days')
plt.ylabel("Number of Users")
plt.xlabel('Number of Days')
plt.legend()
plt.show()

plt.figure(figsize=[11,8])
sns.heatmap(data2[["age", "income", "membershipDays"]].corr(), annot=True, cbar=False)
plt.title('Correlation Heatmap - All Users')
plt.yticks(rotation=0)
plt.show()

men = data2[data2['gender'] == 'M']
ladies = data2[data2['gender'] == 'F']

plt.figure(figsize=[11,8])
sns.heatmap(ladies[["age", "income", "membershipDays"]].corr(), annot=True, cbar=False)
plt.title('Correlation Heatmap - Female Users')
plt.yticks(rotation=0)
plt.show()

plt.figure(figsize=[11,8])
sns.heatmap(men[["age", "income", "membershipDays"]].corr(), annot=True, cbar=False)
plt.title('Correlation Heatmap - Male Users')
plt.yticks(rotation=0)
plt.show()


# # D. Predictive modeling and answers to research question
# ## 1. Tackle reseach questions
 
# # Question I
# ## Who is the typical Starbucks rewards mobile app user?
# - Analysis of descriptive statistics of all variables in the dataset.

# create dataframe with one record per user
single_users = data.drop_duplicates(subset=['person'], keep="last")
print(single_users.person.nunique(), single_users.shape)


# ## Analyze gender:

# find the highest count gender
# the gender with most users is typical
print(single_users.gender.value_counts(normalize=True) * 100)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,8), sharey=True)
plt.subplot(121)

plt.hist('age', data=data2[data2['gender'] == 'M'], alpha=0.5, label='Male', bins=10)
plt.hist('age', data=data2[data2['gender'] == 'F'], alpha=0.5, label='Female', bins=10)
plt.title('Distribution of Age by Gender')
plt.xlabel('Age')
plt.ylabel("Count")
plt.legend()

plt.subplot(122)
# Create a histogram of income among genders or only Males and females
plt.hist('income', data=data2[data2['gender'] == 'M'], alpha=0.5, label='Male', bins=10)
plt.hist('income', data=data2[data2['gender'] == 'F'], alpha=0.5, label='Female', bins=10)
plt.title('Distribution of income by Gender')
plt.xlabel('income')
plt.legend()
plt.show()

# ## Analyze age
# visual Normality check
def plot_dnorm(arr):
    plt.figure(figsize=[10,8])
    mean = np.mean(arr)
    variance = np.var(arr)
    sigma = np.sqrt(variance)
    
    x = np.linspace(min(arr), max(arr), len(arr))
    plt.hist(arr, normed=True)
    plt.xlim((min(arr)-mean/10, max(arr)+mean/10))


    plt.plot(x, mlab.normpdf(x, mean, sigma))


    plt.show()

# Distribution of users age is fairly normal, although the mean and median relatively close
print(single_users.age.describe())
print(single_users.agegroups.value_counts(normalize=True) * 100)

# Density Plot and Histogram of all age
plt.figure(figsize=[10,8])
single_users['age'].plot.hist(density=True, color = 'darkblue',
             edgecolor='black',linewidth = 4)
single_users['age'].plot.kde(legend=False, title="Histogram of all age")
plt.show()

data_age = single_users.query('age >= age.mean()-2 and age <= age.median()+2')
print(data_age.agegroups.value_counts())

# ## Analyze income
# Distribution of users income is poorly normal, although the mean and median relatively close
print(single_users.income.describe())
bb = single_users.incomegroups.value_counts(normalize=True) * 100
print(bb)


# # # plot the room types
plt.figure(figsize=[10,8])
plt.bar(bb.index[0:3], bb.values[0:3], edgecolor="#2b2b28")
plt.xlabel("Income Classification")
plt.xticks(rotation=40, ha='right')
plt.ylabel("Percentage of Total")
plt.title("Starbuck mobile app users")
plt.tight_layout()
plt.show()


bba = single_users.agegroups.value_counts(normalize=True) * 100
# bba =  bba.reset_index()
print(bba)


# # # plot the room types
plt.figure(figsize=[10,8])
plt.bar(bba.index, bba.values, edgecolor="#2b2b28")
plt.xlabel("Age Classification")
plt.xticks(rotation=40, ha='right')
plt.ylabel("Percentage of Total")
plt.title("Starbuck mobile app users")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,8), sharey=True)
plt.subplot(121)
sns.barplot(x = bba.index, y= bba.values,  color = bcolor,   edgecolor=["#2b2b28","#2b2b28","#2b2b28","#2b2b28"] , linewidth = 2)
plt.xlabel(" ")
plt.xticks(rotation=40, ha='right')
plt.ylabel("Percentage of Total")
plt.title("Age Classification")
plt.tight_layout()
bbc = bb[:3]
plt.subplot(122)

sns.barplot(x = bbc.index, y= bbc.values,  color = bcolor,   edgecolor=["#2b2b28","#2b2b28","#2b2b28","#2b2b28"] , linewidth = 2)
plt.xlabel(" ")
plt.xticks(rotation=40, ha='right')
plt.title("Income Classification")
plt.tight_layout()
plt.show()


# Density Plot and Histogram of all income
plt.figure(figsize=[10,8])
single_users['income'].plot.hist(density=True, color = 'darkblue',
             edgecolor='black',linewidth = 4)
single_users['income'].plot.kde(legend=False, title="Histogram of all income")



plot_dnorm(single_users["income"])
data_income = single_users.query('income >= income.median()-2 and income <= income.mean()+2')
print(data_income.incomegroups.value_counts())


#  # Question II
#  ## Which demographic group respond best to which offer type?
#   - Analyzing the response rate of each gender and age groups.

oevent = ['offer received', 'offer viewed', 'offer completed']
ooffer = ['informational', 'discount', 'bogo']
toevent = pd.api.types.CategoricalDtype(categories= oevent, ordered=True)
tooffer = pd.api.types.CategoricalDtype(categories= ooffer, ordered=True)

# keep only users that received an offer
dada = data[data['event'] != "transaction"]
data8 = dada.drop_duplicates(subset=['person'], keep="last")
data8['eventFinal'] = data8['event'].astype(toevent).apply(lambda x: "No response" if x =="offer received" else x)
data8['offer_type'] = data8['offer_type'].astype(tooffer)

# check total number of users and those with complete dataset the received an offer
clean_data.person.nunique(), data8.shape
offf2 = data8.eventFinal.value_counts(normalize=True)*100
print(offf2)

plt.figure(figsize=[10,8])
sns.barplot(x = offf2.index, y= offf2.values,  color = bcolor,   edgecolor=["#2b2b28","#2b2b28","#2b2b28","#2b2b28"] , linewidth = 2)
plt.xlabel(" ")
plt.title("Marketing Offer among users")
plt.ylabel("Percentage")
plt.tight_layout()
plt.show()

print(data8.offer_type.value_counts())

offf = data8.offer_type.value_counts(normalize=True)*100
# offf = offf.reset_index()
print(offf)


plt.figure(figsize=[10,8])
sns.barplot(x = offf.index, y= offf.values,  color = bcolor,   edgecolor=["#2b2b28","#2b2b28","#2b2b28","#2b2b28"] , linewidth = 2)
plt.xlabel(" ")
plt.title("Marketing Offer among users")
plt.ylabel("Percentage")
plt.tight_layout()
plt.show()


# Plot the distribution
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,8), sharey=True)

sns.barplot(x = offf.index, y= offf.values,  color = bcolor,   edgecolor=["#2b2b28","#2b2b28","#2b2b28","#2b2b28"] , linewidth = 2, ax=ax1)
plt.xlabel(" ")
ax1.set_title("Marketing Offers to Users")
ax1.set_ylabel("Percentage")
plt.tight_layout()

sns.barplot(x = offf2.index, y= offf2.values,  color = bcolor,   edgecolor=["#2b2b28","#2b2b28","#2b2b28","#2b2b28"] , linewidth = 2, ax=ax2)
plt.xlabel(" ")
plt.title("Users Response to Marketing offer")
# plt.ylabel("Percentage")
plt.tight_layout()
plt.show()


offerevents2 = pd.crosstab(data8["offer_type"], data8["eventFinal"], normalize="index",  margins = True).fillna(0)*100
offerevents2 = offerevents2.reset_index()
print(offerevents2)


offerevents = pd.crosstab(data8["offer_type"], data8["eventFinal"], normalize="columns",  margins = True).fillna(0)*100
offerevents = offerevents.reset_index()
print(offerevents)

# Plot the distribution
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,8), sharey=True)
offerevents.plot(x='offer_type' , y=offerevents.columns.tolist()[1:4],  kind="bar", cmap="Blues", edgecolor=["#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28"], ax=ax1)

ax1.set_xlabel(" ")
ax1.set_ylabel("Response Rate")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.set_title("Offer type response among users")
ax1.legend(title="Users Response")


offerevents2.plot(x='offer_type' , y=offerevents2.columns.tolist()[1:4],  kind="bar", cmap="Blues", edgecolor=["#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28"], ax=ax2)

plt.xlabel(" ")
plt.ylabel("Response Rate")
plt.xticks(rotation=0)
plt.title("Users response to offer")
plt.legend(title="Users Response")
plt.tight_layout()
plt.show()

hh22 = pd.crosstab(data8["event"], data8["eventFinal"],  margins = True).fillna(0)
print(hh22)

print(data8.eventFinal.value_counts())
print(data8.groupby(['event']).groups.keys())
print(data8['event'].value_counts())

pd.crosstab(data8["offer_type"], data8["gender"],  margins = True).fillna(0)
pd.crosstab(data8["eventFinal"], data8["gender"], normalize="columns", margins = True).fillna(0) * 100
pd.crosstab(data8["eventFinal"], data8["gender"], normalize="index", margins = True).fillna(0) * 100

pd.crosstab(data8["offer_type"], data8["gender"], normalize="index", margins = True).fillna(0) * 100
pd.crosstab(data8["offer_type"], data8["gender"], normalize="columns", margins = True).fillna(0) * 100
yy9 = pd.crosstab(data8["eventFinal"], data8["incomegroups"], normalize="columns", margins = True).fillna(0) * 100
yy9 = yy9.reset_index()


ordering_combine = ['informational-No response','informational-offer viewed', 
                    'discount-No response','discount-offer viewed','discount-offer completed',
                    'bogo-No response', 'bogo-offer viewed', 'bogo-offer completed']
t_combine = pd.CategoricalDtype(categories= ordering_combine, ordered=True)
data8['combine'] = data8[['offer_type','eventFinal']].agg('-'.join, axis=1).astype(t_combine)

print(data8['combine'].value_counts().index)

bbb_p = pd.crosstab(data8["combine"], data8["gender"], normalize="index", margins = True).fillna(0) * 100
bbb_p = bbb_p.reset_index()
print(bbb_p)


plt.figure(figsize=[14,8])
# Plot the distribution
bbb_p.plot(x='combine' , y=['F','M','O'],  kind="bar",figsize=(9,7), cmap="Blues_r", edgecolor=["#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28","#2b2b28", "#2b2b28"])
plt.xlabel(" ")
plt.ylabel("Response Rate ")
plt.xticks(rotation=-40,ha="left")
plt.title("Response to offer among Genders")
# plt.legend(['Female', 'Male','Undisclosed'])
plt.tight_layout()
plt.show()

print(bbb_p)


# ## Response rate among Age groups

age_cp = pd.crosstab(data8["combine"], data8["agegroups"], normalize="index", margins = True).fillna(0) * 100
age_cp = age_cp.reset_index()

# Plot the distribution
age_cp.plot(x='combine' , y=['babyboomer', 'elderlyadult', 'genX', 'millennials'],  kind="bar",figsize=(9,7), cmap="Blues_r", 
                                                                                    edgecolor=["#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28"])
plt.xlabel("Response to Offer")
plt.ylabel("Response Rate")
plt.xticks(rotation=45,ha="right")
plt.title("Response to offer among Age groups")
plt.tight_layout()
plt.show()

print(age_cp)



# ## Response rate among Income groups

income_cp = pd.crosstab(data8["combine"], data8["incomegroups"], normalize="columns", margins = True).fillna(0) * 100
income_cp = income_cp.reset_index()
# Plot the distribution
income_cp.plot(x='combine' , y=['Lowest','Lower-middle', 'Middle-income'],  kind="bar",figsize=(9,7), cmap="Blues", 
                                                                                    edgecolor=["#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28"])
plt.xlabel(" ")
plt.ylabel("Response Rate")
plt.xticks(rotation=-40,ha="left")
plt.title("Response to offer among Income groups")
plt.tight_layout()
plt.show()

print(income_cp)

# # Question III
# ## Who will response to an offer?
# - Build machine learning model that predicts whether or not someone will respond to an offer.


# # Get listing percentage of offer response
# Analyzing the distribution of offer response.
offerevents_p = data8["eventFinal"].value_counts(normalize=True) * 100
offerevents_count = data8["eventFinal"].value_counts() 
print("\n\n\n\nOffer events Percentages :", offerevents_p, "\n\n\n\nOffer events Raw counts" , 
      offerevents_count, sep="\n" )

  
# Plot the distribution of offer response
plt.figure(figsize=[10,8])
sns.barplot(x = offerevents_p.index, y= offerevents_p.values,  color = bcolor,   edgecolor="#2b2b28")
plt.xlabel("Event")
plt.ylabel("Percentage of members")
plt.title("Starbuck users responses to offer")
plt.tight_layout()
plt.show()


def pre_modelling(dataset):
    """The pre_modelling function select necessary features and split the  dataset
        into training and testing datasets
 
    Input:
        dataset(DataFrame) : the data to split
        split(float) : the test sample ration
        
        
    
    
    """
    
    selected_columns = ['time', 'reward', 'gender', 'age', 'income',
       'membershipDays', 'difficulty', 'duration',
       'Channel_email', 'Channel_mobile', 'Channel_social', 'Channel_web',
       'eventFinal', 'offer_type']
    
    data = dataset.copy()
    data = data[selected_columns]
    tt = data.offer_type.values
    
    data = pd.get_dummies(data, columns=['gender', 'offer_type'], prefix_sep="_")

    data['offerType'] = tt
    
    y = data.pop('eventFinal')
    
    y = pd.get_dummies(y, prefix_sep="_")
    y["offer viewed"] = y.apply(lambda x: x[2] if x[2] == 1 else x[1], axis=1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
                        data.iloc[:,:-4], y, test_size=0.2, stratify=data.iloc[:,-1], random_state=0)

    
    return X_train, X_test, Y_train, Y_test

X_train, X_test, y_train, y_test = pre_modelling(data8)


print(X_train.head())
print(y_test.head())


# search for best estimator
category_names = y_train.columns.tolist()
estimators = [
    ("RandomForestClassifier", MultiOutputClassifier(RandomForestClassifier(criterion = 'entropy', n_estimators=500, max_depth=20, random_state=0))),
    ("LinearSVC", MultiOutputClassifier(LinearSVC())),
    ("DecisionTreeClassifier", MultiOutputClassifier(DecisionTreeClassifier(max_depth = 20))),
    ("LogisticRegression", MultiOutputClassifier(LogisticRegression(random_state=0)))
]

for name, estimator in estimators:
    model = make_pipeline(MaxAbsScaler(), estimator)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name," mean_absolute_error : ", mean_absolute_error(y_test, pred))
    print(name," mean_squared_error: ", mean_squared_error(y_test, pred))
    print(name," Root mean_squared_error: ", np.sqrt(mean_squared_error(y_test, pred)))
    print(name," R2_squared : ", r2_score(pred, y_test))
    print(name," Accuracy : ", model.score(X_test, y_test), end="\n\n")
    
    # Calculate accuracy
    accuracy = (pred == y_test).mean()
    accuracyscore = model.score(X_test, y_test)
    
    
    print(f"Model Accuracy:")
    print(f'{accuracy}\n\n')
    print(f"Model Accuracy score: {accuracyscore}\n\n")


def build_model():
    """The build_model function build a model pipeline
    Args: None
    Returns:
        model(pipeline): model pipeline for fitting, prediction and scoring
    """
    # create pipeline
    plu = Pipeline([
                     ("mascaler", MaxAbsScaler()),
                     ('rforest', MultiOutputClassifier(RandomForestClassifier(criterion = 'entropy', n_estimators=500, max_depth=20, 
                                                                              oob_score = True, bootstrap = True, n_jobs=-1, random_state=0)))
            ])


    return plu


def evaluate_model(model, X_text, Y_test):
    """The evaluate_model function scores the performance of trained model
        on test (unseen) text and categories
    Args:
        model (model): model to evaluate
        X_text (numpy arrays): the test (unseen) tokenized text
        Y_test (numpy arrays): the test (unseen) target used for evaluation
        category_names(list): list containing the name of the categories
    Returns: None
            print out the accuracy and confusion metrics
    """
    sp = {"end": "\n\n", "sep": "\n\n"}

    # predict using the model
    pred = model.predict(X_text)

    # Calculate accuracy
    accuracy = (pred == Y_test).mean()
    accuracyscore = model.score(X_text, Y_test)    
    
    print(f"Model Accuracy:")
    print(f'{accuracy}\n\n')
    print(f"Model Accuracy score: {accuracyscore}\n\n")

treemodel = build_model()
treemodel.fit(X_train, y_train)


print(X_test.columns)
print(evaluate_model(treemodel, X_test, y_test))

print(treemodel)
print(treemodel.named_steps['rforest'].estimator.criterion)

# let get the feature importance
def dfform(lstt):
    """The dfform function form pandas dataframe from list
 
    Args: 
        lstt (list, series): the list or series to used in forming dataFrame
 
    Returns: 
        DataFrame: The DataFrame for analysis
 
    """
    df = pd.DataFrame(list(zip(X_test.columns, lstt)), columns=["features", "coefficients"])
    df3 = df.sort_values(by="coefficients", ascending=False).reset_index(drop=True)

    return df3


# important feature affecting prices using eli5 permutation
perm = PermutationImportance(treemodel).fit(X_test, y_test)

ee = dfform(perm.feature_importances_)
print(ee)


# Plot the distribution of offer response
plt.figure(figsize=[10,8])

ax1 = plt.subplot(111)
# ax1.set_frame_on(True)
ax1.spines['left'].set_visible(True)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
sns.barplot(x = ee.features, y= ee.coefficients,  color = bcolor, linewidth = 2,  edgecolor=["#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28","#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                           "#2b2b28", "#2b2b28", "#2b2b28", 
                                                                                          "#2b2b28"])

plt.xlabel(" ")
plt.axhline(y=0.00, color='black', linestyle='-')
plt.xticks(rotation=-40,ha="left")
# plt.title("User Attributes")
plt.tight_layout()
plt.show()

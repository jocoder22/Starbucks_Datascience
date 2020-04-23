# Cross-Industry Standard Process of Data Mining - (CRISP-DM)
# A. Bussiness Understanding                 
#     1. Formulate research/analysis question(s)
#             Question I
#                 Who is the typical Starbucks rewards mobile app user?
#                         Analysis of descriptive statistics of all variables in the dataset.
    
#            Question II
#                 Which demographic groups respond best to which offer type?
#                         Analyzing the response rate of each gender and age groups.

#             Question III
#                 Who will response to an offer?
#                         Build a model that predicts whether or not someone will respond to an offer.


# B. Data Understanding
#     1. Seek for relevant datasets
#     2. Read/Download relevant datasets
    
    
# import modules and libraries
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


% matplotlib inline

bcolor = sns.color_palette()[0]
blued = sns.color_palette("Blues_d")

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
    
    profile2["agegroups"] = pd.cut(profile2.age, bins=[18,34,50,70, profile2.age.max()+1],
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
#           a. Missing data analysis
# analyse missing gender
missing_gender = clean_data[clean_data['gender'].isna()]
missing_gender.event.value_counts()
missing_gender.head()
missing_gender.shape, missing_gender.person.nunique()

# analyse missing income
missing_income = clean_data[clean_data['income'].isna()]
missing_income.event.value_counts()
missing_income.head()
missing_income.shape, missing_income.person.nunique()

# analyse missing gender among those missing income
missing_income_gender = missing_income[missing_income['gender'].isna()]
missing_income_gender.event.value_counts()


# analyse none missing gender among those missing income
missing_income_gender = missing_income[~missing_income['gender'].isna()]
missing_income_gender.shape

# analyse none missing gender among those missing income
missing_income_gender = missing_income[~missing_income['gender'].isna()]
missing_income_gender.shape

# analyse age amongst those missing gender and income
missing_income_gender.age.describe()

# insights
# From the analysis above, members missing gender data also had missing income data.
# Looking carefully at the age for members with missing gender and income data, the mean, max, median, 25% 
# and 75% percentiles are the same, 118 years with standard deviation of 0.0. Therefore,
        # Apart from the age of 118 years being questionable; all 2172 user had the same age, 118 years.
        # Also, many of those missing gender and income data were transaction (1303) with few receiving an offer (872) 
        # but none of them ever viewed or completed the offer.
        # I can conclude that the missing entries in the data is not random.
        # I can conclude that those missing income,age and gender data wouldn't contribute much information toward our analysis.
# Missing reward, difficulty, duration, offer_type, offerType2, Channel_email, Channel_mobile, Channel_social, Channel_web 
# are associated with features of offer. Therefore, they are missing if user didn't receive an offer. 
# Also, for user with missing data on these variable, the event recorded is only transaction, 
# confirming that those user didn't receive any offer.

# Based on the above analysis, it seem reasonable to drop missing gender and age data from further analysis



# Working with complete, non-missing dataset

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


data = clean_data[~clean_data['gender'].isna()]
person_encoded = col_encoder(data, "person")
offer_encoded = col_encoder(data, "offer_id")

data.insert(1, 'user_id', person_encoded)
data.insert(3 , 'offer_id2', offer_encoded)

data = data.fillna({'reward': 99 ,'difficulty':99, 'offer_type':'No_offer', 
                    'offerType2':'No_offer','duration':0, 'Channel_web':0,
                    'Channel_email':0, 'Channel_mobile':0, 'Channel_social':0})
data.info()
data['gender'].value_counts()
data.event.value_counts()
exam_understand(data)
data.shape, data.person.nunique()

# Drop duplicates and get one record per user for further analysis
data2 = data.drop_duplicates(subset=['person'])
data2.shape



#     5. Data visualization
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
ordering_ = ['offer received', 'offer viewed', 'offer completed']
t = pd.CategoricalDtype(categories= ordering_, ordered=True)

devent["event"] = devent["event"].astype(t)
devent.event.value_counts()


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



# Create a histogram of income
plt.figure(figsize=[10,8])
plt.hist(data2['income'], bins=10)
plt.title('Distribution of Income')
plt.ylabel("Count")
plt.xlabel('Income');


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


# Create a histogram of membership Days among genders for only Males and females
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


# Assess correlations
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



# Analyzing response
dada = data[data['event'] != "transaction"]
data8 = dada.drop_duplicates(subset=['person'], keep="last")

data8['eventFinal'] = data8['event'].apply(lambda x: "No response" if x =="offer received" else x)
data8.eventFinal.value_counts()


pecross = pd.crosstab(data["person"], data["event"]).reset_index()
pecross.columns.name = None
pecross.head()

pecross3 = pd.crosstab(data["person"], data["offer_type"]).reset_index()
pecross3.columns.name = None
pecross3.head()


# # Get listing percentage of users response to offer
# Analyzing the distribution of users response to offer
offerresponse = data8["eventFinal"].value_counts(normalize=True) * 100
offerresponse_count = data8["eventFinal"].value_counts() 
print("\n\n\n\nusers response to offer Percentages :", offerresponse, "\n\n\n\nusers response to offer Raw counts" , 
      offerresponse_count, sep="\n" )



  
# Plot the distribution of users response to offer
plt.figure(figsize=[10,8])
sns.barplot(x = offerresponse.index, y= offerresponse.values,  color = bcolor,   edgecolor="#2b2b28")
plt.xlabel("Response ")
plt.ylabel("Percentage of members")
plt.title("Starbucks Users Response to offer")
plt.tight_layout()
plt.show()



offertype2 = pd.crosstab(data8["offer_type"], data8["eventFinal"], normalize="all") * 100
offertype22 = pd.crosstab(data8["offer_type"], data8["eventFinal"], normalize="all", margins = True) * 100
offertype = pd.crosstab(data8["offer_type"], data8["eventFinal"],   margins = True)
offertype


# # Plot the distribution of offer responses
offertype2.plot.bar(stacked=True, cmap='Blues_r', figsize=(10,7), edgecolor=["#2b2b28", "#2b2b28", "#2b2b28")
plt.xticks(rotation=0)
plt.xlabel("Response across offer")
plt.ylabel("Percent")
plt.title("  Starbucks Offer Response Distribution ")
plt.tight_layout()
plt.show()


# D. Modeling
#     1. Tackle reseach questions
#     2. Supervised learning
# E. Deployment
#     1. Summary report
#     2. Conclusion(s)

    
    





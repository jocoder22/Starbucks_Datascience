# Starbucks_Datascience

Writing a Data Scientist Blog Post

# Starbucks Data Scientist Job!

## Introduction
The project analyzed simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, the company sends out an advertisement, discounts or BOGO (buy one get one free) offers to some users of the mobile app..

## Data Sets
The data is contained in three files:

- portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
- profile.json - demographic data for each customer
- transcript.json - records for transactions, offers received, offers viewed, and offers completed

## Here is the schema and explanation of each variable in the files:

### portfolio.json

- id (string) - offer id
- offer_type (string) - type of offer ie BOGO, discount, informational
- difficulty (int) - minimum required spend to complete an offer
- reward (int) - reward given for completing an offer
- duration (int) - time for offer to be open, in days
- channels (list of strings)

### profile.json

- age (int) - age of the customer
- became_member_on (int) - date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

### transcript.json

- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
-- person (str) - customer id
- time (int) - time in hours since start of test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record

The first task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. The data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

* **Required packages** 
  
  The analysis is done using `Python 3.8.1` 
  
  Python Packages used are: 
  
    - `numpy`
    - `Pandas`
    - `Matlabplot`
    - `Scklearn`
    - `Seaborn`
    
    
###  Age groups
- Elder Adults (over 70 years old)
- Baby Boomers (Roughly 51 to 70 years old)
- Generation X (Roughly 35 – 50 years old)
- Millennials, or Generation Y (18 – 34 years old)
- Generation Z, or iGeneration (Teens & younger)


 ## GitHub files
 * analysis_plan.txt
    - Plain text file that lays out Cross-Industry Standard Process of Data Mining - (CRISP-DM) for this project
  
 * analysis.py
    - Python script with codes used in this project
    
 
 ## Analysis questions
 Following the CRISP-DM process this project will find solutions to the following questions
 
 * Question I
  - Who is the typical Starbucks rewards mobile app user?
    -  Analysis of descriptive statistics of all variables in the dataset.

* Question II
   - Which demographic groups respond best to which offer type?
      -  Analyzing the response rate of each gender and age groups.

* Question III
  - Who will response to an offer? 
    - Build a model that predicts whether or not someone will respond to an offer.

## Jupyter Notebook
Find jupyter notebook [here](https://github.com/jocoder22/NewYork_Airbnb/blob/master/analysis.ipynb)

## Conclusion
* Manhattan, Brooklyn and Queens accounted for over 96% of Airbnb listings in New York City. Based on location, Brooklyn is more accessible to Manhattan which is the city center compared to Queens. Initial search for vacation should start with searching listings in Brooklyn and Manhattan.

* Private rooms seem to be the most affordable room type across the boroughs especially in The Bronx ($66), Staten Island ($65) while the prices are $71 in Queens and $76 in Brooklyn.

* Prices of listings on average are higher in Manhattan compared to other Boroughs in New York City. Prices of private rooms is $140 in Manhattan, more than double the price in The Bronx, Staten Island and Queens.

## Github repository 
Github repository at this [link](https://github.com/jocoder22/Starbucks_Datascience)

## blog post 
See blog post at this [link](https://medium.com/@okigbookey/starbucks-data-scientist-a9455d67a1cc) 
   



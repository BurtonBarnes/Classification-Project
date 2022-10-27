# Telco Churn
# Project Description

Telco is a a global leader in the telecommunications field that has ben in business for over forty years. I have decided to look into why customers stopped being customer over a set period of time, or customer churn. I will see if any elements influence the churn rate in a negative or positive manner.

# Project Goal

* Discover drivers of customer churn from telco database.
* Use drivers to develop machine learning model to classify churn rates as positive or negative.
* This information could be used to further our understanding of which elements contribute to or detract from a person's tendency to churn.

# Initial Thoughts

My initial hypothesis is that what will detract from churn rate are positive features and what will make churn rate higher or negative features. I believe that higher monthly rates, low tenure and not being a senior citizen will negatively affect the churn.

# The Plan

* Aquire data

* Explore data in search of drivers of churn
    * Answer the following initial questions
        * How often does churn occur?
        * Does age affect churn?
        * Does monthly charges affect churn?
        * Does tenure affect churn?
        
* Develop a Model to predict if a customer will churn
    * Use drivers identified in explore to build predictibve models of different types
    * Evaluate models on train and validate data
    * Select the best model on test data
    
* Draw Conclusions

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|Churn| Whether or not a customer churned|
|Senior Citizen| True or False, whether a customer is a senior citizen|
|Tenure| How long a customer has been with the company, measured in months|
|Monthly charges| How much someone is charged each month|
|Female Yes| True or False, whether a customer is female or not|

# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from SQL
3) Put the data in the file containing the cloned repo.
4) Run notebook.

# Takeaways and Conclusions
* Churn occurs in 1/4 of customers
* Those with a longer tenure are more likely to stay
* Those with lower monthly charges are more likely to stay
* Senior citizens are more likely to churn
* Being Female has no not a driver of churn

# Recommendations
* Lower monthly charges for the first few months until tenure is higher
* Target non-senior citizens for services due to them being less likely to churn
* Lower monthly charges for non-senior citizens until for first few months of tenure
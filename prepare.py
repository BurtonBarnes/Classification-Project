import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


#split Telco data
def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test

#prepare telco data
def prep_telco_data(df):
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'])
       
    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)
    
    # Convert binary categorical variables to numeric
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    # split the data
    train, validate, test = split_telco_data(df)
    
    return train, validate, test

def prep_telco(df):
    df = df.drop(columns=['internet_service_type_id', 'contract_type_id', 'payment_type_id', 'customer_id'])

    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    df['Female_Yes'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_Yes'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_Yes'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_Yes'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_Yes'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_Yes'] = df.churn.map({'Yes': 1, 'No': 0})
    
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type'
                            ]],
                              drop_first=True)
    df = pd.concat( [df, dummy_df], axis=1 )
    
    return df

# boplot of tenure vs churn
def get_tenure(train):
    sns.boxplot(data=train, y='churn', x='tenure')
    plt.title('Tenure vs. Churn')
    plt.show
    
# boxplot of monthly charges vs churn
def get_monthly_charges(train):
    sns.boxplot(data=train, y='churn', x='monthly_charges')
    plt.title('Monthly charges vs Churn')
    plt.show

# barplot of senior citizens vs churn
def get_senior_citizen(train):
    sns.barplot(data=train, x='churn', y='senior_citizen')
    plt.title('Senior Citizen vs. Churn')
    plt.show

# barplot comparing sex
def compare_sex(train):
    sns.barplot(data=train, x='churn', y='Female_Yes')
    plt.title('Sex vs. Churn')
    plt.show
    
def get_pie_churn(train):
    '''create pie chart for percent of churn'''

    # set values and labels for chart
    values = [len(train.churn[train.churn == 'Yes']), len(train.churn[train.churn == 'No'])] 
    labels = ['Churned', 'Did not churn']
    
    # generate and show chart
    plt.pie(values,labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4'])
    plt.title('1/4 of customers churned')
    plt.show()

########################## prepare chi ############################## 

#show chi for tenure
def get_chi_tenure(train):
    observed = pd.crosstab(train.tenure, train.churn)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

# show chi for senior_citizen
def get_chi_senior_citizen(train):
    observed = pd.crosstab(train.senior_citizen, train.churn)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
##################### prepare data for modeling ######################    

#model preparation    
def model_prep(train, validate, test):
    '''Prepare train, validate, and test data for modeling'''
    
    # drop unused columns
    keep_cols = ['churn',
                 'monthly_charges',
                 'senior_citizen',
                 'tenure']
    
    train = train[keep_cols]
    validate = validate[keep_cols]
    test = test[keep_cols]
    
    # Split data into predicting variables (X) and target variable (y) and reset the index for each dataframe
    train_X = train.drop(columns='churn').reset_index(drop=True)
    train_y = train[['churn']].reset_index(drop=True)
    
    validate_X = validate.drop(columns='churn').reset_index(drop=True)
    validate_y = validate[['churn']].reset_index(drop=True)
    
    test_X = test.drop(columns='churn').reset_index(drop=True)
    test_y = test[['churn']].reset_index(drop=True)
    
    return train_X, validate_X, test_X, train_y, validate_y, test_y


################# model evaluation on train and validate ####################

#tree modeling
def get_tree(train_X, validate_X, train_y, validate_y):
    '''get decision tree accuracy on train and validate data'''
    
    # create classifier object
    clf = DecisionTreeClassifier(max_depth=5,random_state=123)
    
    #fit model on training data
    clf = clf.fit(train_X, train_y)
    
    #print result
    print(f"Accuracy of Decision Tree on train data is {clf.score(train_X, train_y)}")
    print(f"Accuracy of Decision Tree on validate data is {clf.score(validate_X, validate_y)}")
    
#forest modeling
def get_forest(train_X, validate_X, train_y, validate_y):
    '''get random forest accuracy on train and validate data'''
    
    # create model object and fit it to training data
    rf = RandomForestClassifier(max_depth=4, random_state=123)
    rf.fit(train_X, train_y)
    
    # print result
    print(f"Accuracy of Random Forest on train is {rf.score(train_X, train_y)}")
    print(f"Accuracy of Random Forest on validate is {rf.score(validate_X, validate_y)}")
      
# regression modeling
def get_reg(train_X, validate_X, train_y, validate_y):
    '''get logistic regression accuracy on train and validate data'''
    
    #create model object and fit it to the training data
    logit = LogisticRegression(solver='liblinear')
    logit.fit(train_X, train_y)
    
    #print result
    print(f"Accuracy of Logistic Regression on train is {logit.score(train_X, train_y)}")
    print(f"Accuracy of Logistic Regression on validate is {logit.score(validate_X, validate_y)}")
       
# KNN modeling
def get_knn(train_X, validate_X, train_y, validate_y):
    '''get KNN accuracy on train and validate data'''
    
    #create model object and fit it to the training data
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(train_X, train_y)
    
    #print result
    print(f"Accuracy of Logistic Regression on train is {knn.score(train_X, train_y)}")
    print(f"Accuracy of Logistic Regression on validate is {knn.score(validate_X, validate_y)}")
            
        
################ model evaluation on test ################

# forest modeling on test            
def get_forest_test(train_X, test_X, train_y, test_y):
    '''get random forest accuracy on train and test data'''
    
    # create model object and fit it to training data
    rf = RandomForestClassifier(max_depth=4, random_state=123)
    rf.fit(train_X, train_y)
    
    # print result
    print(f"Accuracy of Random Forest on test is {rf.score(test_X, test_y)}")

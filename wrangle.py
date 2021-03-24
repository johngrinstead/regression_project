import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    

## you will need to use the properties_2017 and predictions_2017 tables.

## square feet of the house ("living square feet"), number of bedrooms, number of bathrooms, the assess value of the house by the tax appraisal district ('taxvaluedollarcnt'...this will be your target variable in the project), and 2-3 other variables

## may, june, july or august (date is in predictions_2017 table)

## single unit property values

sql = '''
select parcelid, calculatedfinishedsquarefeet as square_feet, bedroomcnt as bedrooms, bathroomcnt as bathrooms, yearbuilt, fips, lotsizesquarefeet as lot_size, taxvaluedollarcnt as tax_value
from properties_2017
join predictions_2017 using(parcelid)
where transactiondate between "2017-05-01" and "2017-08-31"
AND propertylandusetypeid > 250
AND propertylandusetypeid < 280 
AND propertylandusetypeid != 270 
AND propertylandusetypeid != 271
OR  unitcnt = 1;
'''


def wrangle_zillow():
    '''
    This function will acquire and prep the Zillow data for initial exploration and modeling
    '''
    data = pd.read_csv("zillow.csv")
    
    data['house_age'] = (2021 - data.yearbuilt)
    
    data = data.drop(columns = ['Unnamed: 0', 'yearbuilt'])
    
    ## remove outliers, filter out extreme values and remove 'homes' with no rooms

    index3500 = data.loc[data['square_feet'] >= 3500].index
    data.drop(index3500 ,  inplace=True)
    
    index_tax_value = data.loc[data['tax_value'] >= 1680000].index
    data.drop(index_tax_value ,  inplace=True)
    
    index_lot_size = data.loc[data['lot_size'] >= 23778.0].index
    data.drop(index_lot_size ,  inplace=True)
    
    index_bedrooms = data.loc[data['bedrooms'] == 0].index
    data.drop(index_bedrooms , inplace=True)
    
    index_bathrooms = data.loc[data['bathrooms'] == 0].index
    data.drop(index_bathrooms , inplace=True)
    
    data = data.set_index("parcelid")
    
    data = data.dropna()
    # remove all NaN values
    
    # Remove decimal
    data['bedrooms'] = data['bedrooms'].astype(int)
    data['bathrooms'] = data['bathrooms'].astype(int)
    data['fips'] = data['fips'].astype(int)
    data['house_age'] = data['house_age'].astype(int)
    
    return data


def wrangle_zillow_fe():
    '''
    This function acquires and prep's the data with certain features to be removed after 
    they were found to not be useful predictors with feature engineering.
    '''
    data = pd.read_csv("zillow.csv")
    
    data['house_age'] = (2021 - data.yearbuilt)
    
    data = data.drop(columns = ['Unnamed: 0', 'yearbuilt', 'fips', 'lot_size'])
    # 'fips' and 'lot_size' were removed after being found to be innefective predictors 
    
    
    ## remove outliers, filter out extreme values and remove 'homes' with no rooms

    index3500 = data.loc[data['square_feet'] >= 3500].index
    data.drop(index3500 ,  inplace=True)
    
    index_tax_value = data.loc[data['tax_value'] >= 1680000].index
    data.drop(index_tax_value ,  inplace=True)
    
    index_bedrooms = data.loc[data['bedrooms'] == 0].index
    data.drop(index_bedrooms , inplace=True)
    
    index_bathrooms = data.loc[data['bathrooms'] == 0].index
    data.drop(index_bathrooms , inplace=True)
    
    data = data.set_index("parcelid")
    
    data = data.dropna()
    # remove all NaN values
    
    # Remove decimal
    data['bedrooms'] = data['bedrooms'].astype(int)
    data['bathrooms'] = data['bathrooms'].astype(int)
    data['house_age'] = data['house_age'].astype(int)
    
    return data




def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=319)
        train, validate = train_test_split(train, test_size=.3, random_state=319)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=319, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=319, stratify=train[stratify_by])
    
    return train, validate, test


def seperate_y(train, validate, test):
    '''
    This function will take the train, validate, and test dataframes and seperate the target variable into its
    own panda series
    '''
    X_train = train.drop(columns=['tax_value'])
    y_train = train.tax_value

    X_validate = validate.drop(columns=['tax_value'])
    y_validate = validate.tax_value

    X_test = test.drop(columns=['tax_value'])
    y_test = test.tax_value
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def scale_data(train, validate, test):
    
    '''
    This function will scale numeric data using Min Max transform after 
    it has already been split into train, validate, and test.
    '''
    
    # Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    # We fit on the training data
    # we only .fit on the training data
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
    return train_scaled, validate_scaled, test_scaled



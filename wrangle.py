import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env

from sklearn.model_selection import train_test_split


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
sql = '''
select parcelid, calculatedfinishedsquarefeet as square_feet, bedroomcnt as bedrooms, bathroomcnt as bathrooms, taxamount as taxes, taxvaluedollarcnt as tax_value, yearbuilt, regionidcounty as county, lotsizesquarefeet as lot_size
from properties_2017
join predictions_2017 using(parcelid)
where transactiondate between "2017-05-01" and "2017-06-30"
and unitcnt = 1;
'''


def wrangle_zillow():
    data = pd.read_csv("zillow.csv")
    
    data = data.set_index("parcelid")
    
    data = data.dropna()
    # remove all NaN values
    
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



# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:33:10 2021

@author: Mrityunjay
"""
import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd

#Check if there is special character in columns and fill with na   
def checkSpecialChar_fillna(df):
    """
    Parameters
    ----------
    df : DataFrame
        Dataframe to check if there are any columns having specical.

    Returns
    -------
    None.

    """
    for i in df.columns:
        df[i][df[i].apply(lambda i: True if re.search('^\s*$', str(i)) else False)]=None
        #Replace NA with 0
        df[i].fillna('0',inplace=True)
        return None
    
#Replace Na with No Info
def replaceNa(df):
    """
    Parameters
    ----------
    df : DataFrame
        DataFrame to check if columns have any na or null values.

    Returns
    -------
    column_holder : dict
        column names whose values have been updated from na or null to No Info.

    """
    column_holder={}
    for i in df.columns:
        if df[i].isnull().values.any()==True:
            df[i].fillna('No Info',inplace=True)
            column_holder[i]="Updated with No Info"
    return column_holder

#Function for Gettimg Mapping of values in LabelEncoder
def get_integer_mapping(le):
    """
    Parameters
    ----------
    le : TYPE
        a fitted SKlearn LabelEncoder.

    Returns
    -------
    res : TYPE
        Return a dict mapping labels to their integer values.

    """
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res



#Fitting & Transform Label Encoder
def labelEncoder_Mapper(cat_df):
    """
    Parameters
    ----------
    cat_df : DataFrame
        Categorical DataFrame whose encoding and corresponding Mapping is needed.

    Returns
    -------
    encoder_mapping : Dict
        Dictonary of encoder mapping.

    """
    print("------>Label Encoding and Mapping has begun pls wait<------")
    le=LabelEncoder()
    encoder_mapping = {}
    for i in cat_df.columns:
        
        cat_df[i]=le.fit_transform(cat_df[i])
        print(i,'Fit Done..Mapping Started')
        encoder_mapping[i+'_Mapping'] = get_integer_mapping(le)
    
    return encoder_mapping

def getMonth(df,column):
    """
    Parameters
    ----------
    df : DataFrame
        DataFrame having column for extracting month.
    column : datetime
        date column for extracting month.

    Returns
    -------
    df : DataFrame
        DatFrame having month as column against index.

    """
    month_={}
    for i in range(0,len(df)):
        month_[i]=df[column][i].month
    df=pd.DataFrame.from_dict(month_,orient='index')
    return df
    

def getHour(df,column):
    """
    Parameters
    ----------
    df : DataFrame
        Datraframe having timestamp column for extracting hour.
    column : timestamp
        Timestamp column for extracting hour.

    Returns
    -------
    df : DataFrame
        DatFrame having hour as column against index.

    """
    hour_={}
    for i in range(0,len(df)):
        hour_[i]=df[column][i].hour
    df=pd.DataFrame.from_dict(hour_,orient='index')
    return df    

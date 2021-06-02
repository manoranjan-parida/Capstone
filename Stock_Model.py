# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:47:05 2021

@author: Mrityunjay
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import EDAHelper

#Revenue per Itemcategory
stock_df=df_merged[['index','total_days','invoice_month','State','Make',
                    'PlantName1','ItemCategory', 'OrderQuantity','TotalAmtWtdTax_','City']]

#Segrigating data im Numerical and Categorical  
Numeric_data=stock_df.select_dtypes(exclude='object')
Cat_data=stock_df.select_dtypes(include='object')



#Convert data types of below columns.

                          
#Replace Na with No Info
replaced_columns=EDAHelper.replaceNa(Cat_data)
        
#Check if there is special character in columns and fill with na   
EDAHelper.checkSpecialChar_fillna(Cat_data)

#Changing Data Type by appending fixed character 'TP' infront of each value
#Cat_data_pb.Policyno_='TP'+Cat_data_pb.Policyno_.astype(str)

    
#Creating Model for Label Encoder and store mapping for future use
encoder_mapping=EDAHelper.labelEncoder_Mapper(Cat_data)

temp={}
for i in Cat_data.columns:
    temp[i]=EDAHelper.dataTypeCheck(Cat_data,i)
    
    
#DataFrame for stock Revenu 

data_model_stock=pd.concat([Numeric_data, Cat_data],axis=1)

np.where(stock_df['Total_Value']<5000,0,stock_df['Total_Value'])
data_model_stock['total_Valu_Cat']=np.where(data_model_stock['TotalValue']<5000,0,data_model_stock['TotalValue'])

data_model_stock['total_Valu_Cat']=np.where((data_model_stock['TotalValue']>=5000)&(data_model_stock['TotalValue']<10000),1,data_model_stock['TotalValue'])

data_model_stock['total_Valu_Cat']=np.where((data_model_stock['TotalValue']>=10000)),1,data_model_stock['TotalValue'])

data_model_stock[(data_model_stock >= r) & (dists <= r + dr)]

#Used to replace value in column
'''
0--><=5000
1-->greater than 5000 and less than equal to 10000 
2-->Greater than 10K
'''

data_model_stock.loc[data_model_stock['TotalAmtWtdTax_']<=5000,'total_Amt_Cat']=0

data_model_stock.loc[data_model_stock['TotalAmtWtdTax_']>10000,'total_Amt_Cat']=2

data_model_stock.loc[((data_model_stock['TotalAmtWtdTax_']>5000) & (data_model_stock['TotalAmtWtdTax_']<=10000) ),'total_Amt_Cat']=1

#Columns to Drop
['TotalAmtWtdTax_']

data_model_stock.total_days=data_model_stock.total_days.dt.days

#Splitting Data
y=data_model_stock['total_Amt_Cat']
X=data_model_stock.drop('total_Amt_Cat',axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=9)

rf=RandomForestClassifier()
rf.fit(X_train, y_train)
acc_rf=rf.score(X_test,y_test)
print("Score",acc_rf)

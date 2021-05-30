# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:51:10 2021

@author: Mrityunjay
"""

import EDAHelper
import pandas as pd
import matplotlib as plt
#File Path for Data Read


path='C:/Users/Komali Srinivas/Day1/capstone/fin_data.csv'

#Converting DataType of Date & Timestamp column
df_merged=pd.read_csv(path)
dataType_Conversion=['InvoiceDate', 'InvoiceTime','JobCardDate', 'JobCardTime',]
for i in dataType_Conversion:
    df_merged[i]=pd.to_datetime(df_merged[i])

#Segrigating data im Numerical and Categorical  
Numeric_data=df_merged.select_dtypes(exclude='object')
Cat_data=df_merged.select_dtypes(include='object')
#Convert data types of below columns.

#Dropping Columns not Required for now
Numeric_data.drop(columns = ['index:1','Unnamed_0','CGST_14', 'CGST_2_5','CGST_6', 'CGST_9', 'IGST_12', 'IGST_18', 'IGST_28', 'IGST_5','SGST_UGST_14', 'SGST_UGST_2_5', 'SGST_UGST_6','SGST_UGST_9', 'ServiceAdvisorName', 'TDSamount', 'TotalAmtWtdTax_',
       'SGST_UGST_9','Unnamed_0:1', 'TotalCGST', 'TotalGST', 'TotalIGST', 'TotalSGST_UGST','index:2', 'Unnamed_0:1'],inplace = True)

Cat_data.drop(columns = ['CustomerNo_','Area_Locality','Material','Description','Policyno_', 'Name2', 'Factorycalendar', 'CITY:1','ClaimNo_', 'ExpiryDate', 'Area_Locality','District','GatePassDate', 'GatePassTime', 'Plant:1', 'PlantName1','RegnNo','TechnicianName','LaborValueNumber'],inplace = True)

#Temporarily remove Policyno_,Material,Description for ease of Model Creation
#See problem in invoice/Job date and time.
                          
#Replace Na with No Info
replaced_columns=EDAHelper.replaceNa(Cat_data)
        
#Check if there is special character in columns and fill with na   
EDAHelper.checkSpecialChar_fillna(Cat_data)

#Changing Data Type by appending fixed character 'TP' infront of each value
#Cat_data_pb.Policyno_='TP'+Cat_data_pb.Policyno_.astype(str)

    
#Creating Model for Label Encoder and store mapping for future use
encoder_mapping=EDAHelper.labelEncoder_Mapper(Cat_data)

#Calculating totaldays
total_days=df_merged.InvoiceDate-df_merged.JobCardDate

#Extracting Month of Invoice Date
invoice_month=EDAHelper.getMonth(df_merged, 'InvoiceDate')
#Extracting Month of JobCard Date
JobCard_month=EDAHelper.getMonth(df_merged, 'JobCardDate')

#Extracting Hour of Invoice Time
df_merged['invoice_hour']=EDAHelper.getHour(df_merged,'InvoiceTime')
#Extracting Hour of Invoice Time
df_merged['JobCard_hour']=EDAHelper.getHour(df_merged,'JobCardTime')

#TODO
#1.Calculate difference between Job Date and Invoice date. - Done
#2.Calculate difference between jobtime and invoice time to see if work has been done in less than X hours
#3.Categorise service centers who are operating higher than usual time,
#4. Identify peak hours, less peak hour, free hour
#5.Modal can be created to classify to see if month is given will the service center have peak?
#6.From Description Create bag of words visulization.
#7.Identify Resourcing Needs?
#Is there anything fishy?


import numpy as np
temp['Policyno_'] = np.where(temp['Policyno_']=='0', 'NoInfo',temp['Policyno_'])

######Analysis of Monthly#############
df_merged['invoice_month'] = invoice_month
df_merged['total_days'] = total_days
EDA_Monthly=df_merged[['index','total_days','TotalValue', 'invoice_month','State','Make','Netvalue','PlantName1','TotalValue','invoice_hour','JobCard_hour']]

import seaborn as sns

sns.barplot(EDA_Monthly.invoice_month,EDA_Monthly.State)

sns.set_theme(style="whitegrid")
#EDA_Monthly = sns.load_dataset("EDA_Monthly")

sns.barplot(x="State", y="invoice_month", data=EDA_Monthly)


EDA_Monthly[EDA_Monthly.State=='Assam']['invoice_month'].value_counts()

State_list=set(EDA_Monthly.State)

monthly_vehicle=pd.DataFrame()
for i in State_list:
    monthly_vehicle[i]=EDA_Monthly[EDA_Monthly.State==i]['invoice_month'].value_counts()
    '''for j in range(0,len(monthly_vehicle.index)):
        print(monthly_vehicle[monthly_vehicle.index[j]])'''
        
monthly_revenue=pd.DataFrame()
for i in State_list:
    monthly_revenue[i]=EDA_Monthly[EDA_Monthly.State==i]['invoice_month'].value_counts()
    '''for j in range(0,len(monthly_vehicle.index)):
        print(monthly_vehicle[monthly_vehicle.index[j]])'''

makevsmonth = pd.DataFrame()
Make_List = set(EDA_Monthly.Make)
for i in Make_List:
    makevsmonth[i]=EDA_Monthly[EDA_Monthly.Make==i]['invoice_month'].value_counts()

#For getting state wise revenue model wise revenue  
#state_revenue = pd.DataFrame()
#for i in State_list:
#    print(i)
#    print(EDA_Monthly[EDA_Monthly.State==i]['TotalValue'].sum())
#    state_revenue[i]=EDA_Monthly[EDA_Monthly.State==i]['TotalValue'].sum()

monthly_state_revenue = pd.DataFrame()

monthly_state_revenue = pd.pivot_table(EDA_Monthly, values='TotalValue', index=['invoice_month'],
                    columns=['State'], aggfunc=np.sum)

#For getting make wise revenue model wise revenue 
monthly_make_revenue = pd.DataFrame()

monthly_make_revenue = pd.pivot_table(EDA_Monthly, values='TotalValue', index=['invoice_month'],
                    columns=['Make'], aggfunc=np.sum)

#Plotting a bar plot for make and state vs revenue

plt.figure(fig_size=(10,10))
monthly_state_revenue.plot.bar()


###########Model Building##################
data_model_cust_suggestion=df_merged[['OrderType','PlantName1','total_days']]

#TODO
#1.Add date and time for getting time taken in hour
#Converting time delta in int
data_model_cust_suggestion.total_days=data_model_cust_suggestion.total_days.dt.days

#encoder_mapping=EDAHelper.labelEncoder_Mapper(Cat_data)

Y=data_model_cust_suggestion['total_days']
x=data_model_cust_suggestion.drop('total_days',axis=1)

encoder_mapping_dmcs=EDAHelper.labelEncoder_Mapper(x)

from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test=train_test_split(x,Y,test_size=0.2,random_state=9)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=27)

log_reg.fit(x_train,Y_train)

acc = log_reg.score(x_test,Y_test)


#Model Selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

max_features=['sqrt','log2']
max_depth=[10,20,30,40,50]
min_samples_leaf=[1,2,5]
criterion=['gini','entropy']
n_estimators=[100,500,1000]
class_weight=['balanced','balanced_subsample']

grid_param={"max_features":max_features,
            "max_depth":max_depth,
            "min_samples_leaf":min_samples_leaf,
            "criterion":criterion,
            "n_estimators":n_estimators,
            "class_weight":class_weight}

rf=RandomForestClassifier()

rf_modal_selection=RandomizedSearchCV(estimator=rf,param_distributions=grid_param,n_iter=10,cv=5,random_state=3)

rf_modal_selection.fit(x_train,Y_train)
print(rf_modal_selection.best_params_)
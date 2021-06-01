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
data_model_cust_suggestion=df_merged[['OrderType','PlantName1','Make','Model',
                                      'invoice_month','KMsReading','Service_Class']]

#TODO
#1.Add date and time for getting time taken in hour
#Converting time delta in int
data_model_cust_suggestion.total_days=data_model_cust_suggestion.total_days.dt.days

#Segrigating data im Numerical and Categorical  
Numeric_data_modal=data_model_cust_suggestion.select_dtypes(exclude='object')
Cat_data_modal=data_model_cust_suggestion.select_dtypes(include='object')
#Replace Na
EDAHelper.replaceNa(Cat_data_modal)
#Encoding Categorical Data
encoder_mapping_dmcs=EDAHelper.labelEncoder_Mapper(Cat_data_modal)

#Merging two dataframe to form original
data_model_cust=pd.concat([Numeric_data_modal, Cat_data_modal],axis=1)

#Splitting Data
y=data_model_cust['Service_Class']
X=data_model_cust.drop('Service_Class',axis=1)

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=9)



#Model Selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

max_features=['sqrt','log2']
max_depth=[10,20,30,40,50]
min_samples_leaf=[1,2,5]
criterion=['mse', 'mae']
n_estimators=[100,500,1000]
#class_weight=['balanced','balanced_subsample']

grid_param={"max_features":max_features,
            "max_depth":max_depth,
            "min_samples_leaf":min_samples_leaf,
            "criterion":criterion,
            "n_estimators":n_estimators}

rf=RandomForestRegressor()

rf_modal_selection=RandomizedSearchCV(estimator=rf,param_distributions=grid_param,n_iter=10,cv=5,random_state=3)

rf_modal_selection.fit(X_train,y_train)
print(rf_modal_selection.best_params_)

################### Servicing Days Finder ######################
'''
Below Modals are for Prediciting 'In how many days servicing will be done' given 
'OrderType'-->What type of Service needed
'PlantName1'-->In which service center Customer want to Visit
'Make'-->What is the make of the Vehicle
'invoice_month'-->Month in which Customer want to visit
'KMsReading'-->What is the current KM Reading of the vehicle

'total_days'-->By when servicing will be done

Suggestion to Business:
Offer a feature in Mahindra First Choice website with above model to suggest users 
visiting for service if they will get their job done in no of days.

Also we can suggest to visit specific or next day when they will get thier vehicles job done
in less than today's.    
'''
#Trying KNN
from sklearn.neighbors import KNeighborsRegressor
KNN_Model=KNeighborsRegressor()
KNN_Model.fit(X_train, y_train)
acc_KNN = KNN_Model.score(X_test,y_test)
print("KNN Accuracy Score",acc_KNN)

#Check Score

#Trying Linear Regression
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
acc_linear=linear_model.score(X_test,y_test)
print("Linear Accuracy Score",acc_linear)


#Trying RF Regressior
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()
rf.fit(X_train, y_train)
acc_rf=rf.score(X_test,y_test)
print("Linear Accuracy Score",acc_rf)


#Trying Decision Tree Regressior
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train, y_train)
acc_dt=dt.score(X_test,y_test)
print("Linear Accuracy Score",acc_dt)

################## Stock Prediciton #########
'''
Will build Model to Predict Stock of Parts to be kept in given Month
df_merged[['index','total_days','TotalValue', 'invoice_month','State','Make','Netvalue','PlantName1','TotalValue','invoice_hour','JobCard_hour']]
'''
df_merged[['index','JobCard_Month','State','Make','Model','PlantName1',]]





['index', 'Plant', 'Name1', 'ValuationArea', 'Customernoplant',
       'Vendornumberplant', '', 'Name2', '',
       '', 'PostalCode', 'City', 'Salesorganization', 'State', '',
       '', '', '', 'Area_Locality',
        'CustType', 'CustomerNo_', 'District',
       'KMsReading',
       'LabourTotal', 'Make',  'Model', '', 'OSLTotal',
       'OrderType', '', 'PartsTotal', '', 
       'PlantName1', '', '', '', '',
       
       'ServiceAdvisorName', '', 'TechnicianName', '',
        'TotalValue',
       'UserID', 'OrderItem', 'Material',
       'LaborValueNumber', 'Description', 'ItemCategory', 'OrderQuantity',
       'TargetquantityUoM', 'Netvalue', 'total_days', 'invoice_month',
       'invoice_hour', 'JobCard_hour', 'inv_job_hour_diff']



########### Classifier Model Building##################
data_model_cust_suggestion=df_merged[['OrderType','PlantName1','Make','Model',
                                      'invoice_month','KMsReading','Service_Class']]

#TODO
#1.Add date and time for getting time taken in hour
#Converting time delta in int
data_model_cust_suggestion.total_days=data_model_cust_suggestion.total_days.dt.days

#Segrigating data im Numerical and Categorical  
Numeric_data_modal=data_model_cust_suggestion.select_dtypes(exclude='object')
Cat_data_modal=data_model_cust_suggestion.select_dtypes(include='object')
#Replace Na
EDAHelper.replaceNa(Cat_data_modal)
#Encoding Categorical Data
encoder_mapping_dmcs=EDAHelper.labelEncoder_Mapper(Cat_data_modal)

#Merging two dataframe to form original
data_model_cust=pd.concat([Numeric_data_modal, Cat_data_modal],axis=1)

#Splitting Data
y=data_model_cust['Service_Class']
X=data_model_cust.drop('Service_Class',axis=1)


#Trying RF Classifier
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train, y_train)
acc_rf=rf.score(X_test,y_test)
print("Score",acc_rf)

data_model_cust['Service_Class'] = np.where((data_model_cust.total_days>1) & (data_model_cust.total_days<=7), 'Within 7 Days', data_model_cust['Service_Class'])
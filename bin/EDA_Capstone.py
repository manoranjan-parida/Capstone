# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:51:10 2021

@author: Mrityunjay
"""

import EDAHelper
import pandas as pd
#File Path for Data Read


path='E:/GreyAtom/Capstone/230521/fin_data.csv'

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
invoice_hour=EDAHelper.getHour(df_merged,'InvoiceTime')
#Extracting Hour of Invoice Time
JobCard_hour=EDAHelper.getHour(df_merged,'JobCardTime')

#TODO
#1.Calculate difference between Job Date and Invoice date.
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
EDA_Monthly=df_merged[['index','total_days', 'invoice_month','invoice_hour','State','Make','Netvalue','PlantName1']]

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
    
    



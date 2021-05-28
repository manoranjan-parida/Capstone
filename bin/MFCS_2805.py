# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:02:04 2021

@author: Mrityunjay
"""
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

#File Path for Data Read
path='E:/GreyAtom/Capstone/230521/fin_data.csv'

#Converting DataType of Date & Timestamp column
fin_date=pd.read_csv(path)
dataType_Conversion=['InvoiceDate', 'InvoiceTime','JobCardDate', 'JobCardTime',]
for i in dataType_Conversion:
    pd.to_datetime(fin_date.i)

#Segrigating data im Numerical and Categorical  
Numeric_data_pb=fin_date.select_dtypes(exclude='object')
Cat_data_pb=fin_date.select_dtypes(include='object')
#Convert data types of below columns.

#Dropping Columns not Required for now
Numeric_data_pb.drop(columns = ['index:1','Unnamed_0','CGST_14', 'CGST_2_5','CGST_6', 'CGST_9', 'IGST_12', 'IGST_18', 'IGST_28', 'IGST_5','SGST_UGST_14', 'SGST_UGST_2_5', 'SGST_UGST_6','SGST_UGST_9', 'ServiceAdvisorName', 'TDSamount', 'TotalAmtWtdTax_',
       'SGST_UGST_9','Unnamed_0:1', 'TotalCGST', 'TotalGST', 'TotalIGST', 'TotalSGST_UGST','index:2', 'Unnamed_0:1'],inplace = True)

Cat_data_pb.drop(columns = ['CustomerNo_','Area_Locality','Material','Description','Policyno_', 'Name2', 'Factorycalendar', 'CITY:1','ClaimNo_', 'ExpiryDate', 'Area_Locality','District','GatePassDate', 'GatePassTime', 'Plant:1', 'PlantName1','RegnNo','TechnicianName','LaborValueNumber'],inplace = True)

#Temporarily remove Policyno_,Material,Description for ease of Model Creation
#See problem in invoice/Job date and time.
                          
#Replace Na with No Info
for i in Cat_data_pb.columns:
    if Cat_data_pb[i].isnull().values.any()==True:
        print(i)
        Cat_data_pb[i].fillna('No Info',inplace=True)
        print('Replace with No Info Done')
        
#Check if there is special character in columns and fill with na   

for i in fin_date.columns:
    fin_date[i][fin_date[i].apply(lambda i: True if re.search('^\s*$', str(i)) else False)]=None
    #Replace NA with 0
    fin_date.i.fillna('0',inplace=True)

#Changing Data Type by appending fixed character 'TP' infront of each value
#Cat_data_pb.Policyno_='TP'+Cat_data_pb.Policyno_.astype(str)

    
#Creating Model for Label Encoder
le=LabelEncoder()

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

dictionaries = {}

#Fitting & Transform Label Encoder
for i in Cat_data_pb.columns:
    print("Attempting Fit Transform of ",i)
    Cat_data_pb[i]=le.fit_transform(Cat_data_pb[i])
    print(i, "Fit Done")
    print(i, "Attemnpting for Mapping")
    dictionaries[i+'_Mapping'] = get_integer_mapping(le)
    print("Mapping Done")
    

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:49:28 2021

@author: wille
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pandas
import pandas as pd

# Numpy
import numpy as np

# Statsmodels
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

#Linear Imputation
from scipy.interpolate import interp1d

# Plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Augmented Dickey Fuller Test
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

#Sqrt function
from math import sqrt

# MSE Function
from sklearn.metrics import mean_squared_error

# Options
pd.options.display.max_rows = 2500

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)
warnings.filterwarnings("ignore")






# In[2]:

# Reading Data
df = pd.read_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/CapstoneMainDF.csv", index_col=0)
productsdf = pd.read_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/ProductsDF.csv", index_col=0)

df.columns = map(str.lower, df.columns)


# In[3]:


# impute item names
productdict = dict(zip(productsdf.Item_ID, productsdf.Item_Desc))
df["product_name"] = df.item_id
df.product_name = df.product_name.map(productdict)
df['date'] = pd.to_datetime(df['date'])

del(productdict, productsdf)


# In[4]:


#create new project data set
projectdf = df.loc[(df["project"]!="NONE")]

# In[5]:


# Remove Project from DF
df = df.loc[(df["project"]=="NONE")]


# In[6]:


#Imputation of Temperature
mintemp_bu = df['mintemp']
maxtemp_bu = df['maxtemp']


df['mintemp'].interpolate(method='linear', inplace=True)    
df['maxtemp'].interpolate(method='linear', inplace=True) 


del(mintemp_bu, maxtemp_bu)

# In[7]:


#outlier detection
df["price"] = df["sales"] / df["quantity_sold"]


# In[8]:


#most sold items and pre calculations
x = df.groupby(["date","product_name"]).agg(daily_sales = pd.NamedAgg(column="sales", aggfunc = sum), 
                                            daily_quantity = pd.NamedAgg(column = "quantity_sold", aggfunc = sum)).reset_index()
x["date"] = pd.to_datetime(x["date"])

x2 = x.groupby(["product_name"]).agg(Total_sales = pd.NamedAgg(column = "daily_sales", aggfunc = sum),
                                    Total_quantity = pd.NamedAgg(column = "daily_quantity", aggfunc = sum), 
                                    mean_daily_sales = pd.NamedAgg(column = "daily_sales", aggfunc = "mean"),
                                    mean_daily_quantity = pd.NamedAgg(column = "daily_quantity", aggfunc = "mean"),
                                    Transaction_days = pd.NamedAgg(column = "product_name", aggfunc = "count")).sort_values("Total_sales",ascending=False)

x2["Total_Avg_Price"] = x2["Total_sales"]/x2["Total_quantity"]
#x2 = x2.round(2)

del(x2,x)

# In[9]:


#all prices except for tootsie rolls look normal at first glance but seeing occasional mispricings in the next step

# Impute States Tootsie Roll Price
df.loc[df.product_name == "TTS TOOTSIE ROLL $.10", "price"] = .10
# Recalculate Sales
df.sales = df.quantity_sold * df.price


# In[10]:


x = df.groupby(["date","site_id"]).agg(daily_sales = pd.NamedAgg(column="sales", aggfunc = sum), 
                                            daily_quantity = pd.NamedAgg(column = "quantity_sold", aggfunc = sum)).reset_index()
x["date"] = pd.to_datetime(x["date"])
x["price"] = (x["daily_sales"]/x["daily_quantity"]).round(2)

z = df.groupby(["date","site_id"]).agg(total_sales = pd.NamedAgg(column="sales", aggfunc = sum)).reset_index()
z = z.groupby(["site_id"]).agg(average_sales = pd.NamedAgg(column="total_sales", aggfunc = 'mean')).reset_index()
salesdict = dict(zip(z.site_id, z.average_sales))
x["average_sales"] = x.site_id
x.average_sales = x.site_id.map(salesdict)

x["Sales_Difference"] =  np.absolute(((x["daily_sales"]-x["average_sales"])/x["average_sales"])*100).round(2)
x = x.sort_values("Sales_Difference",ascending=False)


del(z, salesdict)


# In[11]:


print("std.dev = ",np.std(x.Sales_Difference).round(4))
print("mean = ",np.mean(x.Sales_Difference).round(4))
# Lets take three std.devs from the mean to classify outliers. (consider 4 to account for discounts)
# We calculate only the upper bound because we are working with absolute numbers.


# In[12]:


print("outlier threshold 4std= ",(np.std(x.Sales_Difference)*4+np.mean(x.Sales_Difference)).round(4))
th_4std =(np.std(x.Sales_Difference)*4+np.mean(x.Sales_Difference))




# In[13]:

temp = x

temp2 = temp.loc[(temp['Sales_Difference']>=th_4std)].sort_values("Sales_Difference",ascending=False)
print("This method at 4 std classifies",round((len(temp2)/len(df)*100),4), "% as outliers or ", len(temp2),"days")

del(temp, x, th_4std)

# In[14]:


# Take STD.dev
site_vector = list(temp2["site_id"])
date_vector = list(temp2["date"])
index_list = []

df.reset_index(inplace=True)

for i in range(len(site_vector)):
    temp = df.loc[(df['site_id']==site_vector[i]) & (df["date"]==date_vector[i])].sort_values("sales",ascending=False).head(1)  
    temp = temp.drop(columns = ["location_id", "open_date", "sq_footage", "locale", "maxtemp", "mintemp", "fiscal_period", "periodic_gbv", "current_gbv", "mpds"])
    index_list.append(temp.iloc[0,0])

del(temp2, date_vector, site_vector, i, temp)

# In[15]:

df.rename(columns={ df.columns[0]: "index_set" }, inplace = True)

df = df[~df.index_set.isin(index_list)]

del(index_list)


# In[16]:

#df.to_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/DF_cleaned.csv")


# In[17]:
    
# Clean Dimensionality Reduction in Dataframe
df.drop(["index_set", "location_id","current_gbv", "product_name", "open_date", "fiscal_period", "project", "price"], axis = 1, inplace = True)

locale_string = list(df.locale.unique())
locale_replace = list(range(1,len(locale_string)+1))
df.locale.replace(locale_string, locale_replace, inplace=True)
del(locale_string, locale_replace)


df = df[["site_id", "item_id", "date", "sq_footage","mpds", "locale", "periodic_gbv", "maxtemp", "mintemp","quantity_sold", "sales"]]

# In[18]:

df.to_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/df_clean_opt.csv")
print("")
print("----dataframe Saved----")


# In[19]:

__df_agg = df.groupby(["item_id"]).agg(total_sales = pd.NamedAgg(column="sales", aggfunc = sum), 
                                     days_sold = pd.NamedAgg(column="sales", aggfunc = "count")).reset_index().sort_values("total_sales", ascending=False)

item_list = list(__df_agg.item_id.head(2))
site_list = list(df.site_id.unique())
date_list = list(df.sort_values("date").date.unique())
__app_list = []
prediction_dict = {}
test_dict = {}
model_sum_dict = {}

# Create a dictionary for none changing values
df_dict = {}

for __i in site_list:
    df_dict[int(__i)] = []


for __i in range(0, len(site_list)):
    
    __site_id_value = site_list[__i]
    
    __df_temp = df.loc[(df['site_id']== __site_id_value)].reset_index(drop=True).sort_values('date')
    
    df_dict[__site_id_value].append(int(__df_temp.sq_footage.mode()))
    df_dict[__site_id_value].append(int(__df_temp.mpds.mode()))
    df_dict[__site_id_value].append(int(__df_temp.locale.mode()))
    df_dict[__site_id_value].append(int(__df_temp.periodic_gbv.mode()))

# In[20]

# Create new Data Frames
import time
__p_values = [2,7,14]
__d_values = range(0, 3)
__q_values = range(0, 3)
__best_score = 1000000000
__best_cfg = 0
cfg_list = []
__timeA = time.time()
__timeB = time.time()

for __i in range(0,len(item_list)):
    
    if __i > 0:
        __item_id_value = item_list[__i]
        __timeB = time.time()
        __timediff = __timeB-__timeA
        print("Progress = ", __i,"/",int(len(item_list)-1)," Items Done | Time Passed = ",round(__timediff,0)," Seconds | Time Left = ", (int(len((item_list-1))/__i)*__timediff))
        
    else:
        print("-----Initializing-----")
    for __x in range(0,len(site_list)):
        __site_id_value = site_list[__x]
        
        __df_name = 'df_' +str(__site_id_value)+"_"+str(__item_id_value)
        
        __df_temp = df.loc[(df['site_id']== __site_id_value) & (df['item_id'] == __item_id_value)].reset_index(drop=True).sort_values('date')
        
        pre_test_df = __df_temp
        
        # Get List for Imputation
        __index_pos = date_list.index(__df_temp.date.unique()[0])
        trunc_date_list = list(date_list[__index_pos:len(date_list)])
        
        diff_list = list(set(trunc_date_list)-set(list(__df_temp.date.unique())))
        
        if len(diff_list) > 0:
            
            for __y in range(0, len(diff_list)):
                __app_list = []
                __app_list.extend((__site_id_value, __item_id_value, diff_list[__y], 
                                 df_dict[__site_id_value][0], df_dict[__site_id_value][1], 
                                 df_dict[__site_id_value][2], df_dict[__site_id_value][3], 
                                 np.nan, np.nan, 0, 0))
                __app_series = pd.Series(__app_list, index=__df_temp.columns)                
                __df_temp = __df_temp.append(__app_series, ignore_index=True)
            
        else: 
            pass
            
        __df_temp = __df_temp.sort_values("date")
        __df_temp['mintemp'].interpolate(method='linear', inplace=True)    
        __df_temp['maxtemp'].interpolate(method='linear', inplace=True) 
        __df_temp["dateind"] = __df_temp['date']
        __df_temp.set_index('dateind', inplace=True)
        __df_temp.index = pd.DatetimeIndex(__df_temp.index.values, freq=__df_temp.index.inferred_freq)
        __test_temp = __df_temp.tail(14)
        __df_temp = __df_temp.drop(__df_temp.tail(14).index)
        
        
        __ad_out = adfuller(__df_temp.sales)
        __lags = __ad_out[2]
        __p_val = __ad_out[1]
        
        if __p_val <= 1:
            # Data is Stationary so ARIMA(p,0,q)
        
            __best_score = 1000000000
            __best_cfg = (0,0,0)
            

            for __p in __p_values:
                for __d in __d_values:
                    for __q in __q_values:
                        __order = (__p,__d,__q)
                        try:
                            __model = ARIMA(exog=__df_temp[['maxtemp', 'mintemp']],endog = __df_temp['sales'],order = __order)
                            __model_fit = __model.fit()
                            __output = __model_fit.predict(start=len(__df_temp), end = int(len(__df_temp)+13),dynamic=False, exog = __test_temp[['maxtemp', 'mintemp']]) 
                            __rmse = sqrt(mean_squared_error(__output, __test_temp.sales))
                            if __rmse < __best_score:
                                __best_score = __rmse
                                __best_cfg = __order
                            else:
                                pass
                        except:
                            continue
            __model = ARIMA(exog=__df_temp[['maxtemp', 'mintemp']],endog = __df_temp['sales'],order = __best_cfg)
            __model_fit = __model.fit()
            __output = __model_fit.predict(start=len(__df_temp), end = int(len(__df_temp)+13),dynamic=False, exog = __test_temp[['maxtemp', 'mintemp']])
            model_sum_dict[__site_id_value,__item_id_value] = __model_fit.summary()
            prediction_dict[__site_id_value,__item_id_value] = __output
            test_dict[__site_id_value,__item_id_value] = __test_temp.sales
            cfg_list.append(__best_cfg)
            
        else:
            # Data is non Stationary so ARIMA(p,1,q)
            pass
        
        

# In[20]




                    



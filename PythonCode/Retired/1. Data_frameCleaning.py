#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pandas
import pandas as pd
print('Pandas: %s' % pd.__version__)

# Numpy
import numpy as np
print('Numpy: %s' % np.__version__)

# Statsmodels
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
print('Statsmodels: %s' % sm.__version__)

#
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

pd.options.display.max_rows = 2500


# In[2]:


df = pd.read_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/CapstoneMainDF.csv", index_col=0)
productsdf = pd.read_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/ProductsDF.csv", index_col=0)

df.columns = map(str.lower, df.columns)


# In[3]:


# impute item names
productdict = dict(zip(productsdf.Item_ID, productsdf.Item_Desc))
df["product_name"] = df.item_id
df.product_name = df.product_name.map(productdict)
df['date'] = pd.to_datetime(df['date'])

display(df.product_name.head(10))

del(productdict, productsdf)


# In[4]:


#create new project data set
projectdf = df.loc[(df["project"]!="NONE")]
display(projectdf.head())
print("Format of ProjectDF subset = ", format(projectdf.shape))


# ## Data Cleaning

# In[5]:


# Remove Project from DF
df = df.loc[(df["project"]=="NONE")]
print("Format of DF subset = ", format(df.shape))


# In[6]:


#Imputation of Temperature
mintemp_bu = df['mintemp']
maxtemp_bu = df['maxtemp']


df['mintemp'].interpolate(method='linear', inplace=True)    
df['maxtemp'].interpolate(method='linear', inplace=True) 


del(mintemp_bu, maxtemp_bu)

# ### Outlier Detection

# In[7]:


#outlier detection
df["price"] = df["sales"] / df["quantity_sold"]

df.head(5)


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
x2.head(15)

del(x2,x)

# In[9]:


#all prices except for tootsie rolls look normal at first glance but seeing occasional mispricings in the next step

df.loc[df.product_name == "TTS TOOTSIE ROLL $.10", "price"] = .10
df.sales = df.quantity_sold * df.price


# In[10]:


x = df.groupby(["date","site_id"]).agg(daily_sales = pd.NamedAgg(column="sales", aggfunc = sum), 
                                            daily_quantity = pd.NamedAgg(column = "quantity_sold", aggfunc = sum)).reset_index()
x["date"] = pd.to_datetime(x["date"])
x["price"] = (x["daily_sales"]/x["daily_quantity"]).round(2)
x.head(15)

z = df.groupby(["date","site_id"]).agg(total_sales = pd.NamedAgg(column="sales", aggfunc = sum)).reset_index()
z = z.groupby(["site_id"]).agg(average_sales = pd.NamedAgg(column="total_sales", aggfunc = 'mean')).reset_index()
salesdict = dict(zip(z.site_id, z.average_sales))
x["average_sales"] = x.site_id
x.average_sales = x.site_id.map(salesdict)

x["Sales_Difference"] =  np.absolute(((x["daily_sales"]-x["average_sales"])/x["average_sales"])*100).round(2)
x = x.sort_values("Sales_Difference",ascending=False)
x.head(15)

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
    print("\n")
    print("-------------------------------------------------")
    print("Day Statistics")
    display(temp2.iloc[[i]])
    print("-------------------------------------------------")
    print("Item Specifics")                  
    display(temp)
    print("-------------------------------------------------")      
    print("\n")
    index_list.append(temp.iloc[0,0])

del(temp2, date_vector, site_vector, index_list, i, temp)

# In[15]:


df.rename(columns={ df.columns[0]: "index_set" }, inplace = True)

df = df[~df.index_set.isin(index_list)]


# In[16]:


z = df.groupby(["date","site_id"]).agg(daily_sales = pd.NamedAgg(column="sales", aggfunc = sum)).reset_index()
z["date"] = pd.to_datetime(z["date"])

f, axes =  plt.subplots(5, 2, figsize=(20, 12), sharex=True, sharey=False)
f.autofmt_xdate()
f.suptitle('Outliers in aggregate sales data after removing quantities based on relative daily sales deviation from the mean', x=0.05, y=1, horizontalalignment='left', verticalalignment='top', fontsize = 15)
axes[0,0].plot(z["date"][z["site_id"]==554],z["daily_sales"][z["site_id"]==554],'tab:blue', linewidth=.5)
axes[0,0].set_title('Site_Id = 554')
axes[0,1].plot(z["date"][z["site_id"]==459],z["daily_sales"][z["site_id"]==459],'tab:green', linewidth=.5)
axes[0,1].set_title('Site_Id = 459')
axes[1,0].plot(z["date"][z["site_id"]==380],z["daily_sales"][z["site_id"]==380],'tab:orange', linewidth=.5)
axes[1,0].set_title('Site_Id = 380')
axes[1,1].plot(z["date"][z["site_id"]==516],z["daily_sales"][z["site_id"]==516],'tab:red', linewidth=.5)
axes[1,1].set_title('Site_Id = 516')
axes[2,0].plot(z["date"][z["site_id"]==517],z["daily_sales"][z["site_id"]==517],'tab:purple', linewidth=.5)
axes[2,0].set_title('Site_Id = 517')
axes[2,1].plot(z["date"][z["site_id"]==399],z["daily_sales"][z["site_id"]==399],'tab:brown', linewidth=.5)
axes[2,1].set_title('Site_Id = 399')
axes[3,0].plot(z["date"][z["site_id"]==280],z["daily_sales"][z["site_id"]==280],'tab:pink', linewidth=.5)
axes[3,0].set_title('Site_Id = 280')
axes[3,1].plot(z["date"][z["site_id"]==580],z["daily_sales"][z["site_id"]==580],'tab:gray', linewidth=.5)
axes[3,1].set_title('Site_Id = 580')
axes[4,0].plot(z["date"][z["site_id"]==589],z["daily_sales"][z["site_id"]==589],'tab:olive', linewidth=.5)
axes[4,0].set_title('Site_Id = 589')
axes[4,1].plot(z["date"][z["site_id"]==601],z["daily_sales"][z["site_id"]==601],'tab:cyan', linewidth=.5)
axes[4,1].set_title('Site_Id = 601')

del(z, axes, f)

# In[17]:
df.to_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/DF_cleaned.csv")

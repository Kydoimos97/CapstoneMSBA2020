#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning
# 
# Willem van der schans
# 
# 
# 3/19/2021

# In[1]:


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


display(df.head(10))
print(list(df.columns))
print(format(df.shape))


# In[4]:


display(productsdf.head(10))
print(list(productsdf.columns))
print(format(productsdf.shape))


# # Impute Item Names 

# In[5]:


productdict = dict(zip(productsdf.Item_ID, productsdf.Item_Desc))
df["product_name"] = df.item_id
df.product_name = df.product_name.map(productdict)
df['date'] = pd.to_datetime(df['date'])

display(df.product_name.head(10))


# # Create new Project Data Set

# In[6]:


projectdf = df.loc[(df["project"]!="NONE")]
display(projectdf.head())
print("Format of ProjectDF subset = ", format(projectdf.shape))


# In[7]:


# Remove Porject form DF
df = df.loc[(df["project"]=="NONE")]
print("Format of DF subset = ", format(df.shape))


# # Imputation of Temprature

# In[8]:


mintemp_bu = df['mintemp']
maxtemp_bu = df['maxtemp']


df['mintemp'].interpolate(method='linear', inplace=True)    
df['maxtemp'].interpolate(method='linear', inplace=True) 


# In[9]:


fig, (ax1,ax2) = plt.subplots(2, sharex=True,figsize=(20, 20))
fig.autofmt_xdate()
fig.suptitle('Before[1] and After[2] imputation of MinTemp. Blue = Mintemp, Orange=Maxtemp')
ax1.plot(df["date"], df["mintemp"], alpha = 1, linewidth=.5)
ax1.plot(df["date"], df["maxtemp"], alpha = 1, linewidth=.5)
ax2.plot(df["date"], mintemp_bu, alpha = 1, linewidth=.5)
ax2.plot(df["date"], maxtemp_bu, alpha = 1, linewidth=.5)


# # Outlier Detection

# ## Pre Engineering

# In[10]:


# Creation of Price
df["price"] = df["sales"] / df["quantity_sold"]


# In[11]:


z = df.groupby(["date","site_id"]).agg(daily_sales = pd.NamedAgg(column="sales", aggfunc = sum)).reset_index()
z["date"] = pd.to_datetime(z["date"])

f, axes =  plt.subplots(5, 2, figsize=(20, 20), sharex=True, sharey=False)
f.autofmt_xdate()
f.suptitle('Outliers in aggregate Sales data')
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


# ## Most Sold items and Pre Calculations
# 
# Calculating reference values and checking the most selling items in terms of total sales.

# In[12]:


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


# All Prices except for tootsie rolls look normal at first glance but seeing occasional mispricings is the next step. I will change the tootsie roll price to 10 cents to have it not show up in the analysis anymore as it's clear it's a mistake at this point.
# 

# In[13]:


df.loc[df.product_name == "TTS TOOTSIE ROLL $.10", "price"] = .10
df.sales = df.quantity_sold * df.price

# Recalc mean_price
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
x2.reset_index(inplace=True)
x2.head(15)


# ## Pricing Mistakes
# 
# Try to catch pricing mistakes by calculating the price difference from the mean. This captures relativety and makes interpreation easier. Using absolute numbers simplifies the process even further.

# In[14]:


pricedict = dict(zip(x2.product_name, x2.Total_Avg_Price))
df["mean_price"] = df.item_id
df.mean_price = df.product_name.map(pricedict)

df["price_diff_per"] = np.absolute(((df["price"]-df["mean_price"])/df["mean_price"])*100).round(2)
temp = df
temp = temp = temp.drop(columns = ["location_id", "open_date", "sq_footage", "locale", "maxtemp", "mintemp", "fiscal_period", "periodic_gbv", "current_gbv", "mpds"])

print("std.dev = ",np.std(df.price_diff_per).round(4))
print("mean = ",np.mean(df.price_diff_per).round(4))
# Lets take three std.devs from the mean to classify outliers. (consider 4 to account for discounts)
# We calculate only the upper bound because we are working with absolute numbers.

print("outlier threshold 3std= ",(np.std(df.price_diff_per)*3+np.mean(df.price_diff_per)).round(4))
th_3std =(np.std(df.price_diff_per)*3+np.mean(df.price_diff_per))
print("outlier threshold 4std= ",(np.std(df.price_diff_per)*4+np.mean(df.price_diff_per)).round(4))
th_4std =(np.std(df.price_diff_per)*4+np.mean(df.price_diff_per))
print("outlier threshold 5std= ",(np.std(df.price_diff_per)*5+np.mean(df.price_diff_per)).round(4))
th_5std =(np.std(df.price_diff_per)*5+np.mean(df.price_diff_per))
print("outlier threshold 6std= ",(np.std(df.price_diff_per)*6+np.mean(df.price_diff_per)).round(4))
th_6std =(np.std(df.price_diff_per)*6+np.mean(df.price_diff_per))


# In[15]:


df.reset_index(drop=True)


# In[16]:


temp = df

temp1 = temp.loc[(temp['price_diff_per']>=th_3std)].sort_values("price_diff_per",ascending=False)
print("This method at 3 std classifies",round((len(temp1)/len(df)*100),4), "% as outliers or ", len(temp1),"rows")

temp2 = temp.loc[(temp['price_diff_per']>=th_4std)].sort_values("price_diff_per",ascending=False)
print("This method at 4 std classifies",round((len(temp2)/len(df)*100),4), "% as outliers or ", len(temp2),"rows")

temp3 = temp.loc[(temp['price_diff_per']>=th_5std)].sort_values("price_diff_per",ascending=False)
print("This method at 5 std classifies",round((len(temp3)/len(df)*100),4), "% as outliers or ", len(temp3),"rows")

temp4 = temp.loc[(temp['price_diff_per']>=th_6std)].sort_values("price_diff_per",ascending=False)
print("This method at 6 std classifies",round((len(temp4)/len(df)*100),4), "% as outliers or ", len(temp4),"rows")


# In[17]:


temp4.head(25)


# In[18]:


temp1.reset_index(inplace=True)
index_drop_list = list(temp1["index"])
dfx=df
dfx.drop(index_drop_list,axis=0)



# In[19]:


z = dfx.groupby(["date","site_id"]).agg(daily_sales = pd.NamedAgg(column="sales", aggfunc = sum)).reset_index()
z["date"] = pd.to_datetime(z["date"])

f, axes =  plt.subplots(5, 2, figsize=(20, 20), sharex=True, sharey=False)
f.autofmt_xdate()
f.suptitle('Outliers in aggregate sales data after chaning tootsieroll price')
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


#  This all didn't do that much so we can assume that a lot of the outliers are based on quantities being input wrong not the prices. I won't pursue this method in the final project.

# ## Quantity Outliers

# In[20]:


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


# In[21]:


print("std.dev = ",np.std(x.Sales_Difference).round(4))
print("mean = ",np.mean(x.Sales_Difference).round(4))
# Lets take three std.devs from the mean to classify outliers. (consider 4 to account for discounts)
# We calculate only the upper bound because we are working with absolute numbers.

print("outlier threshold 3std= ",(np.std(x.Sales_Difference)*3+np.mean(x.Sales_Difference)).round(4))
th_3std =(np.std(x.Sales_Difference)*3+np.mean(x.Sales_Difference))
print("outlier threshold 4std= ",(np.std(x.Sales_Difference)*4+np.mean(x.Sales_Difference)).round(4))
th_4std =(np.std(x.Sales_Difference)*4+np.mean(x.Sales_Difference))
print("outlier threshold 5std= ",(np.std(x.Sales_Difference)*5+np.mean(x.Sales_Difference)).round(4))
th_5std =(np.std(x.Sales_Difference)*5+np.mean(x.Sales_Difference))
print("outlier threshold 6std= ",(np.std(x.Sales_Difference)*6+np.mean(x.Sales_Difference)).round(4))
th_6std =(np.std(x.Sales_Difference)*6+np.mean(x.Sales_Difference))


# In[22]:


temp = x

temp1 = temp.loc[(temp['Sales_Difference']>=th_3std)].sort_values("Sales_Difference",ascending=False)
print("This method at 3 std classifies",round((len(temp1)/len(df)*100),4), "% as outliers or ", len(temp1),"days")

temp2 = temp.loc[(temp['Sales_Difference']>=th_4std)].sort_values("Sales_Difference",ascending=False)
print("This method at 4 std classifies",round((len(temp1)/len(df)*100),4), "% as outliers or ", len(temp2),"days")

temp3 = temp.loc[(temp['Sales_Difference']>=th_5std)].sort_values("Sales_Difference",ascending=False)
print("This method at 5 std classifies",round((len(temp1)/len(df)*100),4), "% as outliers or ", len(temp3),"days")

temp4 = temp.loc[(temp['Sales_Difference']>=th_6std)].sort_values("Sales_Difference",ascending=False)
print("This method at 6 std classifies",round((len(temp1)/len(df)*100),4), "% as outliers or ", len(temp4),"days")


# In[23]:


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


# In[24]:


df.rename(columns={ df.columns[0]: "index_set" }, inplace = True)

df = df[~df.index_set.isin(index_list)]


# In[25]:


z = df.groupby(["date","site_id"]).agg(daily_sales = pd.NamedAgg(column="sales", aggfunc = sum)).reset_index()
z["date"] = pd.to_datetime(z["date"])

f, axes =  plt.subplots(5, 2, figsize=(20, 20), sharex=True, sharey=False)
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


# In[26]:


x.loc[(x['site_id']==517)  & (x['daily_sales']< 250)]


# In[27]:


df.loc[(df["date"]=="2018-09-11") & (df["site_id"] == 517)]


# Only three items got sold this whole day and there was no project interesting to see why this is the case.

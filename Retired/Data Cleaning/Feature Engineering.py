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


# # Tests

# ## Dicky Fuller

# In[17]:


#Autocorrelation
from matplotlib import pyplot 
from statsmodels.graphics.tsaplots import plot_acf

#exclude date time from acf plot
df_acf = df.loc[ : , (df.columns != 'date') & (df.columns != 'open_date')]
#exclude product name
df_acf = df_acf.loc[:, (df_acf.columns != 'product_name')]
#exclude locale
df_acf = df_acf.loc[:, (df_acf.columns != 'locale')]
#exclude project
df_acf = df_acf.loc[:, (df_acf.columns != 'project')]


fig, ax = plt.subplots(figsize=(20, 6))
plot_acf(df_acf['sales'].sample(500000), ax=ax)
plt.show()



# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


from pylab import rcParams 
rcParams['figure.figsize'] = 20,8
x = df['date']
y = df['sales']
plt.plot(x,y)


# H0: The Null Hypothesis: It is a statement about the population that either is believed to be true or it is used to put forth an argument unless it can be shown to be incorrect beyond a reasonable doubt
# 
#  H1: The alternative hypothesis: It is a claim about the population that is contradictory to H0 and what we conclude when we reject H0
#  
#  *H0: it is non-stationary*
#  *H1: it is stationary*
#  
#  We will be considering the null hypothesis that data is not stationary and the alternative hypothesis that data is stationary

# In[20]:


from statsmodels.tsa.stattools import adfuller


# In[21]:


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")


# In[22]:


df_samp = df.sample(5000)

adfuller_test(df_samp['sales'])


# In[23]:


from pylab import rcParams 
rcParams['figure.figsize'] = 20,8
x = df.loc[df.product_name != 'PLACEHOLDER']['date']
y = df.loc[df.product_name != 'PLACEHOLDER']['sales']
plt.plot(x,y)


# In[24]:


z = df.groupby(["date","site_id"]).agg(daily_sales = pd.NamedAgg(column="sales", aggfunc = 'count')).reset_index()
z["date"] = pd.to_datetime(z["date"])

f, axes =  plt.subplots(5, 2, figsize=(20, 12), sharex=True, sharey=False)
f.autofmt_xdate()
f.suptitle('Different Amount of products per day sold per Location', x=0.05, y=1, horizontalalignment='left', verticalalignment='top', fontsize = 15)
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


# In[95]:


df = df_bu

#df = df.sample(50000).reset_index()


# In[79]:





import time

lag = 7
days = 86400
counter = 0
timeA = time.time()
timediff2= 0
counterdiff = 100

for i in range(0,len(df)) :
    
    date_t = df.loc[i,'date']
    item_id_t = df.loc[i,'item_id']
    site_id_t = df.loc[i,'site_id']
    
    date_lagged = pd.Timestamp(df.loc[i,'date'].timestamp() - (lag*days), unit='s')
    
    sub = df.loc[(df['date']==date_lagged) & (df['site_id']== site_id_t) & (df['item_id'] == item_id_t)]
    
    if len(sub) > 0:
        df.loc[i,'lag_sales'] = sub.iloc[0,5]
    else:
        df.loc[i,'lag_sales'] =  0
        
    if i%int(len(df)/counterdiff) == 0:
        if counter == 0:
            counter = counter+1 
            print("===Initializing===")
        else:
            timeB = time.time()
            timediff = timeB-timeA
            timediff2 = timediff2+timediff
            print("Progress =", counter/(counterdiff/100),"% | ",
                  "Time Elapsed = ", round(timediff2,2), 
                  " Seconds | ", "Est. Time Remaining = ", round(timediff*(counterdiff-counter),2), " Seconds" )
            counter = counter+1
            timeA = time.time()
    else: 
        pass
    
    
    


# In[43]:


df.head()


# In[80]:


z = df.groupby(["date","site_id"]).agg(daily_sales = pd.NamedAgg(column="sales", aggfunc = sum), lag_sales = pd.NamedAgg(column='lag_sales', aggfunc=sum)).reset_index()
z["date"] = pd.to_datetime(z["date"])

f, axes =  plt.subplots(5, 2, figsize=(20, 20), sharex=True, sharey=False)
f.autofmt_xdate()
f.suptitle('Outliers in aggregate Sales data')
axes[0,0].plot(z["date"][z["site_id"]==554],z["daily_sales"][z["site_id"]==554],'tab:blue', linewidth=.5)
axes[0,0].plot(z["date"][z["site_id"]==554],z["lag_sales"][z["site_id"]==554],'tab:red', linewidth=.5, alpha = .5)
axes[0,0].set_title('Site_Id = 554')
axes[0,1].plot(z["date"][z["site_id"]==459],z["daily_sales"][z["site_id"]==459],'tab:blue', linewidth=.5)
axes[0,1].plot(z["date"][z["site_id"]==459],z["lag_sales"][z["site_id"]==459],'tab:red', linewidth=.5, alpha = .5)
axes[0,1].set_title('Site_Id = 459')
axes[1,0].plot(z["date"][z["site_id"]==380],z["daily_sales"][z["site_id"]==380],'tab:blue', linewidth=.5)
axes[1,0].plot(z["date"][z["site_id"]==380],z["lag_sales"][z["site_id"]==380],'tab:red', linewidth=.5, alpha = .5)
axes[1,0].set_title('Site_Id = 380')
axes[1,1].plot(z["date"][z["site_id"]==516],z["daily_sales"][z["site_id"]==516],'tab:blue', linewidth=.5)
axes[1,1].plot(z["date"][z["site_id"]==516],z["lag_sales"][z["site_id"]==516],'tab:red', linewidth=.5, alpha = .5)
axes[1,1].set_title('Site_Id = 516')
axes[2,0].plot(z["date"][z["site_id"]==517],z["daily_sales"][z["site_id"]==517],'tab:blue', linewidth=.5)
axes[2,0].plot(z["date"][z["site_id"]==517],z["lag_sales"][z["site_id"]==517],'tab:red', linewidth=.5, alpha = .5)
axes[2,0].set_title('Site_Id = 517')
axes[2,1].plot(z["date"][z["site_id"]==399],z["daily_sales"][z["site_id"]==399],'tab:blue', linewidth=.5)
axes[2,1].plot(z["date"][z["site_id"]==399],z["lag_sales"][z["site_id"]==399],'tab:red', linewidth=.5, alpha = .5)
axes[2,1].set_title('Site_Id = 399')
axes[3,0].plot(z["date"][z["site_id"]==280],z["daily_sales"][z["site_id"]==280],'tab:blue', linewidth=.5)
axes[3,0].plot(z["date"][z["site_id"]==280],z["lag_sales"][z["site_id"]==280],'tab:red', linewidth=.5, alpha = .5)
axes[3,0].set_title('Site_Id = 280')
axes[3,1].plot(z["date"][z["site_id"]==580],z["daily_sales"][z["site_id"]==580],'tab:blue', linewidth=.5)
axes[3,1].plot(z["date"][z["site_id"]==580],z["lag_sales"][z["site_id"]==580],'tab:red', linewidth=.5, alpha = .5)
axes[3,1].set_title('Site_Id = 580')
axes[4,0].plot(z["date"][z["site_id"]==589],z["daily_sales"][z["site_id"]==589],'tab:blue', linewidth=.5)
axes[4,0].plot(z["date"][z["site_id"]==589],z["lag_sales"][z["site_id"]==589],'tab:red', linewidth=.5, alpha = .5)
axes[4,0].set_title('Site_Id = 589')
axes[4,1].plot(z["date"][z["site_id"]==601],z["daily_sales"][z["site_id"]==601],'tab:blue', linewidth=.5)
axes[4,1].plot(z["date"][z["site_id"]==601],z["lag_sales"][z["site_id"]==601],'tab:red', linewidth=.5, alpha = .5)
axes[4,1].set_title('Site_Id = 601')


# In[62]:


from pylab import rcParams 
rcParams['figure.figsize'] = 20,8
x = df.loc[df.product_name != 'PLACEHOLDER']['sales']
y = df.loc[df.product_name != 'PLACEHOLDER']['lag_sales']
plt.plot(x, 'o', color="black")
plt.plot(y, 'o',color="blue")


# In[96]:


from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(df['sales'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()


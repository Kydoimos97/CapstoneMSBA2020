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


df = pd.read_csv("CapstoneMainDF.csv")
productsDF = pd.read_csv("ProductsDF.csv", encoding = None)

df.columns = map(str.lower, df.columns)


# In[3]:


# impute item names
productdict = dict(zip(productsDF.Item_ID, productsDF.Item_Desc))
df["product_name"] = df.item_id
df.product_name = df.product_name.map(productdict)
df['date'] = pd.to_datetime(df['date'])

display(df.product_name.head(10))


# In[4]:


#create new project data set
projectdf = df.loc[(df["project"]!="NONE")]
display(projectdf.head())
print("Format of ProjectDF subset = ", format(projectdf.shape))


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


# In[7]:


fig, (ax1,ax2) = plt.subplots(2, sharex=True,figsize=(20, 20))
fig.autofmt_xdate()
fig.suptitle('Before[1] and After[2] imputation of MinTemp. Blue = Mintemp, Orange=Maxtemp')
ax1.plot(df["date"], df["mintemp"], alpha = 1, linewidth=.5)
ax1.plot(df["date"], df["maxtemp"], alpha = 1, linewidth=.5)
ax2.plot(df["date"], mintemp_bu, alpha = 1, linewidth=.5)
ax2.plot(df["date"], maxtemp_bu, alpha = 1, linewidth=.5)


# In[8]:


#outlier detection
df["price"] = df["sales"] / df["quantity_sold"]


# In[9]:


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


# In[10]:


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


# In[11]:


#all prices except for tootsie rolls look normal at first glance but seeing occasional mispricings in the next step


# In[12]:


df.loc[df.product_name == "TTS TOOTSIE ROLL $.10", "price"] = .10
df.sales = df.quantity_sold * df.price


# In[13]:


#recalc mean price
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


# In[14]:


#Pricing Mistakes
#Try to catch pricing mistakes by calculating the price difference from the mean. This captures relativety and makes interpreation easier. Using absolute numbers simplifies the process even further.

pricedict = dict(zip(x2.product_name, x2.Total_Avg_Price))
df["mean_price"] = df.item_id
df.mean_price = df.product_name.map(pricedict)

df["price_diff_per"] = np.absolute(((df["price"]-df["mean_price"])/df["mean_price"])*100)
temp = df
temp = temp = temp.drop(columns = ["location_id", "open_date", "sq_footage", "locale", "maxtemp", "mintemp", "fiscal_period", "periodic_gbv", "current_gbv", "mpds"])

print("std.dev = ",np.std(df.price_diff_per))
print("mean = ",np.mean(df.price_diff_per))


# In[15]:


# Lets take three std.devs from the mean to classify outliers. (consider 4 to account for discounts)
# We calculate only the upper bound because we are working with absolute numbers.
print("outlier threshold 3std= ",(np.std(df.price_diff_per)*3+np.mean(df.price_diff_per))
      
th_3std = (np.std(df.price_diff_per)*3 + np.mean(df.price_diff_per))
#print("outlier threshold 4std= ",(np.std(df.price_diff_per)*4+np.mean(df.price_diff_per))
#th_4std =(np.std(df.price_diff_per)*4+np.mean(df.price_diff_per))
#print("outlier threshold 5std= ",(np.std(df.price_diff_per)*5+np.mean(df.price_diff_per))
#th_5std =(np.std(df.price_diff_per)*5+np.mean(df.price_diff_per))
#print("outlier threshold 6std= ",(np.std(df.price_diff_per)*6+np.mean(df.price_diff_per))
#th_6std =(np.std(df.price_diff_per)*6+np.mean(df.price_diff_per))


# In[16]:


df.reset_index(drop=True)


# ## Dickey Fuller Test

# #### Multivariate Time-series 

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

plot_acf(df_acf)
pyplot.show()


# #### Arima

# In[18]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# split data into train and test sets

# In[19]:


df.head()


# In[20]:


from pylab import rcParams 
rcParams['figure.figsize'] = 10,5
x = df['date']
y = df['sales']
plt.plot(x,y)


# H0: The Null Hypothesis: It is a statement about the population that either is believed to be true or it is used to put forth an argument unless it can be shown to be incorrect beyond a reasonable doubt
# 
# H1: The alternative hypothesis: It is a claim about the population that is contradictory to H0 and what we conclude when we reject H0
# 
# *H0: it is non-stationary*
# *H1: it is stationary*
# 
# We will be considering the null hypothesis that data is not stationary and the alternative hypothesis that data is stationary

# In[21]:


from statsmodels.tsa.stattools import adfuller


# In[22]:


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")


# *Note: I cannot run the **adfuller_test(df['sales'])** function as I do not have enough memory on my computer to do so*

# In[23]:


df_samp = df.sample(20000)

adfuller_test(df_samp['sales'])


# The above code should give you a p-value and should have something like:
# 
# ADF Test Statistic : -1.3399234
# p-value : 0.3659394
# num_lags used: 11
# Number of observations: 300 
# 
# **The numbers are obviously just fillers for an example's sake**

# In[24]:


df['Sales First Difference'] = df['sales'] - df['sales'].shift(1)
df['Seasonal First Difference']=df['sales']-df['sales'].shift(12)
df[['date','sales','Sales First Difference','Seasonal First Difference']].head()


# In[25]:


df_samp = df.sample(20000)

adfuller_test(df_samp['Seasonal First Difference'])


# In[26]:


x = df['date']
y = df['Seasonal First Difference']
rcParams['figure.figsize'] = 10,5

plt.plot(x,y)


# In[27]:


from pandas.plotting import autocorrelation_plot
df_samp = df.sample(1000)
autocorrelation_plot(df_samp['sales'])
rcParams['figure.figsize'] = 10,5
plt.show()


# In[28]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_samp['Seasonal First Difference'].dropna(),lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_samp['Seasonal First Difference'].dropna(),lags=40,ax=ax2)


# In[30]:


# For non-seasonal data
#p=1, d=1, q=0 or 1

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(df_samp['sales'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()


# In[31]:


df_samp['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df_samp[['sales','forecast']].plot(figsize=(12,8))


# In[34]:


from pandas.tseries.offsets import DateOffset

future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

future_datest_df.tail()

future_df=pd.concat([df,future_datest_df])

future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)
future_df[['sales', 'forecast']].plot(figsize=(12, 8))


# In[ ]:





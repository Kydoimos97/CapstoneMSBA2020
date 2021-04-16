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
from math import ceil
from math import floor

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

import time
import os

import tkinter as tk
from tkinter import ttk
from tkinter import *

# In[2]:
def check_dataframe_path():
    value = dataframe_path.get()
    value = str(value)
    try:
        pd.read_csv(value, index_col=0)
        dataframe_path_text_box.delete(1.0, "end-1c")
        dataframe_path_text_box.insert("end-1c",u'\u2714')
    except: 
        dataframe_path_text_box.delete(1.0, "end-1c")
        dataframe_path_text_box.insert("end-1c",u'\u2717') 

def check_productdf_path():
    value = productdf_path.get()
    value = str(value)
    try:
        pd.read_csv(value, index_col=0)
        productdf_path_text_box.delete(1.0, "end-1c")
        productdf_path_text_box.insert("end-1c",u'\u2714')
    except: 
        productdf_path_text_box.delete(1.0, "end-1c")
        productdf_path_text_box.insert("end-1c",u'\u2717') 
        
def check_standard_dev():
    value = std_dev_cl.get()
    if value.isdigit():
        std_dev_cl_text_box.delete(1.0, "end-1c")
        std_dev_cl_text_box.insert("end-1c",u'\u2714')
    else:
        std_dev_cl_text_box.delete(1.0, "end-1c")
        std_dev_cl_text_box.insert("end-1c",u'\u2717')
        
def check_item_id(): 
    value = item_id_inp.get()
    value_str = str(value)
    if value_str.startswith("-"):
        item_id_inp_text_box.delete(1.0, "end-1c")
        item_id_inp_text_box.insert("end-1c","Item_Id \u2714")
    elif value.isdigit():
        item_id_inp_text_box.delete(1.0, "end-1c")
        item_id_inp_text_box.insert("end-1c","Top_N \u2714")
    else: 
        item_id_inp_text_box.delete(1.0, "end-1c")
        item_id_inp_text_box.insert("end-1c",u'\u2717')
        
def check_site_id():
    value = site_id_inp.get()
    if value == "":
        site_id_text_box.delete(1.0, "end-1c")
        site_id_text_box.insert("end-1c", 'All_Sites \u2714')
    elif value.isdigit():
        if len(value)==3:
            site_id_text_box.delete(1.0, "end-1c")
            site_id_text_box.insert("end-1c", 'Site_Id \u2714')
        else:
            pass
    else:
        site_id_text_box.delete(1.0, "end-1c")
        site_id_text_box.insert("end-1c",u'\u2717')
        
def check_p():
    value = p_inp.get()
    if value.isdigit():
        p_inp_text_box.delete(1.0, "end-1c")
        p_inp_text_box.insert("end-1c",u'\u2714')
    else: 
        p_inp_text_box.delete(1.0, "end-1c")
        p_inp_text_box.insert("end-1c",u'\u2717')
        
        
def check_d():
    value = d_inp.get()
    if value.isdigit():
        d_inp_text_box.delete(1.0, "end-1c")
        d_inp_text_box.insert("end-1c",u'\u2714')
    else: 
        d_inp_text_box.delete(1.0, "end-1c")
        d_inp_text_box.insert("end-1c",u'\u2717')
        
        
def check_q():
    value = q_inp.get()
    if value.isdigit():
        q_inp_text_box.delete(1.0, "end-1c")
        q_inp_text_box.insert("end-1c",u'\u2714')
    else: 
        q_inp_text_box.delete(1.0, "end-1c")
        q_inp_text_box.insert("end-1c",u'\u2717')
        

def check_time():
    value = time_inp.get()
    if value.isdigit():
        time_inp_text_box.delete(1.0, "end-1c")
        time_inp_text_box.insert("end-1c",u'\u2714')
    else:
        time_inp_text_box.delete(1.0, "end-1c")
        time_inp_text_box.insert("end-1c",u'\u2717')
        
# In[3]:
    


def getInput():
    

    sub_button_text.set("Running...")
    tk.messagebox.showinfo(title = "Applet", message = "Program Starts after pressing OK")
    
    try: 
        df = pd.read_csv(str(dataframe_path.get()), index_col=0)
    except:
        df = pd.read_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/CapstoneMainDF.csv", index_col=0)
        
    try:
        productsdf = pd.read_csv(str(productdf_path.get()), index_col=0)
    except:
        productsdf = pd.read_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/ProductsDF.csv", index_col=0)
        
    df.columns = map(str.lower, df.columns)
    
    productdict = dict(zip(productsdf.Item_ID, productsdf.Item_Desc))
    df["product_name"] = df.item_id
    df.product_name = df.product_name.map(productdict)
    df['date'] = pd.to_datetime(df['date'])

    #Project Column
    projectdf = df.loc[(df["project"]!="NONE")]
    df = df.loc[(df["project"]=="NONE")]

    # Temprature Imputations
    df['mintemp'].interpolate(method='linear', inplace=True)    
    df['maxtemp'].interpolate(method='linear', inplace=True) 

    #outlier Removal
    # Write REGEX TO Improve automatic mistakes imputation
    df["price"] = df["sales"] / df["quantity_sold"]
    df.loc[df.product_name == "TTS TOOTSIE ROLL $.10", "price"] = .10
    df.sales = df.quantity_sold * df.price

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

    
    th_std =(np.std(x.Sales_Difference)*int(std_dev_cl.get())+np.mean(x.Sales_Difference))
    temp = x

    temp2 = temp.loc[(temp['Sales_Difference']>=th_std)].sort_values("Sales_Difference",ascending=False)


    
      
    # Take STD.dev
    site_vector = list(temp2["site_id"])
    date_vector = list(temp2["date"])
    index_list = []

    df.reset_index(inplace=True)

    for i in range(len(site_vector)):
        temp = df.loc[(df['site_id']==site_vector[i]) & (df["date"]==date_vector[i])].sort_values("sales",ascending=False).head(1)  
        temp = temp.drop(columns = ["location_id", "open_date", "sq_footage", "locale", "maxtemp", "mintemp", "fiscal_period", "periodic_gbv", "current_gbv", "mpds"])
        index_list.append(temp.iloc[0,0])


    df.rename(columns={ df.columns[0]: "index_set" }, inplace = True)

    df = df[~df.index_set.isin(index_list)]



    df.drop(["index_set", "location_id","current_gbv", "product_name", "open_date", "fiscal_period", "project", "price"], axis = 1, inplace = True)

    locale_string = list(df.locale.unique())
    locale_replace = list(range(1,len(locale_string)+1))
    df.locale.replace(locale_string, locale_replace, inplace=True)
    
    df = df[["site_id", "item_id", "date", "sq_footage","mpds", "locale", "periodic_gbv", "maxtemp", "mintemp","quantity_sold", "sales"]]




    __df_agg = df.groupby(["item_id"]).agg(total_sales = pd.NamedAgg(column="sales", aggfunc = sum), 
                                     days_sold = pd.NamedAgg(column="sales", aggfunc = "count")).reset_index().sort_values("total_sales", ascending=False)

    item_list = []
    site_list = []
    
    value = item_id_inp.get()
    value_str = str(value)
    if value_str.startswith("-"):
        item_list = [int(item_id_inp.get())]
    elif value.isdigit():
        item_list = list(__df_agg.item_id.head(int(value)))
    else: 
        item_list = list(__df_agg.item_id.unique())
        
    value = site_id_inp.get()
    if value == "":
        site_list = list(df.site_id.unique())
    elif value.isdigit():
        if len(value)==3:
            site_list = [int(site_id_inp.get())]
        else: 
            pass
    else: 
        site_list = list(df.site_id.unique())[0:2]
        
    

    
    date_list = list(df.sort_values("date").date.unique())
    __app_list = []
    prediction_dict = {}
    insamp_prediction_dict={}
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

      
 
    if gridsearch_inp.get() == "Yes":
        if int(p_inp.get())  == 0:
            if int(q_inp.get()) == 0:
                __p_values = [int(p_inp.get()),int(p_inp.get())+1,int(p_inp.get())+2]
                __d_values = [int(d_inp.get()),int(d_inp.get())+1,int(d_inp.get())+2]
                __q_values = [int(q_inp.get()),int(q_inp.get())+1,int(q_inp.get())+2]
            else: 
                __p_values = [int(p_inp.get()),int(p_inp.get())+1,int(p_inp.get())+2]
                __d_values = [int(d_inp.get()),int(d_inp.get())+1,int(d_inp.get())+2]
                __q_values = [int(q_inp.get()),int(q_inp.get())*2,int(q_inp.get())*4]
        elif int(q_inp.get()) == 0:
                __p_values = [int(p_inp.get()),int(p_inp.get())*2,int(p_inp.get())*4]
                __d_values = [int(d_inp.get()),int(d_inp.get())+1,int(d_inp.get())+2]
                __q_values = [int(q_inp.get()),int(q_inp.get())+2,int(q_inp.get())+2]        
        else: 
            __p_values = [int(p_inp.get()),int(p_inp.get())*2,int(p_inp.get())*4]
            __d_values = [int(d_inp.get()),int(d_inp.get())+1,int(d_inp.get())+2]
            __q_values = [int(q_inp.get()),int(q_inp.get())*2,int(q_inp.get())*4]
    else: 
        __p_values = [int(p_inp.get())]
        __d_values = [int(d_inp.get())]
        __q_values = [int(q_inp.get())]
    
    if gridsearch_d_inp.get() == "Yes":
        __d_values = [int(d_inp.get()),int(d_inp.get())+1,int(d_inp.get())+2]
    else:
        __d_values = [int(d_inp.get())]
    
        
    
    __best_score = 1000000000
    __best_cfg = 0
    score_dict={}
    __timeA = time.time()
    __timeB = time.time()
    __prediction_time_frame = int(time_inp.get())

    
    counter = 0
    for __i in range(0,len(item_list)):
        __item_id_value = item_list[__i]
        if __i > 0:
            __timeB = time.time()
            __timediff = __timeB-__timeA
            print("Progress = ", __i,"/",int(len(item_list)),
          " Items Done | Time Passed = ",round(__timediff,0),
          " Seconds | Time Left = ", round((__timediff/__i)*(len(item_list)-__i),0)," Seconds")
            
        else:
            print("-----Initializing-----")

        for __x in range(0,len(site_list)):
            __site_id_value = site_list[__x]
            
            __df_name = 'df_' +str(__site_id_value)+"_"+str(__item_id_value)
            
            __df_temp = df.loc[(df['site_id']== __site_id_value) & (df['item_id'] == __item_id_value)].reset_index(drop=True).sort_values('date')            
            
            # Get List for Imputation
            __index_pos = date_list.index(__df_temp.date.unique()[0])
            __trunc_date_list = list(date_list[__index_pos:len(date_list)])
            
            __diff_list = list(set(__trunc_date_list)-set(list(__df_temp.date.unique())))
            
            if len(__diff_list) > 0:
                
                for __y in range(0, len(__diff_list)):
                    __app_list = []
                    __app_list.extend((__site_id_value, __item_id_value, __diff_list[__y], 
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
            __test_temp = __df_temp.tail(__prediction_time_frame)
            __df_temp_bu = __df_temp
            __df_temp = __df_temp.drop(__df_temp.tail(__prediction_time_frame).index)
            
            
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

                                __model = ARIMA(endog = __df_temp['sales'],order = __order)
                                __model_fit = __model.fit()
                                __output = __model_fit.predict(start=len(__df_temp), end = int(len(__df_temp)+__prediction_time_frame-1),dynamic=False) 
                                __rmse = sqrt(mean_squared_error(__output, __test_temp.sales))
                                counter = counter + 1
                                print("Progress ", counter, "models ran of ", len(__p_values)*len(__d_values)*len(__q_values)*len(item_list)*len(site_list))
                                if __rmse < __best_score:
                                    __best_score = __rmse
                                    __best_cfg = __order
                                    insamp_prediction_dict[__site_id_value,__item_id_value] = __output
                                else:
                                    pass
                            except:
                                tk.messagebox.showwarning(title = 'Outcome',message = 'ARIMA FAILED')
                                break
                if __best_cfg == (0,0,0):
                    __best_cfg = __order
                else: 
                    pass     
                
                # In_sample Prep                    
                __model = ARIMA(exog=__df_temp[['maxtemp', 'mintemp']],endog = __df_temp['sales'],order = __best_cfg)
                __model_fit = __model.fit()
                __output = __model_fit.predict(start=len(__df_temp), end = int(len(__df_temp)+__prediction_time_frame-1),dynamic=False, exog = __test_temp[['maxtemp', 'mintemp']])
                model_sum_dict[__site_id_value,__item_id_value] = __model_fit.summary()
                test_dict[__site_id_value,__item_id_value] = __test_temp.sales
                score_dict[__site_id_value,__item_id_value,"cfg"] = __best_cfg
                score_dict[__site_id_value,__item_id_value,"rmse"] = __best_score
                score_dict[__site_id_value,__item_id_value,"sum"] = ceil(sum(__test_temp.sales))
                
                # Out_of_Sample Prediction
                __model = ARIMA(endog = __df_temp_bu['sales'],order = __best_cfg)
                __model_fit = __model.fit()
                __output = __model_fit.predict(start=len(__df_temp_bu), end = int(len(__df_temp_bu)+__prediction_time_frame-1),dynamic=False)
                model_sum_dict[__site_id_value,__item_id_value] = __model_fit.summary()
                test_dict[__site_id_value,__item_id_value] = __test_temp.sales


                
                prediction_dict[__site_id_value,__item_id_value] = __output
            
  
    out_list = []
    if save_inp.get() == "Yes":
        (pd.DataFrame.from_dict(data=prediction_dict, orient='index').to_csv('prediction_output.csv', header=False))
    else:
        pass
    
    


    for i in item_list:
        for x in site_list:
            sum_val = ceil(sum(prediction_dict[x,i]))
            sum_val2 = score_dict[x,i,"sum"]
            val3 = ceil(sum(insamp_prediction_dict[x,i]))-sum_val2
            item_name = productdict[i]
            string_output = "Item_ID = " + str(i) + " | Site = "+str(x)+ "\n" +  "Item_Name = " + str(item_name) + "\n" +"---Prediction---" + "\n" +prediction_dict[x,i].to_string() + "\n" + "----------------" + "\n" +"Sum of Sales predicted 10 days = " + str(sum_val) + "\n" + "Expected Sum of Sales based on previous 10 Days = " + str(sum_val2) + "\n" + "Insample Total Error = " + str(val3) + "\n" + "Best In-Sample CFG = " + str(score_dict[x,i,"cfg"]) + "\n" + "Best In-Sample RMSE = " + str(round(score_dict[x,i,"rmse"],2)) + "\n" + "\n" 
            out_list.append(string_output)
            
    if save_inp.get() == "Yes":
        message_string = "Predictions Saved at: " + str(os.getcwd()) + "/" +'prediction_output.csv'
        tk.messagebox.showinfo(title = "Outcome", message = message_string)
    else:
        pass
    
    if output_inp.get() == "Yes":
        if len(out_list) == 0:
            tk.messagebox.showwarning(title = 'Outcome',message = 'ARIMA FAILED')
        else:
            for i in range(0,int(ceil(len(out_list)/3))):
                x = (len(out_list)%3)*3
                z = floor(len(out_list)/3)
                o = int(ceil(len(out_list)/3))-1     
                title_string = "Outcome Screen " + str(i+1) + " of " + str(o+1)
                if i == o:
                    tk.messagebox.showinfo(title = title_string, message = out_list[i*3:i*3+x])
                else:
                    tk.messagebox.showinfo(title = title_string, message = out_list[i*3:i*3+3])
    else:
        pass
    
    sub_button_text.set("Submit and Run")

                

    
    
            
    
# In[4]:

app = tk.Tk() 
app.geometry('700x600')
app.title('Maverik: Candy Bar prediction')

s=ttk.Style()
s.theme_use('alt')

sub_button_text = tk.StringVar()
sub_button_text.set("Submit and Run")

#Explanation
logo = tk.PhotoImage(file="C:/Users/wille/Documents/GitHub/CapstoneMSBA2020/Resources/logo_maverick.gif",master=app)

label00 = tk.Label(app,text = "")
label00.grid(column=0, row=0)

w1 = tk.Label(app, image=logo).grid(column=2, row=1, columnspan = 2)

explanation = """Fill in the requested inputs. 
Use the buttons for free input to check if the data is correct.
The Program can be killed by using the X on the top right of the Dialogue box."""

w2 = tk.Label(app, 
              justify=tk.LEFT,
              padx = 20, 
              text=explanation).grid(column=0, row=1, columnspan = 2)

# Padding
label0 = tk.Label(app,text = "")
label0.grid(column=0, row=2)

# Dataframe
tk.Label(app, text="Input DataFrame .CSV Path").grid(row=3)
dataframe_path = tk.Entry(app)
dataframe_path.grid(row=3, column=1)

submit_btn1 = Button(app, text="Check", width=10, command=check_dataframe_path)
submit_btn1.grid(row=3, column=2)

dataframe_path_text_box = tk.Text(app, width = 15, height = 1)
dataframe_path_text_box.grid(row = 3, column = 3)
dataframe_path_text_box.insert("end-1c","waiting")

#Product_df
tk.Label(app, text="Input Productdf .CSV Path").grid(row=4)
productdf_path = tk.Entry(app)
productdf_path.grid(row=4, column=1)

submit_btn2 = Button(app, text="Check", width=10, command=check_productdf_path)
submit_btn2.grid(row=4, column=2)

productdf_path_text_box = tk.Text(app, width = 15, height = 1)
productdf_path_text_box.grid(row = 4, column = 3)
productdf_path_text_box.insert("end-1c","waiting")

# Padding
label1 = tk.Label(app,text = "")
label1.grid(column=0, row=5)

# Cleaning_Std
tk.Label(app, text="Input Outlier Std. Deviation").grid(row=6)
std_dev_cl = tk.Entry(app)
std_dev_cl.grid(row=6, column=1)

submit_btn3 = Button(app, text="Check", width=10, command=check_standard_dev)
submit_btn3.grid(row=6, column=2)

std_dev_cl_text_box = tk.Text(app, width = 15, height = 1)
std_dev_cl_text_box.grid(row = 6, column = 3, columnspan = 1)
std_dev_cl_text_box.insert("end-1c","waiting")

# Item_id
tk.Label(app, text="Input Top N or Item_ID").grid(row=7)
item_id_inp = tk.Entry(app)
item_id_inp.grid(row=7, column=1)

submit_btn4 = Button(app, text="Check", width=10, command=check_item_id)
submit_btn4.grid(row=7, column=2)

item_id_inp_text_box = tk.Text(app, width = 15, height = 1)
item_id_inp_text_box.grid(row = 7, column = 3, columnspan = 1)
item_id_inp_text_box.insert("end-1c","waiting")

# Site_id
tk.Label(app, text="Input Site_ID | Blank for All").grid(row=8)
site_id_inp = tk.Entry(app)
site_id_inp.grid(row=8, column=1)

submit_btn5 = Button(app, text="Check", width=10, command=check_site_id)
submit_btn5.grid(row=8, column=2)

site_id_text_box = tk.Text(app, width = 15, height = 1)
site_id_text_box.grid(row = 8, column = 3, columnspan = 1)
site_id_text_box.insert("end-1c","waiting")


# Padding
label2 = tk.Label(app,text = "")
label2.grid(column=0, row=9)

# P
tk.Label(app, text="Input starting P").grid(row=10)
p_inp = tk.Entry(app)
p_inp.grid(row=10, column=1)

submit_btn6 = Button(app, text="Check", width=10, command=check_p)
submit_btn6.grid(row=10, column=2)

p_inp_text_box = tk.Text(app, width = 15, height = 1)
p_inp_text_box.grid(row = 10, column = 3, columnspan = 1)
p_inp_text_box.insert("end-1c","waiting")

# d
tk.Label(app, text="Input starting D").grid(row=11)
d_inp = tk.Entry(app)
d_inp.grid(row=11, column=1)

submit_btn7 = Button(app, text="Check", width=10, command=check_d)
submit_btn7.grid(row=11, column=2)

d_inp_text_box = tk.Text(app, width = 15, height = 1)
d_inp_text_box.grid(row = 11, column = 3, columnspan = 1)
d_inp_text_box.insert("end-1c","waiting")

# q
tk.Label(app, text="Input starting Q").grid(row=12)
q_inp = tk.Entry(app)
q_inp.grid(row=12, column=1)

submit_btn8 = Button(app, text="Check", width=10, command=check_q)
submit_btn8.grid(row=12, column=2)

q_inp_text_box = tk.Text(app, width = 15, height = 1)
q_inp_text_box.grid(row = 12, column = 3, columnspan = 1)
q_inp_text_box.insert("end-1c","waiting")

# Time
tk.Label(app, text="Input Prediction Time Frame").grid(row=13)
time_inp = tk.Entry(app)
time_inp.grid(row=13, column=1)

submit_btn9 = Button(app, text="Check", width=10, command=check_time)
submit_btn9.grid(row=13, column=2)

time_inp_text_box = tk.Text(app, width = 15, height = 1)
time_inp_text_box.grid(row = 13, column = 3, columnspan = 1)
time_inp_text_box.insert("end-1c","waiting")

#Padding
label5 = tk.Label(app,text = "")
label5.grid(column=0, row=14)

#Input
label4 = tk.Label(app,text = "Gridsearch All?")
label4.grid(column=0, row=15)

gridsearch_inp = ttk.Combobox(app, values=["Yes", "No"])
gridsearch_inp.grid(column=1, row=15)

# Gridsearch 2
label5 = tk.Label(app,text = "Gridsearch D only?")
label5.grid(column=0, row=16)

gridsearch_d_inp = ttk.Combobox(app, values=["Yes", "No"])
gridsearch_d_inp.grid(column=1, row=16)

# Ouput
label5 = tk.Label(app,text = "See Output Promt?")
label5.grid(column=0, row=17)

output_inp = ttk.Combobox(app, values=["Yes", "No"])
output_inp.grid(column=1, row=17)

# Save
label5 = tk.Label(app,text = "Save Prediction?")
label5.grid(column=0, row=18)

save_inp = ttk.Combobox(app, values=["Yes", "No"])
save_inp.grid(column=1, row=18)


## Submit
Submit_btn = tk.Button(app, textvariable=sub_button_text ,command = getInput, height=2, width=20,  bg='#cc0000', fg='white')
Submit_btn.grid(row=21, column=1, columnspan = 2,padx=10, pady=25)

app.mainloop()
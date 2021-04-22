# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:49:28 2021

@author: willem
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Math
from math import ceil, floor, sqrt

# Plotting
import matplotlib.pyplot as plt

# Numpy & Pandas
import numpy as np
import pandas as pd

# Stats models
import statsmodels.api as sm
from matplotlib.pyplot import figure

# Linear Imputation
from scipy.interpolate import interp1d

# Machine Learning
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

# Augmented Dickey Fuller Test
from statsmodels.tsa.stattools import adfuller

# Options
pd.options.display.max_rows = 2500

import warnings

warnings.filterwarnings("ignore", "statsmodels.tsa.arima_model.ARMA", FutureWarning)
warnings.filterwarnings("ignore", "statsmodels.tsa.arima_model.ARIMA", FutureWarning)
warnings.filterwarnings("ignore")

import os
import time
import tkinter as tk
import urllib
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.ttk import Separator, Style
import tkinter.font as tkFont
import sys


# In[2]:
def check_dataframe_path():
    value = __df_folder_path.get()
    value = str(value)
    try:
        pd.read_csv(value, index_col=0)
        __dataframe_path_text_box.delete(1.0, "end-1c")
        __dataframe_path_text_box.insert("end-1c", u"\u2714")
    except:
        __dataframe_path_text_box.delete(1.0, "end-1c")
        __dataframe_path_text_box.insert("end-1c", u"\u2717")


def check_productdf_path():
    value = __df_folder_path2.get()
    value = str(value)
    try:
        pd.read_csv(value, index_col=0)
        __productdf_path_text_box.delete(1.0, "end-1c")
        __productdf_path_text_box.insert("end-1c", u"\u2714")
    except:
        __productdf_path_text_box.delete(1.0, "end-1c")
        __productdf_path_text_box.insert("end-1c", u"\u2717")


def check_standard_dev():
    value = __std_dev_inp.get()
    if value.isdigit():
        __std_dev_inp_text_box.delete(1.0, "end-1c")
        __std_dev_inp_text_box.insert("end-1c", u"\u2714")
    else:
        __std_dev_inp_text_box.delete(1.0, "end-1c")
        __std_dev_inp_text_box.insert("end-1c", u"\u2717")


def check_item_id():
    value = __item_id_inp.get()
    value_str = str(value)
    if ("," in str(value)) and (value_str.startswith("-")):
        __item_id_inp_text_box.delete(1.0, "end-1c")
        __item_id_inp_text_box.insert("end-1c", "Item_ID List \u2714")
    elif value.isdigit():
        __item_id_inp_text_box.delete(1.0, "end-1c")
        __item_id_inp_text_box.insert("end-1c", "Top_N \u2714")
    elif value_str.startswith("-"):
        __item_id_inp_text_box.delete(1.0, "end-1c")
        __item_id_inp_text_box.insert("end-1c", "Item_ID \u2714")
    else:
        __item_id_inp_text_box.delete(1.0, "end-1c")
        __item_id_inp_text_box.insert("end-1c", u"\u2717")


def check_site_id():
    value = __site_id_inp.get()
    if value == "":
        __site_id_text_box.delete(1.0, "end-1c")
        __site_id_text_box.insert("end-1c", "All_Sites \u2714")
    elif value.isdigit():
        if len(value) == 3:
            __site_id_text_box.delete(1.0, "end-1c")
            __site_id_text_box.insert("end-1c", "Site_ID \u2714")
        else:
            pass
    elif "," in str(value):
        __site_id_text_box.delete(1.0, "end-1c")
        __site_id_text_box.insert("end-1c", "Site_ID List \u2714")        
    else:
        __site_id_text_box.delete(1.0, "end-1c")
        __site_id_text_box.insert("end-1c", u"\u2717")


def check_time():
    value = __time_inp.get()
    if value.isdigit():
        __time_inp_text_box.delete(1.0, "end-1c")
        __time_inp_text_box.insert("end-1c", u"\u2714")
    else:
        __time_inp_text_box.delete(1.0, "end-1c")
        __time_inp_text_box.insert("end-1c", u"\u2717")
        
def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global __df_folder_path
    __filename = filedialog.askopenfilename()
    __df_folder_path.set(str(__filename))
    __browse_button_text.set(str(__filename.rsplit('/', 1)[-1]))

    
def browse_button2():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global __df_folder_path2
    __filename2 = filedialog.askopenfilename()
    __df_folder_path2.set(str(__filename2))
    __browse_button_text2.set(str(__filename2.rsplit('/', 1)[-1]))

def duplicate_remover(x):
    return list(dict.fromkeys(x))



# In[3]:


def getInput():

    __sub_button_text.set("Running...")
    tk.messagebox.showinfo(title="Applet", message="Program Starts after pressing OK")
    
    #urllib.request.urlretrieve("https://www.dropbox.com/s/gy5n1dvzwoiepvz/CapstoneMainDF.csv?dl=1","CapstoneMainDF.csv")
    
    #urllib.request.urlretrieve("https://www.dropbox.com/s/pdgo1c2u3575pee/ProductsDF.csv?dl=1","ProductsDF.csv")

    try:
        df = pd.read_csv(str(__df_folder_path.get()), index_col=0)
    except:
        df = pd.read_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/CapstoneMainDF.csv", index_col=0)
        #raise

    try:
        productsdf = pd.read_csv(str(__df_folder_path2.get()), index_col=0)
    except:
        #raise
        productsdf = pd.read_csv("C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/ProductsDF.csv", index_col=0)
        
    df.columns = map(str.lower, df.columns)

    productdict = dict(zip(productsdf.Item_ID, productsdf.Item_Desc))
    df["product_name"] = df.item_id
    df.product_name = df.product_name.map(productdict)
    df["date"] = pd.to_datetime(df["date"])

    # Project Column
    projectdf = df.loc[(df["project"] != "NONE")]
    df = df.loc[(df["project"] == "NONE")]

    # Temprature Imputations
    df["mintemp"].interpolate(method="linear", inplace=True)
    df["maxtemp"].interpolate(method="linear", inplace=True)

    # outlier Removal
    # Write REGEX TO Improve automatic mistakes imputation
    df["price"] = df["sales"] / df["quantity_sold"]
    df.loc[df.product_name == "TTS TOOTSIE ROLL $.10", "price"] = 0.10
    df.sales = df.quantity_sold * df.price

    x = (
        df.groupby(["date", "site_id"])
        .agg(
            daily_sales=pd.NamedAgg(column="sales", aggfunc=sum),
            daily_quantity=pd.NamedAgg(column="quantity_sold", aggfunc=sum),
        )
        .reset_index()
    )
    x["date"] = pd.to_datetime(x["date"])
    x["price"] = (x["daily_sales"] / x["daily_quantity"]).round(2)
    z = (
        df.groupby(["date", "site_id"])
        .agg(total_sales=pd.NamedAgg(column="sales", aggfunc=sum))
        .reset_index()
    )
    z = (
        z.groupby(["site_id"])
        .agg(average_sales=pd.NamedAgg(column="total_sales", aggfunc="mean"))
        .reset_index()
    )
    salesdict = dict(zip(z.site_id, z.average_sales))
    x["average_sales"] = x.site_id
    x.average_sales = x.site_id.map(salesdict)
    x["Sales_Difference"] = np.absolute(
        ((x["daily_sales"] - x["average_sales"]) / x["average_sales"]) * 100
    ).round(2)
    x = x.sort_values("Sales_Difference", ascending=False)

    th_std = np.std(x.Sales_Difference) * int(__std_dev_inp.get()) + np.mean(
        x.Sales_Difference
    )
    temp = x

    temp2 = temp.loc[(temp["Sales_Difference"] >= th_std)].sort_values(
        "Sales_Difference", ascending=False
    )

    # Take STD.dev
    site_vector = list(temp2["site_id"])
    date_vector = list(temp2["date"])
    index_list = []

    df.reset_index(inplace=True)

    for i in range(len(site_vector)):
        temp = (
            df.loc[(df["site_id"] == site_vector[i]) & (df["date"] == date_vector[i])]
            .sort_values("sales", ascending=False)
            .head(1)
        )
        temp = temp.drop(
            columns=[
                "location_id",
                "open_date",
                "sq_footage",
                "locale",
                "maxtemp",
                "mintemp",
                "fiscal_period",
                "periodic_gbv",
                "current_gbv",
                "mpds",
            ]
        )
        index_list.append(temp.iloc[0, 0])

    df.rename(columns={df.columns[0]: "index_set"}, inplace=True)

    df = df[~df.index_set.isin(index_list)]

    df.drop(
        [
            "index_set",
            "location_id",
            "current_gbv",
            "product_name",
            "open_date",
            "fiscal_period",
            "project",
            "price",
        ],
        axis=1,
        inplace=True,
    )

    locale_string = list(df.locale.unique())
    locale_replace = list(range(1, len(locale_string) + 1))
    df.locale.replace(locale_string, locale_replace, inplace=True)

    df = df[
        [
            "site_id",
            "item_id",
            "date",
            "sq_footage",
            "mpds",
            "locale",
            "periodic_gbv",
            "maxtemp",
            "mintemp",
            "quantity_sold",
            "sales",
        ]
    ]

    __df_agg = (
        df.groupby(["item_id"])
        .agg(
            total_sales=pd.NamedAgg(column="sales", aggfunc=sum),
            days_sold=pd.NamedAgg(column="sales", aggfunc="count"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )

    item_list = []
    site_list = []

    value = __item_id_inp.get()
    value_str = str(value)
    if ("," in str(value)) and (value_str.startswith("-")):
        value_str = value_str.replace(" ", "")
        item_list = value_str.split(",")
        for __i in range(0, len(item_list)):
           item_list[__i] = int(item_list[__i])
    elif value.isdigit():
        item_list = list(__df_agg.item_id.head(int(value)))
        
    elif value_str.startswith("-"):
        item_list = [int(__item_id_inp.get())]
    else:
        item_list = list(__df_agg.item_id.unique())


    value = __site_id_inp.get()
    if value == "":
        site_list = list(df.site_id.unique())
    elif value.isdigit():
        if len(value) == 3:
            site_list = [int(__site_id_inp.get())]
        else:
            pass
    elif "," in str(value):
       site_list = str(value).replace(" ", "").split(",")
       for __i in range(0, len(site_list)):
           site_list[__i] = int(site_list[__i])


    date_list = list(df.sort_values("date").date.unique())
    __app_list = []
    prediction_dict = {}
    holdout_prediction_dict = {}
    test_dict = {}
    model_sum_dict = {}

    # Create a dictionary for none changing values
    df_dict = {}

    for __i in site_list:
        df_dict[int(__i)] = []

    for __i in range(0, len(site_list)):

        __site_id_value = site_list[__i]

        __df_temp = (
            df.loc[(df["site_id"] == __site_id_value)]
            .reset_index(drop=True)
            .sort_values("date")
        )

        df_dict[__site_id_value].append(int(__df_temp.sq_footage.mode()))
        df_dict[__site_id_value].append(int(__df_temp.mpds.mode()))
        df_dict[__site_id_value].append(int(__df_temp.locale.mode()))
        df_dict[__site_id_value].append(int(__df_temp.periodic_gbv.mode()))
        
    
    __p_values = []
    __d_values = []
    __q_values = []
    
    value = __p_gs_list_inp.get()
    value_str = str(value)
    if "," in str(value):
        value_str = value_str.replace(" ", "")
        __p_gs_list = value_str.split(",")
        for __i in range(0, len(__p_gs_list)):
           __p_gs_list[__i] = int(__p_gs_list[__i])       
    else:
        __p_gs_list = [0,1,2]
        
    value = __d_gs_list_inp.get()
    value_str = str(value)
    if "," in str(value):
        value_str = value_str.replace(" ", "")
        __d_gs_list = value_str.split(",")
        for __i in range(0, len(__d_gs_list)):
           __d_gs_list[__i] = int(__d_gs_list[__i])       
    else:
        __d_gs_list = [0,1,2]
        
    value = __q_gs_list_inp.get()
    value_str = str(value)
    if "," in str(value):
        value_str = value_str.replace(" ", "")
        __q_gs_list = value_str.split(",")
        for __i in range(0, len(__q_gs_list)):
           __q_gs_list[__i] = int(__q_gs_list[__i])       
    else:
        __q_gs_list = [0,1,2]


    if __gridsearch_inp.get() == "Yes":
        if int(__p_inp.get()) == 0:
            if int(__q_inp.get()) == 0:
                for __p in __p_gs_list:
                    __p_values.append(int(int(__p_inp.get())+int(__p)))
                for __d in __d_gs_list:
                    __d_values.append(int(int(__d_inp.get())+int(__d)))
                for __q in __q_gs_list:
                    __q_values.append(int(int(__q_inp.get())+int(__q)))
            else:
                for __p in __p_gs_list:
                    __p_values.append(int(int(__p_inp.get())+int(__p)))
                for __d in __d_gs_list:
                    __d_values.append(int(int(__d_inp.get())+int(__d)))
                for __q in __q_gs_list:
                    __q_values.append(int(int(__q_inp.get())*int(__q)))
        elif int(__q_inp.get()) == 0:
            for __p in __p_gs_list:
                __p_values.append(int(int(__p_inp.get())*int(__p)))
            for __d in __d_gs_list:
                __d_values.append(int(int(__d_inp.get())+int(__d)))
            for __q in __q_gs_list:
                __q_values.append(int(int(__q_inp.get())+int(__q)))
        else:
            for __p in __p_gs_list:
                __p_values.append(int(int(__p_inp.get())*int(__p)))
            for __d in __d_gs_list:
                __d_values.append(int(int(__d_inp.get())+int(__d)))
            for __q in __q_gs_list:
                __q_values.append(int(int(__q_inp.get())*int(__q)))
    else:
        __p_values = [int(__p_inp.get())]
        __d_values = [int(__d_inp.get())]
        __q_values = [int(__q_inp.get())]

    if __gridsearch_d_inp.get() == "Yes":
        for __d in __d_gs_list:
            __d_values.append(int(int(__d_inp.get())+int(__d)))
    else:
        __d_values = [int(__d_inp.get())]
        
    __p_values = duplicate_remover(__p_values)
    __d_values = duplicate_remover(__d_values)
    __q_values = duplicate_remover(__q_values)


    __best_score = 1000000000
    __best_cfg = 0
    score_dict = {}
    __timeA = time.time()
    __timeB = time.time()
    __prediction_time_frame = int(__time_inp.get())
    __site_counter = 0
    trend_error = 0
    array_error = 0
    combination_error = 0
    error_dict = {} 
    
    counter = 0
    for __i in range(0, len(item_list)):
        __item_id_value = item_list[__i]
        if __i > 0:
            __timeB = time.time()
            __timediff = __timeB - __timeA
            print(
                "Progress = ",
                __i,
                "/",
                int(len(item_list)),
                " Items Done | Time Passed = ",
                round(__timediff, 0),
                " Seconds | Time Left = ",
                round((__timediff / __i) * (len(item_list) - __i), 0),
                " Seconds",
            )

        else:
            print("-----Initializing-----")
            
        for __x in range(0, len(site_list)):
            __site_id_value = site_list[__x]

            __df_temp = (
                df.loc[
                    (df["site_id"] == __site_id_value)
                    & (df["item_id"] == __item_id_value)
                ]
                .reset_index(drop=True)
                .sort_values("date")
            )

            # Get List for Imputation
            try:
                __index_pos = date_list.index(__df_temp.date.unique()[0])
            except:
                combination_error=1
                error_dict[__item_id_value, __site_id_value] = "Combination Does not Exist"
                continue
              
            __trunc_date_list = list(date_list[__index_pos : len(date_list)])

            __diff_list = list(
                set(__trunc_date_list) - set(list(__df_temp.date.unique()))
            )

            if len(__diff_list) > 0:

                for __y in range(0, len(__diff_list)):
                    __app_list = []
                    __app_list.extend(
                        (
                            __site_id_value,
                            __item_id_value,
                            __diff_list[__y],
                            df_dict[__site_id_value][0],
                            df_dict[__site_id_value][1],
                            df_dict[__site_id_value][2],
                            df_dict[__site_id_value][3],
                            np.nan,
                            np.nan,
                            0,
                            0,
                        )
                    )
                    __app_series = pd.Series(__app_list, index=__df_temp.columns)
                    __df_temp = __df_temp.append(__app_series, ignore_index=True)

            else:
                pass

            __df_temp = __df_temp.sort_values("date")
            __df_temp["mintemp"].interpolate(method="linear", inplace=True)
            __df_temp["maxtemp"].interpolate(method="linear", inplace=True)
            __df_temp["dateind"] = __df_temp["date"]
            __df_temp.set_index("dateind", inplace=True)
            __df_temp.index = pd.DatetimeIndex(
                __df_temp.index.values, freq=__df_temp.index.inferred_freq
            )
            __test_temp = __df_temp.tail(__prediction_time_frame)
            __df_temp_bu = __df_temp
            __df_temp = __df_temp.drop(__df_temp.tail(__prediction_time_frame).index)
            
            # Item and Site Combination Age Check
            if int(len(__df_temp)) < int(__time_inp.get()):
                if __warning_inp.get() == 'Yes':                
                    message_list = (
                        "Not enough data For Item_ID = " 
                        + str(__item_id_value) 
                        + " & Site_ID = " 
                        + str(__site_id_value) 
                        + "\n" 
                        + "Total Data Points = " 
                        + str(len(__df_temp_bu)) 
                        + "\n" 
                        + "Not enough histrorical data present for this combination"
                        + "\n"
                        + "No Prediction can be made at this time"
                        )
                    
                    tk.messagebox.showerror(title="Warning", message=message_list )
                    __sub_button_text.set("Submit and Run")
                    
                    array_error = 1
                    error_dict[__item_id_value, __site_id_value] = "No Historical Data | Item too new for the store"
                elif __warning_inp.get() != 'Yes': 
                        array_error = 1
                        error_dict[__item_id_value, __site_id_value] = "No Historical Data | Item too new for the store"
                        
                        
                continue
            else:
                pass
            

            # Gridsearching
            __best_score = 1000000000
            __best_cfg = (0, 0, 0)

            for __p in __p_values:
                for __d in __d_values:
                    for __q in __q_values:
                        __order = (__p, __d, __q)
                        try:
                            
                            __model = ARIMA(endog=__df_temp["quantity_sold"], order=__order)
                            __model_fit = __model.fit()
                            __output = __model_fit.predict(
                                start=len(__df_temp),
                                end=int(
                                    len(__df_temp) + __prediction_time_frame - 1
                                ),
                                dynamic=False,
                            )
                            __rmse = sqrt(
                                mean_squared_error(__test_temp.quantity_sold,__output)
                            )
                            counter = counter + 1
                            print(
                                "Progress ",
                                counter,
                                "models completed out of ",
                                len(__p_values)
                                * len(__d_values)
                                * len(__q_values)
                                * len(item_list)
                                * __site_counter,
                            )
                            
                            if __rmse < __best_score:
                                __best_score = __rmse
                                __best_cfg = __order
                                holdout_prediction_dict[
                                    __site_id_value, __item_id_value
                                ] = __output
                            else:
                                pass
                        except ValueError as VE:
                            if __warning_inp.get() == "Yes":
                                if "A constant trend was included in the model" in str(VE):
                                    tk.messagebox.showerror(
                                        title="ERROR", message="""Trend Included in the Model. Allow for gridsearching of Parameter [d]"""
                                        )
                                else:    
                                        pass
                            else:    
                                pass

            if __best_cfg == (0, 0, 0):
                __best_cfg = __order
            else:
                pass

            # Main prediction
            __model = ARIMA(
                endog=__df_temp["quantity_sold"],
                order=__best_cfg,
                )
            try:
                __model = ARIMA(
                    endog=__df_temp["quantity_sold"],
                    order=__best_cfg,
                    )
            except ValueError as VE:
                if __warning_inp.get() == "Yes":
                    if "A constant trend was included in the model" in str(VE):
                        message_List = ("Trend Included in the Model." + "\n" + "Allow for gridsearching of Parameter [d]" + "\n" + "\n" + "----- Model Quitting -----")
                        tk.messagebox.showerror(
                            title="ERROR", message= message_List
                            )
                        __sub_button_text.set("Submit and Run")
                        continue
                    elif "zero-size array" in str(VE):
                        message_List = ("Zero-Size Array Error" + "\n" + "Not enough histrorical data present" + "\n" + "\n" + "----- Model Quitting -----")
                        tk.messagebox.showerror(
                            title="ERROR", message="""Trend Included in the Model.
                            Allow for gridsearching of Parameter [d]"""
                            )
                        __sub_button_text.set("Submit and Run")
                        continue
                elif __warning_inp.get()!= "Yes":
                    if "A constant trend was included in the model" in str(VE):       
                        trend_error = 1
                        error_dict[__site_id_value, __item_id_value] = "Trend Error | Allow for Gridsearching of d"
                    elif "zero-size array" in str(VE):
                        array_error = 1
                        error_dict[__site_id_value, __item_id_value] = "Array Error | Item too new to predict"                                 
                else:
                    continue
                    
            __model_fit = __model.fit()
            __output = __model_fit.predict(
                start=len(__df_temp),
                end=int(len(__df_temp) + __prediction_time_frame - 1),
                dynamic=False,
            )
            model_sum_dict[__site_id_value, __item_id_value] = __model_fit.summary()
            test_dict[__site_id_value, __item_id_value] = __test_temp.quantity_sold
            score_dict[__site_id_value, __item_id_value, "cfg"] = str(__best_cfg)
            score_dict[__site_id_value, __item_id_value, "rmse"] = round(__best_score,2)
            score_dict[__site_id_value, __item_id_value, "sum"] = ceil(
                sum(__test_temp.quantity_sold)
            )

            # Out_of_Sample Prediction
            __model = ARIMA(endog=__df_temp_bu["quantity_sold"], order=__best_cfg)
            __model_fit = __model.fit()
            __output = __model_fit.predict(
                start=len(__df_temp_bu),
                end=int(len(__df_temp_bu) + __prediction_time_frame - 1),
                dynamic=False,
            )
            model_sum_dict[__site_id_value, __item_id_value] = __model_fit.summary()
            test_dict[__site_id_value, __item_id_value] = __test_temp.quantity_sold

            prediction_dict[__site_id_value, __item_id_value] = round(__output,2)

    out_list = []
    
    if __save_inp.get() == "Yes":
        
            pd.DataFrame.from_dict(data=prediction_dict, orient="index").to_csv("prediction_output.csv", header=True)
            pd.DataFrame.from_dict(data=prediction_dict, orient="index").to_csv("test_data.csv", header=True)
            pd.DataFrame.from_dict(data=score_dict, orient="index").to_csv("scores_output.csv", header=True)
                  
    else:
        pass

    for i in item_list:
        for x in site_list:
            try:
                sum_val = ceil(sum(prediction_dict[x, i]))
                sum_val2 = score_dict[x, i, "sum"]
                val3 = ceil(sum(holdout_prediction_dict[x, i])) - sum_val2
                item_name = productdict[i]
                string_output = str(
                    "Item_ID = "
                    + str(i)
                    + " | Site = "
                    + str(x)
                    + "\n"
                    + "Item_Name = "
                    + str(item_name)
                    + "\n"
                    + "---Prediction---"
                    + "\n"
                    + prediction_dict[x, i].to_string()
                    + "\n"
                    + "----------------"
                    + "\n"
                    + "Sum of Quantity predicted "
                    + str(__time_inp.get())
                    + " Days = "
                    + str(sum_val)
                    + "\n"
                    + "Expected Sum of Quantity based on previous "
                    + str(__time_inp.get())
                    + " Days = "
                    + str(sum_val2)
                    + "\n"
                    + "OOS Total Error = "
                    + str(val3)
                    + "\n"
                    + "Best OOS Config = "
                    + str(score_dict[x, i, "cfg"])
                    + "\n"
                    + "Best OOS RMSE = "
                    + str(round(score_dict[x, i, "rmse"], 2))
                    + "\n"
                    + "\n"
                    )
                out_list.append(string_output)
            except:
                continue
    
    
    if len(error_dict) > 0:
        pd.DataFrame.from_dict(data=error_dict, orient="index").to_csv("error_output.csv", header=True)
    else: 
        pass

    if __save_inp.get() == "Yes":
        message_string = (
            "Predictions and Scores Saved at: " + str(os.getcwd())
        )
        tk.messagebox.showinfo(title="Outcome", message=message_string)
    else:
        pass
    
    
    if (trend_error == 1) & (array_error == 1):
         message_string = (
             "Trend and Array Errors Encountered | Errors Saved at : " + "\n" + str(os.getcwd()))
         tk.messagebox.showwarning(
             title="Error Warning", message= message_string
             )
    elif trend_error == 1:
        message_string = (
            "Trend Errors Encountered | Errors Saved at : " + "\n" + str(os.getcwd()))
        tk.messagebox.showwarning(
                title="Error Warning", message= message_string
                )                
        
    elif array_error == 1:
        message_string = (
            "Array Errors Encountered | Errors Saved at : " + "\n" + str(os.getcwd()))
        tk.messagebox.showwarning(
                title="Error Warning", message= message_string
                )
    else:
        pass
    

    if __output_inp.get() == "Yes":
        if len(out_list) == 0:
            tk.messagebox.showerror(
                title="ERROR", message="Model Failed for all Item_Id and Site_ID combinations"
                )
        else:
            for i in range(0, int(ceil(len(out_list) / 3))):
                x = (len(out_list) % 3) * 3
                z = floor(len(out_list) / 3)
                o = int(ceil(len(out_list) / 3)) - 1
                title_string = "Outcome Screen " + str(i + 1) + " of " + str(o + 1)
                if i == o:
                    tk.messagebox.showinfo(
                        title=title_string, message=out_list[i * 3 : i * 3 + x]
                    )
                else:
                    tk.messagebox.showinfo(
                        title=title_string, message=out_list[i * 3 : i * 3 + 3]
                    )
    else:
        pass
    __sub_button_text.set("Submit and Run")
    
    


# In[4]:

__app = tk.Tk()
__app.geometry()
__app.title("Maverik: Candy Bar prediction")

__s = ttk.Style()
__s.theme_use("alt")

__sub_button_text = tk.StringVar()
__sub_button_text.set("Submit and Run")

__browse_button_text = tk.StringVar()
__browse_button_text.set("Browse")

__browse_button_text2 = tk.StringVar()
__browse_button_text2.set("Browse")

__header_fontstyle = tkFont.Font(size=15)
__header_fontstyle.configure(underline=True)


sep_ver = Separator(__app, orient="vertical")

urllib.request.urlretrieve("https://www.dropbox.com/s/hph99elpmvwjjjl/logo_maverick.gif?dl=1", 
                   "logo_maverick.gif")

__df_folder_path = StringVar()
__df_folder_path2 = StringVar()

# Explanation
__logo = tk.PhotoImage(
    file="logo_maverick.gif",
    master=__app,
)

__label00 = tk.Label(__app, text="")
__label00.grid(column=1, row=0)

w1 = tk.Label(__app, image=__logo).grid(column=4, row=1, columnspan=3)

__explanation = """Fill in all inputs.
check input validity with buttons.
Run program with submit and run.
Program can be killed by closing dialogue."""

w2 = tk.Label(__app, justify=tk.LEFT, padx=20, text=__explanation).grid(
    column=1, row=1, columnspan=3
)

# Padding
__label00 = tk.Label(__app, text="   ")
__label00.grid(column=6, row=0)
__label01 = tk.Label(__app, text="   ")
__label01.grid(column=0, row=0)


sep_hor1 = Separator(__app, orient="horizontal")
sep_hor1.grid(column=1, row=2, columnspan=5, sticky="ew")

# Header
tk.Label(__app, text="Data", font=__header_fontstyle).grid(row=3, column=0, columnspan = 7)

# Dataframe
tk.Label(__app, text="Main Dataframe").grid(row=4, column=1, sticky=E)
__browse_btn_1 = Button(textvariable= __browse_button_text, command = browse_button).grid(row=4,column=2, sticky=EW)

__submit_btn_1 = Button(__app, text="Check", width=10, command=check_dataframe_path)
__submit_btn_1.grid(row=4, column=4)

__dataframe_path_text_box = tk.Text(__app, width=15, height=1)
__dataframe_path_text_box.grid(row=4, column=5)
__dataframe_path_text_box.insert("end-1c", "waiting")

# Product_df
tk.Label(__app, text="Product Dataframe").grid(row=5, column=1, sticky=E)
__browse_btn_2 = Button(textvariable= __browse_button_text2, command = browse_button2).grid(row=5,column=2, sticky=EW)

__submit_btn_2 = Button(__app, text="Check", width=10, command=check_productdf_path)
__submit_btn_2.grid(row=5, column=4)

__productdf_path_text_box = tk.Text(__app, width=15, height=1)
__productdf_path_text_box.grid(row=5, column=5)
__productdf_path_text_box.insert("end-1c", "waiting")

# Padding
__label02 = tk.Label(__app, text="      ")
__label02.grid(column=0, row=6)

sep_hor2 = Separator(__app, orient="horizontal")
sep_hor2.grid(column=1, row=7, columnspan=5, sticky="ew")

__label03 = tk.Label(__app, text="      ")
__label03.grid(column=0, row=8)

# Header
tk.Label(__app, text="General Inputs", font=__header_fontstyle).grid(row=8, column=0, columnspan = 7)

# Item_id
tk.Label(__app, text="Item_ID").grid(row=9, column=1, sticky=E)
__item_id_inp = tk.Entry(__app)
__item_id_inp.insert(END, "10")
__item_id_inp.grid(row=9, column=2)

__submit_btn_4 = Button(__app, text="Check", width=10, command=check_item_id)
__submit_btn_4.grid(row=9, column=4)

__item_id_inp_text_box = tk.Text(__app, width=15, height=1)
__item_id_inp_text_box.grid(row=9, column=5, columnspan=1)
__item_id_inp_text_box.insert("end-1c", "waiting")

# Site_id
tk.Label(__app, text="Site_ID").grid(row=10, column=1, sticky=E)
__site_id_inp = tk.Entry(__app)
__site_id_inp.grid(row=10, column=2)

__submit_btn_5 = Button(__app, text="Check", width=10, command=check_site_id)
__submit_btn_5.grid(row=10, column=4)

__site_id_text_box = tk.Text(__app, width=15, height=1)
__site_id_text_box.grid(row=10, column=5, columnspan=1)
__site_id_text_box.insert("end-1c", "waiting")

# Cleaning_Std
tk.Label(__app, text="Outlier Std.Dev.").grid(row=11, column=1, sticky=E)
__std_dev_inp = tk.Entry(__app)
__std_dev_inp.insert(END, "4")
__std_dev_inp.grid(row=11, column=2)

__submit_btn_3 = Button(__app, text="Check", width=10, command=check_standard_dev)
__submit_btn_3.grid(row=11, column=4)

__std_dev_inp_text_box = tk.Text(__app, width=15, height=1)
__std_dev_inp_text_box.grid(row=11, column=5, columnspan=1)
__std_dev_inp_text_box.insert("end-1c", "waiting")

# Time
tk.Label(__app, text="Days to Predict").grid(row=12, column=1, sticky=E)
__time_inp = tk.Entry(__app)
__time_inp.insert(END, "7")
__time_inp.grid(row=12, column=2)

__submit_btn_9 = Button(__app, text="Check", width=10, command=check_time)
__submit_btn_9.grid(row=12, column=4)

__time_inp_text_box = tk.Text(__app, width=15, height=1)
__time_inp_text_box.grid(row=12, column=5, columnspan=1)
__time_inp_text_box.insert("end-1c", "waiting")


# Padding
__label04 = tk.Label(__app, text="   ")
__label04.grid(column=0, row=13)

sep_hor3 = Separator(__app, orient="horizontal")
sep_hor3.grid(column=1, row=14, columnspan=5, sticky="ew")

ver_hor3 = Separator(__app, orient="horizontal")
ver_hor3.grid(column=3, row=16, rowspan=3, sticky="ns")

# Header
tk.Label(__app, text="Parameter Inputs", font=__header_fontstyle).grid(row=15, column=0, columnspan = 7)

# P
tk.Label(__app, text="Starting p").grid(row=16, column=1, sticky=E)
__p_inp = tk.Entry(__app)
__p_inp.insert(END, "7")
__p_inp.grid(row=16, column=2)


tk.Label(__app, text="p Grid Mult").grid(row=16, column=4, sticky=E)
__p_gs_list_inp = tk.Entry(__app)
__p_gs_list_inp.insert(END, "0,1,2")
__p_gs_list_inp.grid(row=16, column=5)

# d
tk.Label(__app, text="Starting d").grid(row=17, column=1, sticky=E)
__d_inp = tk.Entry(__app)
__d_inp.insert(END, "0")
__d_inp.grid(row=17, column=2)

tk.Label(__app, text="d Grid Sum").grid(row=17, column=4, sticky=E)
__d_gs_list_inp = tk.Entry(__app)
__d_gs_list_inp.insert(END, "0,1,2")
__d_gs_list_inp.grid(row=17, column=5)

# q
tk.Label(__app, text="Starting q").grid(row=18, column=1, sticky=E)
__q_inp = tk.Entry(__app)
__q_inp.insert(END, "1")
__q_inp.grid(row=18, column=2)

tk.Label(__app, text="q Grid Mult").grid(row=18, column=4, sticky=E)
__q_gs_list_inp = tk.Entry(__app)
__q_gs_list_inp.insert(END, "0,1,2")
__q_gs_list_inp.grid(row=18, column=5)

# Padding
__label05 = tk.Label(__app, text="   ")
__label05.grid(column=0, row=19)

sep_hor4 = Separator(__app, orient="horizontal")
sep_hor4.grid(column=1, row=20, columnspan=5, sticky="ew")

# Header
tk.Label(__app, text="Options", font=__header_fontstyle).grid(row=21, column=0, columnspan = 7)

# Input
__label4 = tk.Label(__app, text="Gridsearch All?")
__label4.grid(column=1, row=22, sticky=E)

__gridsearch_inp = ttk.Combobox(__app, values=["Yes", "No"])
__gridsearch_inp.insert(END, "No")
__gridsearch_inp.grid(column=2, row=22)

# Gridsearch 2
__label5 = tk.Label(__app, text="Gridsearch d?")
__label5.grid(column=4, row=22, sticky=E)

__gridsearch_d_inp = ttk.Combobox(__app, values=["Yes", "No"])
__gridsearch_d_inp.insert(END, "Yes")
__gridsearch_d_inp.grid(column=5, row=22)

# Ouput
__label6 = tk.Label(__app, text="Show Promts?")
__label6.grid(column=1, row=23, sticky=E)

__output_inp = ttk.Combobox(__app, values=["Yes", "No"])
__output_inp.insert(END, "Yes")
__output_inp.grid(column=2, row=23)

# Save
__label7 = tk.Label(__app, text="Save Outcome?")
__label7.grid(column=4, row=23, sticky=E)

__save_inp = ttk.Combobox(__app, values=["Yes", "No"])
__save_inp.insert(END, "No")
__save_inp.grid(column=5, row=23)

# Warnings
__label8 = tk.Label(__app, text="Show Warnings?")
__label8.grid(column=1, row=24, sticky=E)

__warning_inp = ttk.Combobox(__app, values=["Yes", "No"])
__warning_inp.insert(END, "Yes")
__warning_inp.grid(column=2, row=24)

# Padding
__label05 = tk.Label(__app, text="   ")
__label05.grid(column=0, row=25, sticky=E)

sep_hor5 = Separator(__app, orient="horizontal")
sep_hor5.grid(column=1, row=26, columnspan=5, sticky="ew")

## Submit
__label06 = tk.Label(__app, text="   ")
__label06.grid(column=0, row=27, sticky=E)

__submit_btn = tk.Button(
    __app,
    textvariable=__sub_button_text,
    command=getInput,
    height=2,
    width=20,
    bg="#cc0000",
    fg="white",
)
__submit_btn.grid(row=28, column=2, columnspan=3, padx=10, pady=25)

# Padding
__label07 = tk.Label(__app, text="")
__label07.grid(column=1, row=29)

sep_hor5 = Separator(__app, orient="horizontal")
sep_hor5.grid(column=1, row=30, columnspan=5, sticky="ew")

__label08 = tk.Label(__app, text="")
__label08.grid(column=1, row=31)


__app.mainloop()




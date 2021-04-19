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
import sys


# In[2]:
def check_dataframe_path():
    value = __dataframe_path.get()
    value = str(value)
    try:
        pd.read_csv(value, index_col=0)
        __dataframe_path_text_box.delete(1.0, "end-1c")
        __dataframe_path_text_box.insert("end-1c", u"\u2714")
    except:
        __dataframe_path_text_box.delete(1.0, "end-1c")
        __dataframe_path_text_box.insert("end-1c", u"\u2717")


def check_productdf_path():
    value = __productdf_path.get()
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
    if value_str.startswith("-"):
        __item_id_inp_text_box.delete(1.0, "end-1c")
        __item_id_inp_text_box.insert("end-1c", "Item_Id \u2714")
    elif value.isdigit():
        __item_id_inp_text_box.delete(1.0, "end-1c")
        __item_id_inp_text_box.insert("end-1c", "Top_N \u2714")
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
            __site_id_text_box.insert("end-1c", "Site_Id \u2714")
        else:
            pass
    else:
        __site_id_text_box.delete(1.0, "end-1c")
        __site_id_text_box.insert("end-1c", u"\u2717")


def check_p():
    value = __p_inp.get()
    if value.isdigit():
        __p_inp_text_box.delete(1.0, "end-1c")
        __p_inp_text_box.insert("end-1c", u"\u2714")
    else:
        __p_inp_text_box.delete(1.0, "end-1c")
        __p_inp_text_box.insert("end-1c", u"\u2717")


def check_d():
    value = __d_inp.get()
    if value.isdigit():
        __d_inp_text_box.delete(1.0, "end-1c")
        __d_inp_text_box.insert("end-1c", u"\u2714")
    else:
        __d_inp_text_box.delete(1.0, "end-1c")
        __d_inp_text_box.insert("end-1c", u"\u2717")


def check_q():
    value = __q_inp.get()
    if value.isdigit():
        __q_inp_text_box.delete(1.0, "end-1c")
        __q_inp_text_box.insert("end-1c", u"\u2714")
    else:
        __q_inp_text_box.delete(1.0, "end-1c")
        __q_inp_text_box.insert("end-1c", u"\u2717")


def check_time():
    value = __time_inp.get()
    if value.isdigit():
        __time_inp_text_box.delete(1.0, "end-1c")
        __time_inp_text_box.insert("end-1c", u"\u2714")
    else:
        __time_inp_text_box.delete(1.0, "end-1c")
        __time_inp_text_box.insert("end-1c", u"\u2717")


# In[3]:


def getInput():

    __sub_button_text.set("Running...")
    tk.messagebox.showinfo(title="Applet", message="Program Starts after pressing OK")

    # urllib.request.urlretrieve("https://www.dropbox.com/s/gy5n1dvzwoiepvz/CapstoneMainDF.csv?dl=1","CapstoneMainDF.csv")

    # urllib.request.urlretrieve("https://www.dropbox.com/s/pdgo1c2u3575pee/ProductsDF.csv?dl=1","ProductsDF.csv")

    try:
        df = pd.read_csv(str(__dataframe_path.get()), index_col=0)
    except:
        raise

    try:
        productsdf = pd.read_csv(str(__productdf_path.get()), index_col=0)
    except:
        raise

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
    if value_str.startswith("-"):
        item_list = [int(__item_id_inp.get())]
    elif value.isdigit():
        item_list = list(__df_agg.item_id.head(int(value)))
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
    else:
        site_list = list(df.site_id.unique())[0:2]

    date_list = list(df.sort_values("date").date.unique())
    __app_list = []
    prediction_dict = {}
    insamp_prediction_dict = {}
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

    if __gridsearch_inp.get() == "Yes":
        if int(__p_inp.get()) == 0:
            if int(__q_inp.get()) == 0:
                __p_values = [
                    int(__p_inp.get()),
                    int(__p_inp.get()) + 1,
                    int(__p_inp.get()) + 2,
                ]
                __d_values = [
                    int(__d_inp.get()),
                    int(__d_inp.get()) + 1,
                    int(__d_inp.get()) + 2,
                ]
                __q_values = [
                    int(__q_inp.get()),
                    int(__q_inp.get()) + 1,
                    int(__q_inp.get()) + 2,
                ]
            else:
                __p_values = [
                    int(__p_inp.get()),
                    int(__p_inp.get()) + 1,
                    int(__p_inp.get()) + 2,
                ]
                __d_values = [
                    int(__d_inp.get()),
                    int(__d_inp.get()) + 1,
                    int(__d_inp.get()) + 2,
                ]
                __q_values = [
                    int(__q_inp.get()),
                    int(__q_inp.get()) * 2,
                    int(__q_inp.get()) * 4,
                ]
        elif int(__q_inp.get()) == 0:
            __p_values = [
                int(__p_inp.get()),
                int(__p_inp.get()) * 2,
                int(__p_inp.get()) * 4,
            ]
            __d_values = [
                int(__d_inp.get()),
                int(__d_inp.get()) + 1,
                int(__d_inp.get()) + 2,
            ]
            __q_values = [
                int(__q_inp.get()),
                int(__q_inp.get()) + 2,
                int(__q_inp.get()) + 2,
            ]
        else:
            __p_values = [
                int(__p_inp.get()),
                int(__p_inp.get()) * 2,
                int(__p_inp.get()) * 4,
            ]
            __d_values = [
                int(__d_inp.get()),
                int(__d_inp.get()) + 1,
                int(__d_inp.get()) + 2,
            ]
            __q_values = [
                int(__q_inp.get()),
                int(__q_inp.get()) * 2,
                int(__q_inp.get()) * 4,
            ]
    else:
        __p_values = [int(__p_inp.get())]
        __d_values = [int(__d_inp.get())]
        __q_values = [int(__q_inp.get())]

    if __gridsearch_d_inp.get() == "Yes":
        __d_values = [
            int(__d_inp.get()),
            int(__d_inp.get()) + 1,
            int(__d_inp.get()) + 2,
        ]
    else:
        __d_values = [int(__d_inp.get())]

    __best_score = 1000000000
    __best_cfg = 0
    score_dict = {}
    __timeA = time.time()
    __timeB = time.time()
    __prediction_time_frame = int(__time_inp.get())
    __site_counter = 0

    for __x in range(0, len(site_list)):
        try:
            __index_pos = date_list.index(__df_temp.date.unique()[0])
            __site_counter = __site_counter + 1
        except:
            continue

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

            __df_name = "df_" + str(__site_id_value) + "_" + str(__item_id_value)

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

            if int(len(__df_temp)) < int(__time_inp.get()):
                message_list = (
                    "Not enough data For Item_ID = "
                    + str(__item_id_value)
                    + " & Site_ID = "
                    + str(__site_id_value)
                    + "\n"
                    + "Total Data Points = "
                    + str(len(__df_temp_bu))
                    + "\n"
                    + "Very infrequently sold item, assume zero sales"
                )
                tk.messagebox.showerror(title="Warning", message=message_list)
                __sub_button_text.set("Submit and Run")
                continue
            else:
                pass

            try:
                __ad_out = adfuller(__df_temp.sales)
                __lags = __ad_out[2]
                __p_val = __ad_out[1]
            except:
                __p_val = 0.05
                tk.messagebox.showwarning(
                    title="Warning", message="Dickey Fuller Test Failed -- Continuing"
                )

            if __p_val <= 1:
                # Data is Stationary so ARIMA(p,0,q)

                __best_score = 1000000000
                __best_cfg = (0, 0, 0)

                for __p in __p_values:
                    for __d in __d_values:
                        for __q in __q_values:
                            __order = (__p, __d, __q)
                            try:

                                __model = ARIMA(endog=__df_temp["sales"], order=__order)
                                __model_fit = __model.fit()
                                __output = __model_fit.predict(
                                    start=len(__df_temp),
                                    end=int(
                                        len(__df_temp) + __prediction_time_frame - 1
                                    ),
                                    dynamic=False,
                                )
                                __rmse = sqrt(
                                    mean_squared_error(__output, __test_temp.sales)
                                )
                                counter = counter + 1
                                print(
                                    "Progress ",
                                    counter,
                                    "models ran of ",
                                    len(__p_values)
                                    * len(__d_values)
                                    * len(__q_values)
                                    * len(item_list)
                                    * __site_counter,
                                )

                                if __rmse < __best_score:
                                    __best_score = __rmse
                                    __best_cfg = __order
                                    insamp_prediction_dict[
                                        __site_id_value, __item_id_value
                                    ] = __output
                                else:
                                    pass
                            except ValueError as VE:
                                if "A constant trend was included in the model" in str(
                                    VE
                                ):
                                    tk.messagebox.showerror(
                                        title="ERROR",
                                        message="""Trend Included in the Model. Allow for gridsearching of Parameter [d]""",
                                    )
                                else:
                                    pass

                if __best_cfg == (0, 0, 0):
                    __best_cfg = __order
                else:
                    pass

                # In_sample Prep
                try:
                    __model = ARIMA(
                        exog=__df_temp[["maxtemp", "mintemp"]],
                        endog=__df_temp["sales"],
                        order=__best_cfg,
                    )
                except ValueError as VE:
                    if "A constant trend was included in the model" in str(VE):
                        message_List = (
                            "Trend Included in the Model."
                            + "\n"
                            + "Allow for gridsearching of Parameter [d]"
                            + "\n"
                            + "\n"
                            + "----- Model Quitting -----"
                        )
                        tk.messagebox.showerror(title="ERROR", message=message_List)
                        __sub_button_text.set("Submit and Run")
                        return
                    elif "zero-size array" in str(VE):
                        message_List = (
                            "Zero-Size Array Error"
                            + "\n"
                            + "Very Infrequenty Sold Item Included, Not Enough DataPoints exist"
                            + "\n"
                            + "\n"
                            + "----- Model Quitting -----"
                        )
                        tk.messagebox.showerror(
                            title="ERROR",
                            message="""Trend Included in the Model.
                            Allow for gridsearching of Parameter [d]""",
                        )
                        __sub_button_text.set("Submit and Run")
                        return

                __model_fit = __model.fit()
                __output = __model_fit.predict(
                    start=len(__df_temp),
                    end=int(len(__df_temp) + __prediction_time_frame - 1),
                    dynamic=False,
                    exog=__test_temp[["maxtemp", "mintemp"]],
                )
                model_sum_dict[__site_id_value, __item_id_value] = __model_fit.summary()
                test_dict[__site_id_value, __item_id_value] = __test_temp.sales
                score_dict[__site_id_value, __item_id_value, "cfg"] = __best_cfg
                score_dict[__site_id_value, __item_id_value, "rmse"] = __best_score
                score_dict[__site_id_value, __item_id_value, "sum"] = ceil(
                    sum(__test_temp.sales)
                )

                # Out_of_Sample Prediction
                __model = ARIMA(endog=__df_temp_bu["sales"], order=__best_cfg)
                __model_fit = __model.fit()
                __output = __model_fit.predict(
                    start=len(__df_temp_bu),
                    end=int(len(__df_temp_bu) + __prediction_time_frame - 1),
                    dynamic=False,
                )
                model_sum_dict[__site_id_value, __item_id_value] = __model_fit.summary()
                test_dict[__site_id_value, __item_id_value] = __test_temp.sales

                prediction_dict[__site_id_value, __item_id_value] = round(__output, 2)

    out_list = []
    if __save_inp.get() == "Yes":
        (
            pd.DataFrame.from_dict(data=prediction_dict, orient="index").to_csv(
                "prediction_output.csv", header=False
            )
        )
    else:
        pass

    for i in item_list:
        for x in site_list:
            try:
                sum_val = ceil(sum(prediction_dict[x, i]))
                sum_val2 = score_dict[x, i, "sum"]
                val3 = ceil(sum(insamp_prediction_dict[x, i])) - sum_val2
                item_name = productdict[i]
                string_output = (
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
                    + "Sum of Sales predicted 10 days = "
                    + str(sum_val)
                    + "\n"
                    + "Expected Sum of Sales based on previous 10 Days = "
                    + str(sum_val2)
                    + "\n"
                    + "Insample Total Error = "
                    + str(val3)
                    + "\n"
                    + "Best In-Sample CFG = "
                    + str(score_dict[x, i, "cfg"])
                    + "\n"
                    + "Best In-Sample RMSE = "
                    + str(round(score_dict[x, i, "rmse"], 2))
                    + "\n"
                    + "\n"
                )
                out_list.append(string_output)
            except:
                continue

    if __save_inp.get() == "Yes":
        message_string = (
            "Predictions Saved at: " + str(os.getcwd()) + "/" + "prediction_output.csv"
        )
        tk.messagebox.showinfo(title="Outcome", message=message_string)
    else:
        pass

    if __output_inp.get() == "Yes":
        if len(out_list) == 0:
            tk.messagebox.showerror(
                title="ERROR",
                message="Model Failed for all Item_Id and Site_ID combinations",
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
__app.geometry("700x600")
__app.title("Maverik: Candy Bar prediction")

__s = ttk.Style()
__s.theme_use("alt")

__sub_button_text = tk.StringVar()
__sub_button_text.set("Submit and Run")

urllib.request.urlretrieve(
    "https://www.dropbox.com/s/hph99elpmvwjjjl/logo_maverick.gif?dl=1",
    "logo_maverick.gif",
)

# Explanation
__logo = tk.PhotoImage(file="logo_maverick.gif", master=__app,)

__label00 = tk.Label(__app, text="")
__label00.grid(column=0, row=0)

w1 = tk.Label(__app, image=__logo).grid(column=2, row=1, columnspan=2)

__explanation = """Fill in the requested inputs. 
Use the buttons for free input to check if the data is correct.
The Program can be killed by using the X on the top right of the Dialogue box."""

w2 = tk.Label(__app, justify=tk.LEFT, padx=20, text=__explanation).grid(
    column=0, row=1, columnspan=2
)

# Padding
__label0 = tk.Label(__app, text="")
__label0.grid(column=0, row=2)

# Dataframe
tk.Label(__app, text="Input DataFrame .CSV Path").grid(row=3)
__dataframe_path = tk.Entry(__app)
__dataframe_path.insert(
    END,
    "C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/CapstoneMainDF.csv",
)
__dataframe_path.grid(row=3, column=1)


__submit_btn_1 = Button(__app, text="Check", width=10, command=check_dataframe_path)
__submit_btn_1.grid(row=3, column=2)

__dataframe_path_text_box = tk.Text(__app, width=15, height=1)
__dataframe_path_text_box.grid(row=3, column=3)
__dataframe_path_text_box.insert("end-1c", "waiting")

# Product_df
tk.Label(__app, text="Input Productdf .CSV Path").grid(row=4)
__productdf_path = tk.Entry(__app)
__productdf_path.insert(
    END,
    "C:/Users/wille/OneDrive/MSF&MSBA/5. Spring 2021/IS 6496 MSBA Capstone 3/ProductsDF.csv",
)
__productdf_path.grid(row=4, column=1)

__submit_btn_2 = Button(__app, text="Check", width=10, command=check_productdf_path)
__submit_btn_2.grid(row=4, column=2)

__productdf_path_text_box = tk.Text(__app, width=15, height=1)
__productdf_path_text_box.grid(row=4, column=3)
__productdf_path_text_box.insert("end-1c", "waiting")

# Padding
__label1 = tk.Label(__app, text="")
__label1.grid(column=0, row=5)

# Cleaning_Std
tk.Label(__app, text="Input Outlier Std. Deviation").grid(row=6)
__std_dev_inp = tk.Entry(__app)
__std_dev_inp.insert(END, "4")
__std_dev_inp.grid(row=6, column=1)

__submit_btn_3 = Button(__app, text="Check", width=10, command=check_standard_dev)
__submit_btn_3.grid(row=6, column=2)

__std_dev_inp_text_box = tk.Text(__app, width=15, height=1)
__std_dev_inp_text_box.grid(row=6, column=3, columnspan=1)
__std_dev_inp_text_box.insert("end-1c", "waiting")

# Item_id
tk.Label(__app, text="Input Top N or Item_ID").grid(row=7)
__item_id_inp = tk.Entry(__app)
__item_id_inp.insert(END, "10")
__item_id_inp.grid(row=7, column=1)

__submit_btn_4 = Button(__app, text="Check", width=10, command=check_item_id)
__submit_btn_4.grid(row=7, column=2)

__item_id_inp_text_box = tk.Text(__app, width=15, height=1)
__item_id_inp_text_box.grid(row=7, column=3, columnspan=1)
__item_id_inp_text_box.insert("end-1c", "waiting")

# Site_id
tk.Label(__app, text="Input Site_ID | Blank for All").grid(row=8)
__site_id_inp = tk.Entry(__app)
__site_id_inp.grid(row=8, column=1)

__submit_btn_5 = Button(__app, text="Check", width=10, command=check_site_id)
__submit_btn_5.grid(row=8, column=2)

__site_id_text_box = tk.Text(__app, width=15, height=1)
__site_id_text_box.grid(row=8, column=3, columnspan=1)
__site_id_text_box.insert("end-1c", "waiting")


# Padding
__label2 = tk.Label(__app, text="")
__label2.grid(column=0, row=9)

# P
tk.Label(__app, text="Input starting p").grid(row=10)
__p_inp = tk.Entry(__app)
__p_inp.insert(END, "7")
__p_inp.grid(row=10, column=1)

__submit_btn_6 = Button(__app, text="Check", width=10, command=check_p)
__submit_btn_6.grid(row=10, column=2)

__p_inp_text_box = tk.Text(__app, width=15, height=1)
__p_inp_text_box.grid(row=10, column=3, columnspan=1)
__p_inp_text_box.insert("end-1c", "waiting")

# d
tk.Label(__app, text="Input starting d").grid(row=11)
__d_inp = tk.Entry(__app)
__d_inp.insert(END, "0")
__d_inp.grid(row=11, column=1)

__submit_btn_7 = Button(__app, text="Check", width=10, command=check_d)
__submit_btn_7.grid(row=11, column=2)

__d_inp_text_box = tk.Text(__app, width=15, height=1)
__d_inp_text_box.grid(row=11, column=3, columnspan=1)
__d_inp_text_box.insert("end-1c", "waiting")

# q
tk.Label(__app, text="Input starting q").grid(row=12)
__q_inp = tk.Entry(__app)
__q_inp.insert(END, "1")
__q_inp.grid(row=12, column=1)

__submit_btn_8 = Button(__app, text="Check", width=10, command=check_q)
__submit_btn_8.grid(row=12, column=2)

__q_inp_text_box = tk.Text(__app, width=15, height=1)
__q_inp_text_box.grid(row=12, column=3, columnspan=1)
__q_inp_text_box.insert("end-1c", "waiting")

# Time
tk.Label(__app, text="Input Prediction Time Frame").grid(row=13)
__time_inp = tk.Entry(__app)
__time_inp.insert(END, "7")
__time_inp.grid(row=13, column=1)

__submit_btn_9 = Button(__app, text="Check", width=10, command=check_time)
__submit_btn_9.grid(row=13, column=2)

__time_inp_text_box = tk.Text(__app, width=15, height=1)
__time_inp_text_box.grid(row=13, column=3, columnspan=1)
__time_inp_text_box.insert("end-1c", "waiting")

# Padding
__label3 = tk.Label(__app, text="")
__label3.grid(column=0, row=14)

# Input
__label4 = tk.Label(__app, text="Gridsearch All?")
__label4.grid(column=0, row=15)

__gridsearch_inp = ttk.Combobox(__app, values=["Yes", "No"])
__gridsearch_inp.insert(END, "No")
__gridsearch_inp.grid(column=1, row=15)

# Gridsearch 2
__label5 = tk.Label(__app, text="Gridsearch D only?")
__label5.grid(column=0, row=16)

__gridsearch_d_inp = ttk.Combobox(__app, values=["Yes", "No"])
__gridsearch_d_inp.insert(END, "Yes")
__gridsearch_d_inp.grid(column=1, row=16)

# Ouput
__label6 = tk.Label(__app, text="See Output Promt?")
__label6.grid(column=0, row=17)

__output_inp = ttk.Combobox(__app, values=["Yes", "No"])
__output_inp.insert(END, "Yes")
__output_inp.grid(column=1, row=17)

# Save
__label7 = tk.Label(__app, text="Save Prediction?")
__label7.grid(column=0, row=18)

__save_inp = ttk.Combobox(__app, values=["Yes", "No"])
__save_inp.insert(END, "No")
__save_inp.grid(column=1, row=18)


## Submit
__submit_btn = tk.Button(
    __app,
    textvariable=__sub_button_text,
    command=getInput,
    height=2,
    width=20,
    bg="#cc0000",
    fg="white",
)
__submit_btn.grid(row=21, column=1, columnspan=2, padx=10, pady=25)

__app.mainloop()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:32:10 2020

@author: jkreula
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import os

curr_dir = os.path.abspath('')
zipfile = "Telco-Customer-Churn.zip"
zip_path = os.path.join(curr_dir,zipfile)

def extract_zip(zip_path: str) -> None:
    try:
        with ZipFile(zip_path, "r") as zf:  
            zf.extractall()
    except:
        print(f"Error in extracting zip file at {zip_path}!")
        

def print_summaries(df):
    print(df.info())
    print("\n\nHead:\n\n",df.head())
    print("\n\nColumns:\n\n",df.columns)
    print("\n\nDescribe:\n\n",df.describe())
    print("\n\nNumber of NaNs:\n\n",df.isnull().sum())
    
def create_kdeplot(column):
    yes = df[df["Churn"]=='Yes'][column]
    no = df[df["Churn"]=='No'][column]
    fig, ax = plt.subplots()
    sns.kdeplot(yes,color='blue',label='Yes')
    sns.kdeplot(no,color='red',label='No')
    plt.title(f"Churn for {column}")
    plt.xlabel(f"{column}")
    plt.ylabel("KDE")
    plt.show()
    
def create_barplot(df,column):
    df_plot = df.groupby(column)["Churn"].value_counts().to_frame()
    new_col_name = "Percentage"
    df_plot = df_plot.rename({"Churn":new_col_name},axis=1)
    df_plot[new_col_name] = df_plot[new_col_name]/df_plot[new_col_name].sum() * 100
    df_plot = df_plot.reset_index()
    sns.barplot(x=column, y=new_col_name, data = df_plot, hue = "Churn")
    plt.show()

if __name__ == "__main__":
    
    extract_zip(zip_path)
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print_summaries(df)
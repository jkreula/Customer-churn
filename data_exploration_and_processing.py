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
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

curr_dir = os.path.abspath('')
zipfile = "Telco-Customer-Churn.zip"
zip_path = os.path.join(curr_dir,zipfile)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, attribute_names: List[str]) -> None:
         self.attribute_names = attribute_names

    def fit(self, X, y=None):
        # Nothing to be fitted
        return self
    
    def transform(self, X):
        return X[self.attribute_names]

class CorrelatedFeaturesRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Nothing to be fitted
        return self

    def transform(self, X: pd.DataFrame, threshold: float = 0.7):
        try:
            corrs = X.corr()
        except:
            print("Works only with numerical features.")
            return None
        
        corr_cols = self._find_correlated_columns(corrs)
        return self._drop_columns(X, corr_cols)
        
    @staticmethod
    def _find_correlated_columns(corr_df: pd.DataFrame, threshold: float = 0.7) -> List[str]:
        corr_cols = set()
        for i in range(len(corr_df.columns)):
            for j in range(i):
                if corr_df.iloc[i,j] > threshold:
                  corr_cols.add(corr_df.columns[i])
        return list(corr_cols)
    
    @staticmethod
    def _drop_columns(df, columns):
        return df.drop(columns,axis=1)

def head(df,n=5):
    return df.head(n)

def tail(df,n=5):
    return df.tail(n)

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
    
def create_piechart(ser: pd.Series, title: str = "", save: bool = False, *, save_folder: str = "", filename: str = "") -> None:
    ser.plot(kind="pie", title = title, autopct = "%.1f%%")
    plt.show()
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,filename))    
    
def create_kdeplot(df: pd.DataFrame, column: str, save: bool = False, *, save_folder: str = "", filename: str = "") -> None:
    yes = df[df["Churn"]=='Yes'][column]
    no = df[df["Churn"]=='No'][column]
    fig, ax = plt.subplots()
    sns.kdeplot(yes,color='blue',label='Yes',ax=ax)
    sns.kdeplot(no,color='red',label='No',ax=ax)
    plt.title(f"Churn for {column}")
    plt.xlabel(f"{column}")
    plt.ylabel("KDE")
    plt.show()
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,filename))
    
def create_barplot(df: pd.DataFrame, column: str, save: bool = False, *, save_folder: str = "", filename: str = "") -> None:
    df_plot = df.groupby(column)["Churn"].value_counts().to_frame()
    new_col_name = "Percentage"
    df_plot = df_plot.rename({"Churn":new_col_name},axis=1)
    df_plot[new_col_name] = df_plot[new_col_name]/df_plot[new_col_name].sum() * 100
    df_plot = df_plot.reset_index()
    sns.barplot(x=column, y=new_col_name, data = df_plot, hue = "Churn")
    plt.show()
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,filename))
        
def replace_values_in_column(df: pd.DataFrame, col: str, mapping_dict: dict) -> None:
    df.replace({col: mapping_dict},inplace=True)

# def compute_correlations(df: pd.DataFrame) -> None:
#     # Factorize for simplicity, we only care about changes in the values, not the values themselves
#     factorized_df = df.apply(lambda x: pd.factorize(x)[0])
#     correlations = factorized_df.corr()
#     return correlations

def create_heatmap(mat, save: bool = False, *, save_folder: str = "", filename: str = ""):
    plt.figure(figsize=(15, 10))
    sns.heatmap(mat,annot=True)
    plt.show()
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,filename))
        
# def find_correlated_columns(corr_df: pd.DataFrame, threshold: float = 0.7) -> List[str]:
#     corr_cols = set()
#     for i in range(len(corr_df.columns)):
#         for j in range(i):
#             if corr_df.iloc[i,j] > threshold:
#               corr_cols.add(corr_df.columns[i])
#     return list(corr_cols)

def separate_numerical_and_categorical_features(df: pd.DataFrame) -> Tuple[List[str],List[str]]:
    # Numerical columns
    num_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_cols = list(df.select_dtypes(include=num_dtypes).columns)
    num_cols.remove('SeniorCitizen')
    
    # Categorical columns
    cat_cols = df.columns[~df.columns.isin(num_cols)]
    
    assert len(num_cols) + len(cat_cols) == len(df.columns), print("Some columns are missing!")
    return num_cols, cat_cols

########################################################################
if __name__ == "__main__":
    
    extract_zip(zip_path)
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    head(df)
    tail(df)
    if df["customerID"].is_unique:
        # Customer ID not going to be a useful feature since unique per customer
        df.drop(["customerID"],axis=1,inplace=True)
    
    print_summaries(df)
    
    # There is a space in TotalCharges, change it to 0.0
    df["TotalCharges"] = df["TotalCharges"].replace(" ","0.0")
    df["TotalCharges"] = df["TotalCharges"].astype("float")
    
    create_piechart(df["Churn"].value_counts())
    
    num_cols, cat_cols = separate_numerical_and_categorical_features(df)
    
    for col in num_cols:
        create_kdeplot(df,col)
    
    for col in cat_cols:
        if col == "customerID" or col == "Churn": continue
        create_barplot(df,col)
        
    mapping_dict_PaymentMethod = {'Electronic check': 'EC', 
                                  'Mailed check': 'MC', 
                                  'Bank transfer (automatic)': 'BT', 
                                  'Credit card (automatic)': 'CC'}
    
    replace_values_in_column(df, "PaymentMethod", mapping_dict_PaymentMethod)
    
    dfs_num = DataFrameSelector(num_cols)
    
    df_num = dfs_num.fit_transform(df)
    
    dfs_cat = DataFrameSelector(cat_cols)
    
    df_cat = dfs_cat.fit_transform(df)
    
    #corrs = compute_correlations(df.drop(["Churn"],axis=1))
    create_heatmap(df_num.corr())
    
    cfr = CorrelatedFeaturesRemover()
    
    df_no_corr = cfr.fit_transform(df_num)
    
    #corr_columns = find_correlated_columns(corrs,0.7)

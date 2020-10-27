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
from sklearn.model_selection import train_test_split

curr_dir = os.path.abspath('')
zipfile = "Telco-Customer-Churn.zip"
zip_path = os.path.join(curr_dir,zipfile)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature_names: List[str]) -> None:
         self.feature_names = feature_names

    def fit(self, X, y=None):
        # Nothing to be fitted
        return self
    
    def transform(self, X):
        return X[self.feature_names]

class CorrelatedFeaturesRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.corr_cols = []
    
    def fit(self, X, y=None, threshold: float = 0.7):
        try:
            corrs = X.corr()
        except:
            print("Works only with numerical features.")
            return None
        self.corr_cols = self._find_correlated_columns(corrs)
        return self

    def transform(self, X: pd.DataFrame):
        return self._drop_columns(X, self.corr_cols)
        
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

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Nothing to be fitted
        return self
    
    def transform(self, X):
        X_ = X.copy(deep=True)
        for col in X_.columns:
            if X_[col].nunique() == 2:
                X_[col] = pd.factorize(X_[col])[0]
            else:
                X_ = pd.get_dummies(X_, columns=[col], drop_first=True)
        return X_

class ColumnValueReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, col, mapping_dict):
        self.col = col
        self.mapping_dict = mapping_dict
    
    def fit(self, X, y=None):
        # Nothing to be fitted
        return self
    
    def transform(self, X):
        return self._replace_values_in_column(X, self.col, self.mapping_dict)
    
    @staticmethod
    def _replace_values_in_column(df: pd.DataFrame, col: str, mapping_dict: dict) -> None:
        return df.replace({col: mapping_dict},inplace=False)

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
        


def create_heatmap(mat, save: bool = False, *, save_folder: str = "", filename: str = ""):
    plt.figure(figsize=(15, 10))
    sns.heatmap(mat,annot=True)
    plt.show()
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,filename))

def separate_numerical_and_categorical_features(df: pd.DataFrame) -> Tuple[List[str],List[str]]:
    # Numerical columns
    num_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_cols = list(df.select_dtypes(include=num_dtypes).columns)
    num_cols.remove('SeniorCitizen')
    
    # Categorical columns
    cat_cols = df.columns[~df.columns.isin(num_cols)]
    
    assert len(num_cols) + len(cat_cols) == len(df.columns), print("Some columns are missing!")
    return num_cols, cat_cols

def encode_y(y):
    return pd.factorize(y)[0]

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
    
    X, y = df.drop(["Churn"],axis=1), df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    num_cols, cat_cols = separate_numerical_and_categorical_features(X_train)
    
    df_viz = pd.concat([X_train,y_train],axis=1)
    
    for col in num_cols:
        create_kdeplot(df_viz,col)
    
    for col in cat_cols:
        if col == "customerID" or col == "Churn": continue
        create_barplot(df_viz,col)
        
    mapping_dict_PaymentMethod = {'Electronic check': 'EC', 
                                  'Mailed check': 'MC', 
                                  'Bank transfer (automatic)': 'BT', 
                                  'Credit card (automatic)': 'CC'}
    
    #replace_values_in_column(df, "PaymentMethod", mapping_dict_PaymentMethod)
    
    # dfs_num = DataFrameSelector(num_cols)
    # df_num = dfs_num.fit_transform(df)
    # create_heatmap(df_num.corr())
    # cfr = CorrelatedFeaturesRemover()
    # df_no_corr = cfr.fit_transform(df_num)
    
    # dfs_cat = DataFrameSelector(cat_cols)
    # df_cat = dfs_cat.fit_transform(df)
    # ce = CategoricalEncoder()
    # df_cat_encoded = ce.fit_transform(df_cat)
    
    
    
    
    pl_num = Pipeline([('sel_num',DataFrameSelector(num_cols)),
                       ('corr_remover',CorrelatedFeaturesRemover())])
    
    pl_cat = Pipeline([('sel_cat',DataFrameSelector(cat_cols)),
                       ('val_rep', ColumnValueReplacer("PaymentMethod", mapping_dict_PaymentMethod)),
                       ('encoder',CategoricalEncoder())])
    
    X_num_prepared = pl_num.fit_transform(X_train)
    X_cat_prepared = pl_cat.fit_transform(X_train)
    
    X_train_prep = pd.concat([X_num_prepared,X_cat_prepared],axis=1)
    
    # full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", pl_num),
    #                                                ("cat_pipeline", pl_cat)])
    
    
    # X_train_prep = full_pipeline.fit_transform(X_train)
    # X_train_prep = pd.DataFrame(X_train_prep, columns=X_train.columns)
    
    y_train, y_test = encode_y(y_train), encode_y(y_test)
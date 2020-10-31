#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:32:10 2020

@author: jkreula
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns ; sns.set()
from zipfile import ZipFile
import os
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

curr_dir = os.path.abspath('')
zipfile = "Telco-Customer-Churn.zip"
zip_path = os.path.join(curr_dir,zipfile)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._feature_names = None
    
    def fit(self, X, y=None):
        # Nothing to be fitted
        return self
    
    def transform(self, X):
        X_ = pd.get_dummies(X, drop_first=True)
        self._feature_names = X_.columns
        return X_
    
    @property
    def feature_names(self):
        return self._feature_names
    
class CorrelatedFeaturesRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._corr_cols = []
        self._feature_names = None
    
    def fit(self, X, y=None, threshold: float = 0.7):
        try:
            corrs = X.corr()
        except:
            print("Works only with numerical features.")
            return None
        self._corr_cols = self._find_correlated_columns(corrs)
        return self

    def transform(self, X: pd.DataFrame):
        X_ = self._drop_columns(X, self._corr_cols)
        self._feature_names = X_.columns
        return X_
    
    @property
    def feature_names(self):
        return self._feature_names
        
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
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,filename)) 
    plt.show()
    
def create_kdeplot(df: pd.DataFrame, column: str, save: bool = False, *, save_folder: str = "", filename: str = "") -> None:
    yes = df[df["Churn"]==1][column]
    no = df[df["Churn"]==0][column]
    fig, ax = plt.subplots()
    sns.kdeplot(yes,color='blue',label='Yes',ax=ax)
    sns.kdeplot(no,color='red',label='No',ax=ax)
    plt.title(f"Churn for {column}")
    plt.xlabel(f"{column}")
    plt.ylabel("KDE")
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,filename))
    plt.show()
    
def create_barplot(df: pd.DataFrame, column: str, save: bool = False, *, save_folder: str = "", filename: str = "") -> None:
    df_plot = df.groupby(column)["Churn"].value_counts().to_frame()
    new_col_name = "Percentage"
    df_plot = df_plot.rename({"Churn":new_col_name},axis=1)
    df_plot[new_col_name] = df_plot[new_col_name]/df_plot[new_col_name].sum() * 100
    df_plot = df_plot.reset_index()
    sns.barplot(x=column, y=new_col_name, data = df_plot, hue = "Churn")
    if column == "PaymentMethod":
        plt.xticks(rotation=90)
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,filename))
    plt.show()
        
def create_heatmap(mat, save: bool = False, *, save_folder: str = "", filename: str = ""):
    plt.figure(figsize=(15, 10))
    sns.heatmap(mat,annot=True)
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,filename))
    plt.show()

def separate_numerical_and_categorical_features(df: pd.DataFrame) -> Tuple[List[str],List[str]]:
    # Numerical columns
    num_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_cols = list(df.select_dtypes(include=num_dtypes).columns)
    
    # Categorical columns
    cat_cols = df.columns[~df.columns.isin(num_cols)]
    
    assert len(num_cols) + len(cat_cols) == len(df.columns), print("Some columns are missing!")
    return num_cols, cat_cols

if __name__ == "__main__":

    extract_zip(zip_path)
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(df.info())
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = 'coerce')
    if df["customerID"].is_unique:
        # Customer ID not going to be a useful feature since unique per customer
        df.drop(["customerID"],axis=1,inplace=True)
    print(df.isnull().sum())
    # There are NaNs in TotalCharges, drop them
    df.dropna(inplace=True)
    df.reset_index(inplace=True,drop=True)
    
    df["Churn"].replace({'Yes': 1, 'No': 0},inplace=True)    
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')
    
    # Extract numerical and categorical features
    num_cols, cat_cols = separate_numerical_and_categorical_features(df)
    num_cols.remove('Churn')
    
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    # Separate target and features
    y = df["Churn"].copy()
    X = df.drop(columns = ['Churn'])
    
    # Separate training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
    
    # Visualisations
    df_viz = pd.concat([X_train,y_train],axis=1)

    for col in num_cols:
        save_folder = os.path.join(curr_dir,"Figures")
        filename = col+ ".pdf"
        create_kdeplot(df_viz,col,save=True,save_folder=save_folder,filename=filename)
    
    for col in cat_cols:
        if col == "Churn": continue
        save_folder = os.path.join(curr_dir,"Figures")
        filename = col+ ".pdf"
        create_barplot(df_viz,col,save=True,save_folder=save_folder,filename=filename)    
    
    # Numerical pipeline
    pipeline_num = Pipeline([('corr_remover',CorrelatedFeaturesRemover()),
                             ('scaler',StandardScaler())])
    
    # Categorical pipeline
    pipeline_cat = Pipeline([('encoder', CategoricalEncoder() )])
    
    
    # Combine pipelines
    full_pipeline = ColumnTransformer([("num",pipeline_num,num_cols),
                                       ("cat",pipeline_cat,cat_cols)])
    
    # Prepare training data
    X_train_prep = full_pipeline.fit_transform(X_train)
    
    # Include columns
    prep_cols_num = full_pipeline.named_transformers_['num']['corr_remover'].feature_names
    prep_cols_cat = full_pipeline.named_transformers_['cat']['encoder'].feature_names
    prepared_columns = np.append(prep_cols_num,prep_cols_cat)
    
    X_train_prep = pd.DataFrame(X_train_prep, columns = prepared_columns, index = X_train.index)
    
    # Transform test data
    X_test_prep = full_pipeline.transform(X_test)
    X_test_prep = pd.DataFrame(X_test_prep, columns = prepared_columns, index = X_test.index)
    
    
    # Train models (to be tuned...)
    lr = LogisticRegression()
    rf = RandomForestClassifier(random_state = 42)
    hgb = HistGradientBoostingClassifier()
    svm = SVC(probability=True)
    
    clfs = [lr, rf, hgb, svm]
    
    for clf in clfs:
        print(f"Training {clf}")
        clf.fit(X_train_prep, y_train)
        y_pred = clf.predict(X_test_prep)
        y_score = clf.predict_proba(X_test_prep)[:,1]
        print(f"Accuracy = {accuracy_score(y_test,y_pred)}")
        print(f"AUC = {roc_auc_score(y_test,y_score)}")
        print(f"Recall = {recall_score(y_test,y_pred)}")
        print(f"F1 score = {f1_score(y_test,y_pred)}")
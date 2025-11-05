import pandas as pd
from sklearn.impute import SimpleImputer




def impute_column(df, column, strategy='median'):
    df = df.copy()
    if df[column].dtype.kind in 'biufc':
        imp = SimpleImputer(strategy=strategy)
        df[[column]] = imp.fit_transform(df[[column]])
        return df, f'Imputed {column} with {strategy}'
    else:
        imp = SimpleImputer(strategy='most_frequent')
        df[[column]] = imp.fit_transform(df[[column]])
        return df, f'Imputed {column} with mode'




def remove_outliers_iqr(df, column, k=1.5):
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    low = q1 - k*iqr
    high = q3 + k*iqr
    before = len(df)
    df = df[(df[column] >= low) & (df[column] <= high)]
    after = len(df)
    return df, f'Removed {before-after} rows outside IQR bounds in {column}'


from sklearn.feature_selection import SelectKBest, f_classif


def select_k_best(X, y, k=10):
    sel = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    sel.fit(X, y)
    mask = sel.get_support()
    return X.columns[mask].tolist()


import pandas as pd
from sklearn.impute import SimpleImputer
import streamlit as st




def auto_clean(df):
    report = []
    df = df.copy()


    # Remove duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        report.append(f'Removed {dup_count} duplicate rows.')


    # Handle missing values column-wise
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                mode = df[col].mode()[0]
                df[col].fillna(mode, inplace=True)
                report.append(f'Filled missing in {col} with mode ({mode}).')
            else:
                median = df[col].median()
                df[col].fillna(median, inplace=True)
                report.append(f'Filled missing in {col} with median ({median}).')


        st.success('Data cleaning completed!')
        for step in report:
                st.write('✅', step)
        return df, report
    

import numpy as np


def detect_outliers(df, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers




def recommend_outlier_treatment(df, col):
    method = 'IQR' if df[col].skew() < 2 else 'Z-score'
    reason = 'moderate distribution' if method == 'IQR' else 'high skewness'
    return method, reason
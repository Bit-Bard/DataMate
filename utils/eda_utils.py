import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor




def generate_eda(df):
    st.subheader('📊 Automated EDA')


    # Overview
    st.write('Shape:', df.shape)
    st.write('Data Types:', df.dtypes)
    st.write('Missing Values:', df.isnull().sum())


    # Correlation heatmap
    num_df = df.select_dtypes(include='number')
    if not num_df.empty:
        st.write('Correlation Heatmap:')
        fig, ax = plt.subplots()
        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)


    # Pairplot suggestion
    st.write('Top correlated pairs:')
    corr_pairs = num_df.corr().abs().unstack().sort_values(ascending=False)
    st.write(corr_pairs.head(10))


    # Feature importance (if target column selected)
    target = st.selectbox('Select target for importance (optional)', df.columns)
    if target:
        X = df.drop(columns=[target]).select_dtypes(include='number').fillna(0)
        y = df[target]
        if y.dtype == 'object':
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()
            model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(importances.sort_values(ascending=False))
    st.write('Feature Importance Summary:')
    st.write(importances.describe())




def custom_plot(df):
    col1 = st.selectbox('Select X column', df.columns)
    col2 = st.selectbox('Select Y column', df.columns)
    chart_type = st.selectbox('Chart type', ['scatter', 'line', 'bar'])


    fig, ax = plt.subplots()
    if chart_type == 'scatter':
        ax.scatter(df[col1], df[col2])
    elif chart_type == 'line':
        ax.plot(df[col1], df[col2])
    else:
        df.groupby(col1)[col2].mean().plot(kind='bar', ax=ax)


    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f'{chart_type.title()} Plot of {col1} vs {col2}')
    st.pyplot(fig)
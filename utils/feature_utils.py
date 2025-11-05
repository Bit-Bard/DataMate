import pandas as pd
import streamlit as st




def auto_feature_engineering(df):
    st.subheader('⚙️ Feature Engineering Studio')
    numeric = df.select_dtypes(include='number')
    created = []


    # Ratio features
    if st.checkbox('Create ratio features'):
        col1 = st.selectbox('Numerator', numeric.columns)
        col2 = st.selectbox('Denominator', numeric.columns)
    if col1 != col2:
        new_name = f'{col1}_div_{col2}'
        df[new_name] = numeric[col1] / (numeric[col2] + 1e-5)
        created.append(new_name)


    # Polynomial features
    if st.checkbox('Generate polynomial features'):
        degree = st.slider('Degree', 2, 5, 2)
        for col in numeric.columns:
            for d in range(2, degree+1):
                new_name = f'{col}_pow_{d}'
                df[new_name] = numeric[col] ** d
                created.append(new_name)


    # Binning
    if st.checkbox('Create bins'):
        col = st.selectbox('Select column to bin', numeric.columns)
        bins = st.slider('Number of bins', 2, 10, 5)
        new_name = f'{col}_binned'
        df[new_name] = pd.cut(numeric[col], bins=bins, labels=False)
        created.append(new_name)


    if created:
        st.success(f'Created features: {created}')
        return df
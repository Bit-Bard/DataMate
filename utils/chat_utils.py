import re
from .eda_utils import columns_with_nulls
from .plot_utils import show_boxplot, show_distribution
from .preprocess_utils import impute_column, remove_outliers_iqr




def handle_query(query: str, df):
    q = query.lower()
    # Missing values
    if 'missing' in q or 'null' in q or 'nan' in q:
        cols = columns_with_nulls(df)
        if cols:
            text = f'Found missing values in: {cols}'
        else:
            text = 'No missing values detected.'
            return {'text': text}


    # Boxplot
    m = re.search(r'boxplot for ([a-zA-Z0-9_ ]+)', q)
    if m:
        col = m.group(1).strip()
        if col in df.columns:
            fig = show_boxplot(df, col)
            return {'text': f'Showing boxplot for {col}', 'plot': fig}
        else:
            return {'text': f'Column {col} not found.'}


    # Remove outliers
    m = re.search(r'remove outliers in ([a-zA-Z0-9_ ]+)', q)
    if m:
        col = m.group(1).strip()
        if col in df.columns:
            new_df, desc = remove_outliers_iqr(df, col)
            return {'text': desc, 'new_df': new_df}
        else:
            return {'text': f'Column {col} not found.'}


    # Default
    return {'text': "I didn't understand. Try: 'show boxplot for Salary', 'any missing values?', 'remove outliers in Salary'"}
import matplotlib.pyplot as plt


def show_distribution(df, column):
    fig, ax = plt.subplots()
    df[column].hist(ax=ax)
    ax.set_title(f'Distribution: {column}')
    return fig




def show_boxplot(df, column):
    fig, ax = plt.subplots()
    df.boxplot(column=column, ax=ax)
    ax.set_title(f'Boxplot: {column}')
    return fig
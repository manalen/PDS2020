import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
%matplotlib inline

def draw_heatmap(data):
    sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

def convert_cast(df,categorical_cols):
   for col_name in categorical_cols:
       df[col_name]=df[col_name].astype("object")
   return df

def input_missing_values(df):
    for col in df.columns:
        if (df[col].dtype == "float64") or (df[col].dtype is int):
            df[col]=df[col].fillna(df[col].median())
        if (df[col].dtype == "object"):
            df[col]=df[col].fillna(df[col].mode()[0].split(" ")[0])
    return df

def parse_model(X, use_columns):
    if "Survived" not in X.columns :
        raise ValueError("target column survived should belong to df")
    target=X["Survived"]
    X=X[use_columns]
    return X, target

def plot_hist(survived, dead, feature, bins=20):
    x1 = np.array(dead[feature].dropna())
    x2 = np.array(survived[feature].dropna())
    plt.hist([x1, x2], label=["Victime", "Survivant"], bins=bins, color=['r', 'b'])
    plt.legend(loc="upper left")
    plt.title('Distribution relative de %s' %feature)
    plt.show()

def dummify(X,cols) :
    for col in cols:
        X_dummies=pd.get_dummies(X[col],prefix=col,drop_first=False,dummy_na=False,prefix_sep='_')
        X=X.join(X_dummies).drop(col,axis=1)
    return X

def convert_df_columns(data_str,features,type_var):
    for feature in features:
        data_str[feature]=data_str[feature].astype(type_var)
    return data_str

def transform_df(X, columns_to_dummify, features=["Pclass"],thres=10):
    X=convert_df_columns(X,features,type_var="object")
    X["is_child"]=X["Age"].apply(lambda x: 0 if x<thres else 1)
    X["title"]=X["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    X["Surname"]=X['Name'].map(lambda x: '(' in x)
    for col in columns_to_dummify:
        X_dummies=pd.get_dummies(X[col],prefix=col,drop_first=False,dummy_na=False, prefix_sep='_')
        X=X.join(X_dummies).drop(col,axis=1)
    return X.drop("Name",axis=1).drop("Age",axis=1)

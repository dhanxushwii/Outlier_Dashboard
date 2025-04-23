'''import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_dataset(name):
    path_map = {
        "AirQualityUCI": "datasets/AirQualityUCI.csv",
        "bank": "datasets/bank.csv",
        "BeijingClimate": "datasets/Beijing Climate.csv",
        "nhanes": "datasets/nhanes.csv",
        "students": "datasets/students.csv",
        "ObesityDataSet": "datasets/ObesityDataSet_raw_and_data_sinthetic.csv"
    }

    label_columns = {
        "nhanes": "RIDAGEYR",
        "students": "Target",
        "ObesityDataSet": "NObeyesdad",
        "bank": "y",
        "AirQualityUCI": "Date",
        "BeijingClimate": "No"
    }

    path = path_map.get(name)
    if not path:
        raise ValueError(f"Unknown dataset: {name}")

    # Dataset-specific loading logic
    if name == "AirQualityUCI":
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
    elif name == "bank":
        df = pd.read_csv(path, sep=';')
    else:
        df = pd.read_csv(path)

    df = df.dropna(axis=1, how='all')
    df = df.drop_duplicates()
    df = df.dropna(thresh=len(df.columns) - 2)

    # Drop label column if defined
    if name in label_columns and label_columns[name] in df.columns:
        df = df.drop(columns=[label_columns[name]])

    # Drop subject ID columns or irrelevant string columns
    if "SEQN" in df.columns:
        df = df.drop(columns=["SEQN"])

    # One-hot encode categoricals
    df = pd.get_dummies(df)

    # Fill missing numeric values with column mean
    df = df.fillna(df.mean(numeric_only=True))

    # Final numeric standardization
    scaled = StandardScaler().fit_transform(df)
    return pd.DataFrame(scaled)
'''










import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_dataset(name):
    path_map = {
        "AirQualityUCI": "datasets/AirQualityUCI.csv",
        "bank": "datasets/bank.csv",
        "BeijingClimate": "datasets/Beijing Climate.csv",
        "nhanes": "datasets/nhanes.csv",
        "students": "datasets/students.csv",
        "ObesityDataSet": "datasets/ObesityDataSet_raw_and_data_sinthetic.csv"
    }

    label_columns = {
        "nhanes": "RIDAGEYR",
        "students": "Target",
        "ObesityDataSet": "NObeyesdad",
        "bank": "y",
        "AirQualityUCI": "Date",
        "BeijingClimate": "No"
    }

    path = path_map.get(name)
    if not path:
        raise ValueError(f"Unknown dataset: {name}")

  
    if name == "AirQualityUCI":
        df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
    elif name == "bank":
        df = pd.read_csv(path, sep=';')
    else:
        df = pd.read_csv(path)


    df = df.dropna(axis=1, how='all')
    df = df.drop_duplicates()


    df = df.dropna(thresh=len(df.columns) - 2)

   
    label_col = label_columns.get(name)
    if label_col and label_col in df.columns:
        df = df.drop(columns=[label_col])


    if "SEQN" in df.columns:
        df = df.drop(columns=["SEQN"])


    df = pd.get_dummies(df)


    df = df.fillna(df.mean(numeric_only=True))

    scaled = StandardScaler().fit_transform(df)
    return pd.DataFrame(scaled)

import pandas as pd
import numpy as np

def columns_names(df):
    df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"[^\w]+", "_", regex=True)
      .str.replace(r"_+", "_", regex=True)
      .str.strip("_")
    )
    return df

def date_type(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    return df

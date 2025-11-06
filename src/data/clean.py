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


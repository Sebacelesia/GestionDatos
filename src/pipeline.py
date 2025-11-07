import pandas as pd

from src.data.load import read_clientes, read_orders, read_oproductos
from src.data.clean import columns_names, date_type

df_clientes = read_clientes()

def build_dataset():
    pass
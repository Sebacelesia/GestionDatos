import pandas as pd

from src.extract_transf.load import load_all
from src.extract_transf.clean import columns_names, date_type
from src.feature.rfm import make_cutoffs, split_hist_future, rfm_features, safe_qcut
from src.feature.temporal import diff_time_stats, tenure_days, diver, stats, ticket_stats_,recent_purchases, promedio_shipping_cost
from src.feature.churn import make_churn   

def build_dataset():
    dfs = load_all()
    dforders, dfclients, dfprods = dfs["orders"], dfs["clients"], dfs["products"]

    df_orders = columns_names(dforders)
    dfclients = columns_names(dfclients)
    dfprods = columns_names(dfprods)

    dforders = date_type(dforders, "order_timestamp")
    dfclients = date_type(dfclients, "registration_date")
    dfprods = date_type(dfprods, "last_seen")

    fecha_corte, fecha_ref= make_cutoffs(dforders)
    df_hist, df_futuro = split_hist_future(dforders,fecha_corte)

    rfm = rfm_features(df_hist, fecha_ref)

    diff_stats = diff_time_stats(df_hist)
    tenure = tenure_days(df_hist)
    diversidad = diver(df_hist)
    moda_categoria, moda_payment, moda_shipping = stats(df_hist)
    ticket_stats = ticket_stats_(df_hist)
    compras_recientes = recent_purchases(df_hist)
    prom_shipping = promedio_shipping_cost(df_hist)

    rfm = safe_qcut(rfm)

    df_churn = make_churn(df_hist, df_futuro)

    dataset = (
    rfm
    .merge(dfclients, on='client_id', how='left')
    .merge(diff_stats, on='client_id', how='left')
    .merge(tenure, on='client_id', how='left')
    .merge(diversidad, on='client_id', how='left')
    .merge(moda_categoria, on='client_id', how='left')
    .merge(moda_payment, on='client_id', how='left')
    .merge(moda_shipping, on='client_id', how='left')
    .merge(ticket_stats, on='client_id', how='left')
    .merge(compras_recientes, on='client_id', how='left')
    .merge(prom_shipping, on='client_id', how='left')
    .merge(df_churn[['client_id', 'churn']], on='client_id', how='left')
    )

    dataset = dataset.drop(columns=['first_name', 'last_name', 'email', 'address', 'postal_code','document_type','document_number','document_number',
         ], errors='ignore')
    return dataset
    


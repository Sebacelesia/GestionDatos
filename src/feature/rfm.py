import pandas as pd

def make_cutoffs(df_orders: pd.DataFrame, horizon_days: int = 60):
    """Devuelve (fecha_corte, fecha_ref). fecha_ref = fecha_corte + 1 día."""
    fecha_max = df_orders["order_timestamp"].max()
    fecha_corte = fecha_max - pd.Timedelta(days=horizon_days)
    fecha_ref = fecha_corte + pd.Timedelta(days=1)
    return fecha_corte, fecha_ref

def split_hist_future(df_orders: pd.DataFrame, fecha_corte: pd.Timestamp):
    """Separa histórico (<= corte) y futuro (> corte)."""
    df_hist = df_orders[df_orders["order_timestamp"] <= fecha_corte].copy()
    df_futuro = df_orders[df_orders["order_timestamp"] >  fecha_corte].copy()
    return df_hist, df_futuro

def rfm_features(df_hist: pd.DataFrame, fecha_ref: pd.Timestamp) -> pd.DataFrame:
    """Recency/Frequency/Monetary y scores en quintiles."""
    rfm = (
        df_hist.groupby("client_id")
        .agg({
            "order_timestamp": lambda x: (fecha_ref - x.max()).days,  # Recency
            "order_id": "count",                                      # Frequency
            "order_price": "sum",                                      # Monetary
        })
        .rename(columns={
            "order_timestamp": "Recency_Dias",
            "order_id": "Frequency",
            "order_price": "Monetary",
        })
        .reset_index()
    )
    return rfm



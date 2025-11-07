import pandas as pd

def diff_time_stats(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Promedio, std, max, min de días entre compras por cliente."""
    s = df_hist.sort_values(["client_id", "order_timestamp"])
    diffs = s.groupby("client_id")["order_timestamp"].apply(lambda x: x.diff().dt.days).reset_index()
    out = (
        diffs.groupby("client_id")["order_timestamp"]
        .agg(["mean","std","max","min"])
        .reset_index()
        .rename(columns={
            "mean":"Promedio_Dias_Compras",
            "std":"Varianza_Dias_Compras",
            "max":"Max_Dias_Entre_Compras",
            "min":"Min_Dias_Entre_Compras"
        })
    )
    return out

def tenure_days(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Antigüedad total (max - min) en días por cliente."""
    agg = (
        df_hist.groupby("client_id")["order_timestamp"]
        .agg(["min","max"]).reset_index()
    )
    agg["Tenure_Dias"] = (agg["max"] - agg["min"]).dt.days
    return agg[["client_id","Tenure_Dias"]]

# --- Diversidad de categorías ---
def diver(df_hist: pd.DataFrame) ->pd.DataFrame:
    diversidad = (
        df_hist.groupby('client_id')['product_category']
        .nunique()
        .reset_index(name='Diversidad_Categorias')
    )
    return diversidad

# --- Moda de categoría, método de pago y envío ---
def stats(df_hist: pd.DataFrame) ->pd.DataFrame:

    moda_categoria = df_hist.groupby('client_id')['product_category'].agg(lambda x: x.mode()[0]).reset_index(name='Moda_Categoria')
    moda_payment = df_hist.groupby('client_id')['payment_method'].agg(lambda x: x.mode()[0]).reset_index(name='Moda_Payment')
    moda_shipping = df_hist.groupby('client_id')['shipping_method'].agg(lambda x: x.mode()[0]).reset_index(name='Moda_Shipping')
    return moda_categoria, moda_payment, moda_shipping

# --- Ticket promedio y variabilidad ---
def ticket_stats_(df_hist:pd.DataFrame) -> pd.DataFrame:

    ticket_stats = (
        df_hist.groupby('client_id')['order_price']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'Ticket_Promedio', 'std': 'Ticket_Desvio'})
    )

    return ticket_stats

def recent_purchases(df_hist: pd.DataFrame, fecha_ref: pd.Timestamp) -> pd.DataFrame:
    """Cantidad de compras últimos 30/90 días (respecto a fecha_ref)."""
    tmp = df_hist.copy()
    tmp["dias_hasta_ref"] = (fecha_ref - tmp["order_timestamp"]).dt.days
    out = (
        tmp.assign(
            ult30=lambda x: x["dias_hasta_ref"] <= 30,
            ult90=lambda x: x["dias_hasta_ref"] <= 90
        )
        .groupby("client_id")
        .agg({"ult30":"sum","ult90":"sum"})
        .reset_index()
        .rename(columns={"ult30":"Compras_Ultimos30","ult90":"Compras_Ultimos90"})
    )
    return out





def promedio_shipping_cost(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Promedio del costo de envío por cliente (histórico hasta la fecha de corte).
    Devuelve: [client_id, Promedio_Shipping_Cost]
    """
    prom_shipping = (
        df_hist.groupby("client_id")["shipping_cost"]
        .mean()
        .reset_index(name="Promedio_Shipping_Cost")
    )
    return prom_shipping



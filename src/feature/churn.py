import pandas as pd

def make_churn(df_hist: pd.DataFrame, df_futuro: pd.DataFrame) -> pd.DataFrame:
    """
    Define churn por cliente:
    churn = 1 si NO hay compras post-corte, 0 si sí hay.
    Devuelve: [client_id, churn]
    """
    ultima_compra = (
        df_hist.groupby("client_id")["order_timestamp"]
        .max()
        .reset_index()
        .rename(columns={"order_timestamp": "ultima_compra"})
    )

    compras_post = (
        df_futuro.groupby("client_id")["order_id"]
        .count()
        .reset_index()
        .rename(columns={"order_id": "compras_post"})
    )

    df_churn = ultima_compra.merge(compras_post, on="client_id", how="left")
    df_churn["compras_post"] = df_churn["compras_post"].fillna(0)
    df_churn["churn"] = (df_churn["compras_post"] == 0).astype(int)

    return df_churn[["client_id", "churn"]]

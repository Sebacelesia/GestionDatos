from pathlib import Path
import pandas as pd
from typing import Union, Dict

PathLike = Union[str, Path]


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"

def read_orders(path: PathLike = DEFAULT_DATA_DIR / "dforders.csv", **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["order_timestamp"], **kwargs)

def read_clients(path: PathLike = DEFAULT_DATA_DIR / "dfclientes.csv", **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["registration_date", "last_seen"], **kwargs)

def read_products(path: PathLike = DEFAULT_DATA_DIR / "dfproductos.csv", **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def load_all(base_dir: PathLike = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    base = Path(base_dir)

    try:
        dforders = read_orders(base / "dforders.csv")
        dfclients = read_clients(base / "dfclientes.csv")
        dfprods  = read_products(base / "dfproductos.csv")

        return {
            "orders":   dforders,
            "clients":  dfclients,
            "products": dfprods,
        }

    except FileNotFoundError as e:
        print(f"[ERROR] No se encontró alguno de los archivos CSV en '{base}': {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Ocurrió un problema al cargar los datos desde '{base}': {e}")
        raise

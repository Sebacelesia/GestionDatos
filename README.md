# Customer Churn Prediction Pipeline

Proyecto de ciencia de datos para modelar **churn de clientes** en un contexto retail/e-commerce, siguiendo un flujo de trabajo completo:

- ETL de datos de órdenes, clientes y productos  
- Construcción de variables de negocio (RFM, métricas temporales, comportamiento de compra, etc.)  
- Generación de la variable objetivo de churn  
- Entrenamiento y evaluación de modelos de Machine Learning  
- Trazabilidad de experimentos con **MLflow**

---

## 1. Requisitos

- Python `3.9+` (desarrollado con Python `3.11`)
- `pip` instalado
- (Opcional pero recomendado) Virtualenv / venv

Las dependencias del proyecto están definidas en `requirements.txt`  
(ej.: `pandas`, `scikit-learn`, `xgboost`, `mlflow`, etc.).

---

## 2. Estructura del proyecto

GestionDatos/
├─ data/
│   ├─ dforders.csv
│   ├─ dfclientes.csv
│   └─ dfproductos.csv
├─ mlruns/              # Se crea automáticamente al correr el pipeline / MLflow
├─ notebooks/
│   └─ models.ipynb     # Exploración y pruebas (opcional)
├─ src/
│   ├─ extract_transf/
│   │   ├─ load.py      # Lectura de datos (dforders, dfclientes, dfproductos)
│   │   └─ clean.py     # Limpieza básica (nombre de columnas, tipos de fecha, etc.)
│   ├─ feature/
│   │   ├─ rfm.py       # Cálculo de RFM y cortes (Recency, Frequency, Monetary)
│   │   ├─ temporal.py  # Métricas temporales: difs entre compras, tenure, etc.
│   │   └─ churn.py     # Construcción de la variable objetivo (churn)
│   ├─ Models/
│   │   ├─ train_test.py    # Preprocesado y train/test split
│   │   └─ models_log.py    # Definición de modelos y logging en MLflow
│   └─ pipeline.py      # Script principal: orquesta todo el flujo
├─ requirements.txt
└─ README.md

---

## 3. Flujo de trabajo 

En una primera etapa realizamos el **análisis exploratorio de datos (EDA)** en notebooks, junto con las pruebas iniciales de *feature engineering*, para iterar rápido sobre ideas y validar qué variables aportaban mayor valor al modelo.

Una vez definida la estrategia de variables y transformaciones, migramos la lógica a **código modular y reproducible** dentro de la carpeta `src`, creando funciones específicas en los módulos de `extract_transf` y centralizando la orquestación del proceso en `pipeline.py`.

Posteriormente, retomamos los notebooks para la **experimentación con modelos de Machine Learning**, comparando diferentes algoritmos y configuraciones. Finalmente, tras seleccionar el modelo ganador, integramos su implementación en `src` para contar con un flujo de entrenamiento completamente automatizado y fácil de reproducir.

---

## 4. Instalación

Desde la raíz del proyecto (`GestionDatos/`):

# 1) Crear entorno virtual (se utilizó Python 3.11)
python -m venv .venv

# 2) Activar entorno
# En Windows:
.\.venv\Scripts\activate

# En macOS / Linux:
# source .venv/bin/activate

# 3) Instalar dependencias
pip install -r requirements.txt

---

## 5. Ejecución del pipeline

El archivo `src/pipeline.py` contiene el flujo completo:

- Carga y limpieza de datos  
  - `load_all`, `columns_names`, `date_type`.
- Generación de fechas de corte y split histórico/futuro.
- Cálculo de variables:  
  - RFM: `rfm_features`, `safe_qcut`.  
  - Métricas temporales: `diff_time_stats`, `tenure_days`, etc.  
  - Comportamiento: diversidad de categorías, ticket promedio, compras recientes, costo de envío, etc.
- Construcción de la etiqueta de churn:  
  - `make_churn`.
- Armado del dataset final y eliminación de datos sensibles  
  - Eliminación de campos como nombre, email, documento, etc.
- Preprocesado y train/test split  
  - `preprocess`, `hacer_train_test_split`.
- Construcción de modelos  
  - `build_models`.
- Entrenamiento, evaluación y logging en MLflow  
  - `correr_todos_los_modelos`.

---

## 6. Cómo ejecutar el pipeline completo

# Desde la raíz del proyecto
cd path/al/proyecto/GestionDatos
.\.venv\Scripts\activate     # Activar entorno virtual (Windows)

python -m src.pipeline

Al ejecutar el pipeline:

- Se construye el dataset a partir de los CSV de `data/`.
- Se entrenan los modelos definidos.
- Se loguean parámetros, métricas y modelos en MLflow.
- Se imprime en consola un DataFrame resumen (`resultados_df`) con las métricas de cada modelo.

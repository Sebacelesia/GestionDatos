from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import pandas as pd

numeric_features = [
    "Recency_Dias", "Frequency", "Monetary",
    "R_Score", "F_Score", "M_Score", "RFM_Score",
    "age",
    "Promedio_Dias_Compras", "Varianza_Dias_Compras",
    "Max_Dias_Entre_Compras", "Min_Dias_Entre_Compras",
    "Tenure_Dias", "Diversidad_Categorias",
    "Ticket_Promedio", "Ticket_Desvio",
    "Compras_Ultimos30", "Compras_Ultimos90",
    "Promedio_Shipping_Cost",
]

categorical_features = [
    "gender", "city", "country", "citizenship",
    "Moda_Categoria", "Moda_Payment", "Moda_Shipping",
]

preprocess_tree = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough"
)


def build_models(y_train):
    """
    Construye los 3 modelos:
      - RandomForest
      - XGBoost_Base
      - XGBoost_Tuned (mejor modelo de GridSearch)

    Usa y_train para calcular scale_pos_weight del XGBoost tuneado.
    Devuelve un diccionario {nombre_modelo: pipeline}.
    """

    # ----- scale_pos_weight para el XGBoost tuneado -----
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # 1) Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf_pipeline = Pipeline(steps=[
        ("preprocess", preprocess_tree),
        ("model", rf_model),
    ])

    # 2) XGBoost base (tu modelo simple inicial)
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    xgb_pipeline = Pipeline(steps=[
        ("preprocess", preprocess_tree),
        ("model", xgb_model),
    ])

    # 3) XGBoost tuneado (mejor GridSearch)
    xgb_model_true = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=2,
        learning_rate=0.02,
        subsample=1.0,
        colsample_bytree=0.7,
        min_child_weight=1,
        gamma=0.0,
        reg_lambda=4.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc"
    )

    xgb_pipeline_true = Pipeline(steps=[
        ("preprocess", preprocess_tree),
        ("model", xgb_model_true),
    ])

    models = {
        "RandomForest": rf_pipeline,
        "XGBoost_Base": xgb_pipeline,
        "XGBoost_Tuned": xgb_pipeline_true,
    }

    return models


def entrenar_y_loggear_modelo(nombre_modelo, pipeline, X_train, X_test, y_train, y_test):
    """
    Entrena un pipeline, calcula métricas y las loggea en MLflow.
    Devuelve un diccionario con las métricas.
    """
    
    with mlflow.start_run(run_name=nombre_modelo):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", auc)
        else:
            auc = None

        modelo_base = pipeline.named_steps["model"]
        mlflow.log_params(modelo_base.get_params())

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return {
        "modelo": nombre_modelo,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def correr_todos_los_modelos(models_dict, X_train, X_test, y_train, y_test):
    """
    Recorre el diccionario de modelos, los entrena y loggea en MLflow.
    Devuelve un DataFrame con las métricas de cada modelo.
    """
    resultados = []
    for nombre, pipe in models_dict.items():
        res = entrenar_y_loggear_modelo(
            nombre_modelo=nombre,
            pipeline=pipe,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        resultados.append(res)

    return pd.DataFrame(resultados)

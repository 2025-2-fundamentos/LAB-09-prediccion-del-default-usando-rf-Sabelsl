# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import json
import gzip
import pickle
import os

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix


def pregunta_1():
    def cargar_datos(ruta_archivo):
        return pd.read_csv(ruta_archivo, index_col=False, compression="zip")

    def depurar_datos(tabla):
        tabla = tabla.rename(columns={"default payment next month": "default"})
        tabla = tabla.drop(columns=["ID"])
        tabla = tabla.dropna()
        tabla["EDUCATION"] = tabla["EDUCATION"].apply(lambda valor: valor if valor < 4 else 4)
        return tabla

    def construir_pipeline():
        columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]

        transformador = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)
            ],
            remainder="passthrough",
        )

        flujo = Pipeline(
            steps=[
                ("preprocessor", transformador),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        )

        return flujo

    def optimizar_modelo(flujo, X_entrenamiento, y_entrenamiento):
        rejilla_parametros = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
        }

        criterio = make_scorer(balanced_accuracy_score)

        busqueda = GridSearchCV(
            estimator=flujo,
            param_grid=rejilla_parametros,
            scoring=criterio,
            cv=10,
            n_jobs=-1,
            verbose=1,
        )

        busqueda.fit(X_entrenamiento, y_entrenamiento)
        return busqueda

    def guardar_modelo(modelo, ruta_salida="files/models/model.pkl.gz"):
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        with gzip.open(ruta_salida, "wb") as archivo_binario:
            pickle.dump(modelo, archivo_binario)

    def evaluar_y_guardar_metricas(
        modelo,
        X_entrenamiento,
        y_entrenamiento,
        X_prueba,
        y_prueba,
        ruta_salida="files/output/metrics.json",
    ):
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        lista_metricas = []

        conjuntos = [
            ("train", X_entrenamiento, y_entrenamiento),
            ("test", X_prueba, y_prueba),
        ]

        for nombre_conjunto, X_actual, y_real in conjuntos:
            y_estimado = modelo.predict(X_actual)
            resultado = {
                "type": "metrics",
                "dataset": nombre_conjunto,
                "precision": precision_score(y_real, y_estimado),
                "balanced_accuracy": balanced_accuracy_score(y_real, y_estimado),
                "recall": recall_score(y_real, y_estimado),
                "f1_score": f1_score(y_real, y_estimado),
            }
            lista_metricas.append(resultado)

        with open(ruta_salida, "w", encoding="utf-8") as archivo_json:
            for registro in lista_metricas:
                archivo_json.write(json.dumps(registro) + "\n")

    def agregar_matrices_confusion(
        modelo,
        X_entrenamiento,
        y_entrenamiento,
        X_prueba,
        y_prueba,
        ruta_salida="files/output/metrics.json",
    ):
        with open(ruta_salida, "r", encoding="utf-8") as archivo_json:
            registros = [json.loads(linea) for linea in archivo_json]

        conjuntos = [
            ("train", X_entrenamiento, y_entrenamiento),
            ("test", X_prueba, y_prueba),
        ]

        for nombre_conjunto, X_actual, y_real in conjuntos:
            y_estimado = modelo.predict(X_actual)
            matriz = confusion_matrix(y_real, y_estimado, labels=[0, 1])
            matriz_dict = {
                "type": "cm_matrix",
                "dataset": nombre_conjunto,
                "true_0": {
                    "predicted_0": int(matriz[0][0]),
                    "predicted_1": int(matriz[0][1]),
                },
                "true_1": {
                    "predicted_0": int(matriz[1][0]),
                    "predicted_1": int(matriz[1][1]),
                },
            }
            registros.append(matriz_dict)

        with open(ruta_salida, "w", encoding="utf-8") as archivo_json:
            for registro in registros:
                archivo_json.write(json.dumps(registro) + "\n")

    # -----------------------------------------
    # Flujo principal
    # -----------------------------------------

    ruta_entrada = "files/input/"
    tabla_train = cargar_datos(ruta_entrada + "train_data.csv.zip")
    tabla_test = cargar_datos(ruta_entrada + "test_data.csv.zip")

    tabla_train = depurar_datos(tabla_train)
    tabla_test = depurar_datos(tabla_test)

    X_prueba = tabla_test.drop(columns=["default"])
    y_prueba = tabla_test["default"]

    X_entrenamiento = tabla_train.drop(columns=["default"])
    y_entrenamiento = tabla_train["default"]

    flujo_modelo = construir_pipeline()
    mejor_estimador = optimizar_modelo(flujo_modelo, X_entrenamiento, y_entrenamiento)

    guardar_modelo(mejor_estimador)
    evaluar_y_guardar_metricas(
        mejor_estimador, X_entrenamiento, y_entrenamiento, X_prueba, y_prueba
    )
    agregar_matrices_confusion(
        mejor_estimador, X_entrenamiento, y_entrenamiento, X_prueba, y_prueba
    )


if __name__ == "__main__":
    pregunta_1()

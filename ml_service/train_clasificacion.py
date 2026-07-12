import os
import json
import pickle
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

sklearn.set_config(transform_output="pandas")

from tools.functions import (
    eliminar_nulos_objetivo,
    separar_objetivo_features,
    corregir_valores_negativos,
    FeatureEngineering,
    Winsorizer,
    CorrelationFilter
)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

url = "https://raw.githubusercontent.com/ramirezluna-david/proyecto_modelado_grp2/rama_david/data/dataset_clientes.csv"
data = pd.read_csv(url)
TARGET = "abandono"

data = eliminar_nulos_objetivo(data, target=TARGET)
data, y, X = separar_objetivo_features(data, target=TARGET, drop_duplicates=True)
X = corregir_valores_negativos(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline_base = [
    ('feature_engineering', FeatureEngineering()),
    ('imputador', SimpleImputer(strategy='median')),
    ('winsorizer', Winsorizer(limits=(0.05, 0.05))),
    ('correlation_filter', CorrelationFilter(threshold=0.9))
]

modelos_a_entrenar = {
    "arbol_decision": DecisionTreeClassifier(
        criterion='entropy', max_depth=4, min_samples_leaf=10, min_samples_split=10, random_state=42, class_weight='balanced'
    ),
    "regresion_logistica": LogisticRegression(
        C=0.01, penalty='l2', solver='saga', random_state=42, class_weight='balanced'
    ),
    "svm": SVC(
        C=0.23273922280628717, gamma='auto', probability=True, random_state=42, class_weight='balanced'
    )
}
metricas_globales = {}

for nombre_modelo, algoritmo in modelos_a_entrenar.items():
    
    pipeline_actual = Pipeline(steps=pipeline_base + [('modelo', algoritmo)])
    pipeline_actual.fit(X_train, y_train)
    
    y_pred = pipeline_actual.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    
    print(f" {nombre_modelo} terminado. Accuracy: {accuracy:.2%}")
    parametros_usados = pipeline_actual.named_steps['modelo'].get_params()
    metricas_globales[nombre_modelo] = {
        "accuracy": float(accuracy),
        "parametros_usados": parametros_usados,
        "reporte_completo": reporte
    }
    
    pickle.dump(pipeline_actual, open(f"models/modelo_{nombre_modelo}.pkl", "wb"))

with open("models/metricas_clasificacion_todas.json", "w") as f:
    json.dump(metricas_globales, f, indent=4)

print("\n Pipelines y métricas guardados en /models")
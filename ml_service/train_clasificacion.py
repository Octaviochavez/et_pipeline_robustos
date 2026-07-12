import os
import json
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import sklearn

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

pipeline_clasificacion = Pipeline(steps=[
    ('feature_engineering', FeatureEngineering()),
    ('imputador', SimpleImputer(strategy='median')), 
    ('winsorizer', Winsorizer(limits=(0.05, 0.05))),
    ('correlation_filter', CorrelationFilter(threshold=0.9)),
    ('modelo', DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=10, min_samples_split=10, random_state=42, class_weight='balanced'))
])
pipeline_clasificacion.fit(X_train, y_train)
y_pred = pipeline_clasificacion.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
reporte = classification_report(y_test, y_pred, output_dict=True)

print(f"Modelo de Clasificación entrenado con éxito.")
print(f"Accuracy del modelo: {accuracy:.2%}")

modelo_entrenado = pipeline_clasificacion.named_steps['modelo']
metricas_clasificacion = {
    "accuracy": float(accuracy),
    "parametros_usados": {
        "criterion": modelo_entrenado.criterion,
        "max_depth_usado": modelo_entrenado.max_depth,
        "min_samples_leaf_usado": modelo_entrenado.min_samples_leaf,
        "min_samples_split_usado": modelo_entrenado.min_samples_split
    },
    "reporte_completo": reporte
}

with open("models/metricas_clasificacion.json", "w") as f:
    json.dump(metricas_clasificacion, f, indent=4)

pickle.dump(pipeline_clasificacion, open("models/modelo_clasificacion.pkl", "wb"))
print("Modelo de clasificación guardado")
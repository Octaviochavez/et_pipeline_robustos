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
    CorrelationFilter,
    build_preprocessor,
    build_model_pipeline_classifier
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
numerical_features = ["deuda_total", "porcentaje_gasto", "ratio_endeudamiento", "gasto_mensual", "score_crediticio", "ingreso_mensual", "edad", "antiguedad_meses", "frecuencia_compra", "ultima_compra_dias", "num_productos", "hora_registro"]
categorical_nominales = ["tiene_tarjeta_credito", "genero", "region", "estado_civil", "canal_registro", "dia_semana_registro"]
categorical_ordinales = ["tipo_plan", "uso_app"]
date_time_features = ["fecha_registro"]

orden_tipo_plan = ["Basico", "Estandar", "Premium"]
orden_uso_app = ["Bajo", "Medio", "Alto"]

_, _, _, preprocesador = build_preprocessor(
    numerical_features=numerical_features,
    categorical_nominales=categorical_nominales,
    categorical_ordinales=categorical_ordinales,
    orden_tipo_plan=orden_tipo_plan,
    orden_uso_app=orden_uso_app
)

modelos_a_entrenar = {
    "arbol_decision": build_model_pipeline_classifier(
        DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=10, min_samples_split=10, random_state=42, class_weight='balanced')
    ),
    "regresion_logistica": build_model_pipeline_classifier(
        LogisticRegression(C=0.01, penalty='l2', solver='saga', random_state=42, class_weight='balanced')
    ),
    "svm": build_model_pipeline_classifier(
        SVC(C=0.23273922280628717, gamma='auto', probability=True, random_state=42, class_weight='balanced')
    )
}

metricas_globales = {}

for nombre_modelo, pipeline_modelo in modelos_a_entrenar.items():
    print(f"Entrenando modelo: {nombre_modelo}...")
    
    pipeline_actual = Pipeline(steps=[
        ('feature_engineering', FeatureEngineering()),
        ('preprocesamiento', preprocesador),
        ('clasificador', pipeline_modelo)
    ])
    
    pipeline_actual.fit(X_train, y_train)
    
    y_pred = pipeline_actual.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"{nombre_modelo} terminado. Accuracy: {accuracy:.2%}")
    
    parametros_usados = pipeline_actual.named_steps['clasificador'].named_steps['modelo'].get_params()
    
    metricas_globales[nombre_modelo] = {
        "accuracy": float(accuracy),
        "parametros_usados": parametros_usados,
        "reporte_completo": reporte
    }
    
    pickle.dump(pipeline_actual, open(f"models/modelo_{nombre_modelo}.pkl", "wb"))

with open("models/metricas_clasificacion_todas.json", "w") as f:
    json.dump(metricas_globales, f, indent=4)

print("Entrenamiento completado.")
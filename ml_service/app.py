import os
import sys
import json
import pickle
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.functions import FeatureEngineering, Winsorizer, CorrelationFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Servicio Integrado de Machine Learning (Segmentación y Clasificación)")

try:
    modelo_kmeans = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
    scaler_kmeans = pickle.load(open("models/scaler.pkl", "rb"))
    with open("models/metricas.json") as f:
        metricas_segmentacion = json.load(f)
        
    modelos_clasificacion = {
        "arbol_decision": pickle.load(open("models/modelo_arbol_decision.pkl", "rb")),
        "regresion_logistica": pickle.load(open("models/modelo_regresion_logistica.pkl", "rb")),
        "svm": pickle.load(open("models/modelo_svm.pkl", "rb"))
    }
    with open("models/metricas_clasificacion.json") as f:
        metricas_clasificacion = json.load(f)
        
    logger.info("¡Todos los modelos y artefactos se cargaron exitosamente en memoria!")
except Exception as e:
    logger.critical(f"Error crítico al cargar los modelos de ML: {e}")
    raise RuntimeError(f"No se pudo iniciar el servicio de ML debido a fallas en los archivos .pkl/.json: {e}")


@app.get("/")
def inicio():
    """Comprobación de que el microservicio multi-modelo está activo."""
    return {
        "mensaje": "Servicio de Machine Learning (Segmentación y Clasificación) funcionando correctamente."
    }

@app.get("/dashboard-data")
def dashboard_data():
    """Expone los datos y métricas necesarios para los gráficos del Dashboard."""
    try:
        usuarios = pd.read_csv("data/usuarios_segmentados.csv")
        centroides = pd.read_csv("data/centroides.csv")
        len_usuarios = len(usuarios)

        if len_usuarios > 0:
            logger.info(f"Datos de dashboard consultados. Total registros: {len_usuarios}")
        else:
            logger.warning("El archivo usuarios_segmentados.csv está vacío.")

        return {
            "usuarios": usuarios.to_dict(orient="records"),
            "centroides": centroides.to_dict(orient="records"),
            "metricas_segmentacion": metricas_segmentacion,
            "metricas_clasificacion": metricas_clasificacion
        }

    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado en dashboard_data: {e}")
        raise HTTPException(status_code=404, detail=f"Archivo de datos no encontrado: {str(e)}")
    except Exception as e:
        logger.error(f"Error inesperado al cargar datos de dashboard: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar los datos del dashboard")

@app.post("/predict/segmentacion")
def predict_segmentacion(datos: dict):
    """
    Recibe los datos crudos de un cliente y le asigna un cluster (KMeans).
    """
    try:
        data_df = pd.DataFrame([datos])
        # Usamos el scaler y el modelo cargados para segmentación
        X_scaled = scaler_kmeans.transform(data_df)
        cluster = modelo_kmeans.predict(X_scaled)
        return {"cluster": int(cluster[0])}

    except ValueError as e:
        logger.error(f"Error de consistencia en datos de segmentación: {e}")
        raise HTTPException(status_code=400, detail=f"Entrada inválida para segmentación: {e}")
    except KeyError as e:
        logger.error(f"Falta columna obligatoria en segmentación: {e}")
        raise HTTPException(status_code=400, detail=f"Falta columna obligatoria: {e}")
    except Exception as e:
        logger.error(f"Error inesperado en endpoint de segmentación: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción de segmentación")


@app.post("/predict/clasificacion")
def predict_clasificacion(datos: dict, modelo: str = "svm"):
    """
    Recibe las características de un cliente y predice su probabilidad de abandono usando 
    el modelo seleccionado (arbol_decision, regresion_logistica, o svm).
    """
    try:
        if modelo not in modelos_clasificacion:
            raise HTTPException(
                status_code=400, 
                detail=f"Modelo '{modelo}' no encontrado. Usa: arbol_decision, regresion_logistica, o svm."
            )
        
        pipeline_actual = modelos_clasificacion[modelo]

        data_df = pd.DataFrame([datos])
        prediccion = pipeline_actual.predict(data_df)
        probabilidades = pipeline_actual.predict_proba(data_df)
        
        return {
            "modelo_usado": modelo, 
            "clase_predicha": int(prediccion[0]),
            "probabilidad_clase": float(probabilidades.max())
        }

    except HTTPException as he:
        raise he
    except ValueError as e:
        logger.error(f"Error de consistencia en datos de clasificación: {e}")
        raise HTTPException(status_code=400, detail=f"Entrada inválida para clasificación: {e}")
    except KeyError as e:
        logger.error(f"Falta columna obligatoria en clasificación: {e}")
        raise HTTPException(status_code=400, detail=f"Falta columna obligatoria: {e}")
    except Exception as e:
        logger.error(f"Error inesperado en endpoint de clasificación: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción de clasificación")
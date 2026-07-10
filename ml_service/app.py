import pandas as pd
import json
import pickle
from fastapi import FastAPI, HTTPException
import logging

app = FastAPI(title="Servicio Segmentación de Usuarios")
data = pd.read_csv("data/usuarios_segmentados.csv")
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
logger = logging.getLogger(__name__) #Genera los prints de consolas en pantalla

with open("models/metricas.json") as f:
    metricas = json.load(f)

@app.get("/")
def inicio():
    """
    inicio(): Es un endpoint GET en / que responde con un JSON simple
    {"mensaje": "Servicio ML funcionando"}. Es una comprobación de que el servicio está activo
    Returns:
        mensaje: Comprobación del funcionamiento del servicio.
    """    
    return {
        "mensaje":
        "Servicio ML funcionando"
    }

@app.get("/dashboard-data")
def dashboard_data():
    """
    dashboard_data(): Es un endpoint GET en /dashboard-data que carga: 
    - usuarios_segmentados.csv como usuarios
    - carga centroides.csv como centroides
    * Es la funcion que expone los datos necesarios para el dashboard
    Returns:
        - usuarios como lista de registros.
        - Centroides como lista de registros.
        - metricas ya cargadas en memoria.
    """    
    usuarios = pd.read_csv(
        "data/usuarios_segmentados.csv"
    )
    try:
        usuarios = pd.read_csv("data/usuarios_segmentados.csv")
        centroides = pd.read_csv("data/centroides.csv")
        len_usuarios = len(usuarios)

        if len_usuarios > 0:
            logger.info(f"Cantidad total de datos: {len_usuarios}")
        else:
            logger.warning("El archivo usuarios_segmentados.csv está vacío.")

    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
        return {"error": "Archivo de datos no encontrado", "detalle": str(e)}
    except pd.errors.EmptyDataError as e:
        logger.error(f"Datos vacíos en el archivo CSV: {e}")
        return {"error": "Datos vacíos en el archivo CSV", "detalle": str(e)}
    except Exception as e:
        logger.error(f"Error al cargar los datos: {e}")
        return {"error": "Error al cargar los datos", "detalle": str(e)}

    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "metricas": metricas
    }

@app.post("/predict")
def predict(datos: dict):
    """
    predict(): Es un endpoint POST en /predict que recibe un JSON con los datos del usuario, convierte
    esa entrada en DataFrame, normaliza las caracteristicas con scaler.transform() y luego usa modelo.predict()
    para asignar un cluster
    - Args:
        datos (dict):
        Un diccionario con las características del usuario que se usan para crear el dataframe
        de entrada antes de escalar y predecir el cluster.

    Returns:
        {"cluster": int(cluster[0])}: Devuelve el resultado como JSON.
    """
    try:
        data = pd.DataFrame([datos])
        X = scaler.transform(data)
        cluster = modelo.predict(X)
        return {"cluster": int(cluster[0])}

    except ValueError as e:
        logger.error(f"Error de valores en la predicción: {e}")
        raise HTTPException(status_code=400, detail=f"Entrada inválida para la predicción: {e}") #Devuelve HTTP 400 si los datos no son válidos para scaler.transform() o Dataframe.
    except KeyError as e:
        logger.error(f"Falta una característica obligatoria: {e}")
        raise HTTPException(status_code=400, detail=f"Falta una característica obligatoria: {e}") #Devuelve HTTP 400 si falta alguna característica esperada en el JSON.
    except Exception as e:
        logger.error(f"Error inesperado en predict(): {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la predicción") #Devuelve HTTP 500 para errores internos inesperados.
import streamlit as st
import pandas as pd
import requests
import pickle
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuración de página para que use todo el ancho de la pantalla
st.set_page_config(page_title="Dashboard ML", layout="wide")

st.title("Plataforma de Modelos de Machine Learning")

# 1. Obtiene los datos desde la API
try:
    respuesta = requests.get("http://ml_service:8000/dashboard-data")
    respuesta.raise_for_status() # Verifica si hay error 500 o 404
    payload = respuesta.json()
    
    data = pd.DataFrame(payload["usuarios"])
    centroides = pd.DataFrame(payload["centroides"])
    metricas_seg = payload["metricas_segmentacion"]
    metricas_clas = payload["metricas_clasificacion"]
    
except Exception as e:
    st.error(f"Error al conectar con la API de Machine Learning: {e}")
    st.stop() # Detiene la ejecución si no hay datos

# 2. Selector de sección (evita ejecutar ambas vistas en cada rerun)
seccion = st.radio(
    "Sección",
    ["Segmentación (KMeans)", "Clasificación (Predicción Manual)"],
    horizontal=True,
)

# Modelo de Segmentación (KMeans)
if seccion == "Segmentación (KMeans)":
    st.header("Análisis de Segmentación de Usuarios")
    
    st.subheader("Métricas del modelo")
    st.markdown("**Filtrar por Cluster:**")
    lista_clusters = sorted(data["cluster"].unique())
    clusters_seleccionados = []

    columnas_filtros = st.columns(len(lista_clusters))

    for i, cluster in enumerate(lista_clusters):
        with columnas_filtros[i]:
            if st.checkbox(f"Cluster {cluster}", value=True):
                clusters_seleccionados.append(cluster)

    if not clusters_seleccionados:
        st.warning("Ningún cluster seleccionado.")
        data_filtrada = pd.DataFrame(columns=data.columns)
    else:
        data_filtrada = data[data["cluster"].isin(clusters_seleccionados)]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silhouette Score Global", f"{metricas_seg['silhouette_score']:.3f}")
    with col2:
        st.metric("Clusters Seleccionados", len(clusters_seleccionados))
    with col3:
        st.metric("Usuarios en Selección", len(data_filtrada))

    st.subheader("Usuarios segmentados")
    st.dataframe(data_filtrada) 

    st.subheader("Distribución de segmentos")
    st.bar_chart(data_filtrada["cluster"].value_counts()) 

    # Perfil de cada segmento
    perfil_segmentos = data.groupby("cluster").agg(
        usuarios=("id_cliente", "count"),
        edad_promedio=("edad", "mean"),
        dispositivos_registrados_promedio=("dispositivos_registrados", "mean"),
        porcentaje_uso_app_movil_promedio=("porcentaje_uso_app_movil", "mean"),
        cantidad_perfiles_creados_promedio=("cantidad_perfiles_creados", "mean"),
        interacciones_mensuales_soporte_promedio=("interacciones_mensuales_soporte", "mean"),
        distancia_promedio_red_km=("distancia_promedio_red_km", "mean"),
        horas_consumo_mensual_promedio=("horas_consumo_mensual", "mean"),
        gasto_mensual_prom=("gasto_mensual", "mean"),
        cantidad_contenidos_vistos_promedio=("cantidad_contenidos_vistos", "mean"),
        sesiones_semana_promedio=("sesiones_semana", "mean"),
        porcentaje_finalizacion_promedio=("porcentaje_finalizacion", "mean"),
        tiempo_promedio_sesion_min=("tiempo_promedio_sesion_min", "mean"),
        cantidad_generos_consumidos_promedio=("cantidad_generos_consumidos", "mean"),
        porcentaje_uso_promociones_promedio=("porcentaje_uso_promociones", "mean"),
        antiguedad_cliente_meses=("antiguedad_cliente_meses", "mean")
    ).round(2)

    st.subheader("Perfil de segmentos (Mapa de Calor)")
    st.dataframe(perfil_segmentos.style.background_gradient(cmap='Blues', axis=0))

    # Grafica resultados de PCA interactivo
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in sorted(data_filtrada["cluster"].unique()): 
        subset = data_filtrada[data_filtrada["cluster"] == cluster]
        ax.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster}", alpha=0.7)

    ax.set_title("Visualización PCA de los segmentos", fontsize=14, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=14, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Visualización de segmentos usando 2 características")
    # Escoge solo columnas que existan en ambas tablas y excluye identificadores
    columnas_numericas = [col for col in data.select_dtypes(include=['float64', 'int64']).columns 
                          if col in centroides.columns and col not in ['cluster', 'id_cliente']]

    col1, col2 = st.columns(2)
    with col1:
        columna_x = st.selectbox("Selecciona Eje X", columnas_numericas, index=columnas_numericas.index('horas_consumo_mensual'))
    with col2:
        columna_y = st.selectbox("Selecciona Eje Y", columnas_numericas, index=columnas_numericas.index('gasto_mensual'))

    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.scatter(data[columna_x], data[columna_y], c=data["cluster"], alpha=0.7, s=40)
    ax2.scatter(centroides[columna_x], centroides[columna_y], marker="X", s=250, edgecolor="black", linewidth=2)
    
    ax2.set_xlabel(columna_x, fontsize=14, fontweight="bold")
    ax2.set_ylabel(columna_y, fontsize=14, fontweight="bold")
    ax2.set_title(f"Clusters según {columna_x} y {columna_y}", fontsize=16, fontweight="bold")
    ax2.grid(True)
    st.pyplot(fig2)


# Modelo de Clasificación (Árbol de Decisión)
if seccion == "Clasificación (Predicción Manual)":
    st.header("Modelo de Clasificación")
    
    opciones_modelo = {
        "Árbol de Decisión": "arbol_decision",
        "Regresión Logística": "regresion_logistica",
        "Support Vector Machine (SVM)": "svm"
    }
    
    modelo_seleccionado_nombre = st.selectbox(
        "Selecciona el algoritmo predictivo:", 
        list(opciones_modelo.keys()),
        key="selector_modelo"
    )
    modelo_seleccionado_key = opciones_modelo[modelo_seleccionado_nombre]
    if 'metricas_clas' in locals() and modelo_seleccionado_key in metricas_clas:
        metrics_actual = metricas_clas[modelo_seleccionado_key]
        st.subheader(f"Rendimiento del Modelo: {modelo_seleccionado_nombre}")
        c1, c2, c3, c4 = st.columns(4)
        metrics_actual = metricas_clas.get(modelo_seleccionado_key, {})
        params_actual = metrics_actual.get("parametros_usados", {})
        c1.metric("Accuracy Global", f"{metrics_actual.get('accuracy', 0.0):.2%}")
    else:
        st.warning("Los datos del modelo seleccionado no están disponibles. Recargando...")
        st.stop()


    if modelo_seleccionado_key == "arbol_decision":
        c2.metric("Max Depth", params_actual.get("max_depth", "N/A"))
        c3.metric("Min Samples Leaf", params_actual.get("min_samples_leaf", "N/A"))
        c4.metric("Criterion", str(params_actual.get("criterion", "N/A")).capitalize())
    elif modelo_seleccionado_key == "regresion_logistica":
        c2.metric("C (Regul.)", params_actual.get("C", "N/A"))
        c3.metric("Penalty", str(params_actual.get("penalty", "N/A")).upper())
        c4.metric("Solver", str(params_actual.get("solver", "N/A")).capitalize())
    elif modelo_seleccionado_key == "svm":
        val_c = params_actual.get("C", 0)
        c2.metric("C (Regul.)", f"{val_c:.4f}" if isinstance(val_c, float) else val_c)
        c3.metric("Gamma", str(params_actual.get("gamma", "N/A")).capitalize())
        c4.metric("Kernel", str(params_actual.get("kernel", "N/A")).capitalize())
    
    st.divider()
    
    st.subheader("Realizar una Predicción Manual")
    st.markdown("Ingresa las características del usuario para predecir a qué clase pertenece.")
    
    with st.form("formulario_prediccion"):
        col_form1, col_form2, col_form3 = st.columns(3)
        
        with col_form1:
            input_edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
            input_ingreso = st.number_input("Ingreso Mensual", min_value=0.0, value=1500.0)
            input_gasto = st.number_input("Gasto Mensual", min_value=0.0, value=500.0)
            input_deuda = st.number_input("Deuda Total", min_value=0.0, value=100.0)
            
        with col_form2:
            input_score = st.number_input("Score Crediticio", min_value=0, max_value=1000, value=600)
            input_antiguedad = st.number_input("Antigüedad (Meses)", min_value=0, value=12)
            input_ratio = st.number_input("Ratio Endeudamiento", min_value=0.0, value=0.1)
            input_porcentaje_gasto = st.number_input("Porcentaje Gasto", min_value=0.0, value=0.3)

        with col_form3:
            input_freq = st.number_input("Frecuencia Compra", min_value=0, value=5)
            input_u_compra = st.number_input("Última Compra (Días)", min_value=0, value=10)
            input_n_prod = st.number_input("Num Productos", min_value=0, value=2)
            input_hora = st.number_input("Hora Registro", min_value=0, max_value=23, value=12)
            
        st.markdown("### Variables Categóricas")
        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            input_tarjeta = st.selectbox("Tiene Tarjeta Crédito", [1, 0])
            input_genero = st.selectbox("Género", ["Masculino", "Femenino"])
            input_region = st.selectbox("Región", ["Metropolitana", "Valparaíso", "Biobío", "Otra"])
            input_plan = st.selectbox("Tipo Plan", ["Basico", "Estandar", "Premium"])
        with col_cat2:
            input_civil = st.selectbox("Estado Civil", ["Soltero", "Casado", "Divorciado", "Viudo"])
            input_canal = st.selectbox("Canal Registro", ["Web", "App", "Presencial"])
            input_dia = st.selectbox("Día Semana Registro", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
            input_uso = st.selectbox("Uso App", ["Bajo", "Medio", "Alto"])
            
        submit_button = st.form_submit_button(label="Clasificar Usuario", use_container_width=True)
        
    if submit_button:
        datos_cliente = {
            "edad": input_edad,
            "ingreso_mensual": input_ingreso,
            "gasto_mensual": input_gasto,
            "deuda_total": input_deuda,
            "score_crediticio": input_score,
            "antiguedad_meses": input_antiguedad,
            "ratio_endeudamiento": input_ratio,
            "porcentaje_gasto": input_porcentaje_gasto,
            "frecuencia_compra": input_freq,
            "ultima_compra_dias": input_u_compra,
            "num_productos": input_n_prod,
            "hora_registro": input_hora,
            "tiene_tarjeta_credito": input_tarjeta,
            "genero": input_genero,
            "region": input_region,
            "estado_civil": input_civil,
            "canal_registro": input_canal,
            "dia_semana_registro": input_dia,
            "tipo_plan": input_plan,
            "uso_app": input_uso
        }
        
        with st.spinner("Procesando predicción..."):
            try:
                url_api = f"http://ml_service:8000/predict/clasificacion?modelo={modelo_seleccionado_key}"
                res = requests.post(url_api, json=datos_cliente)
                
                if res.status_code == 200:
                    resultado = res.json()
                    st.success(f"Predicción Exitosa: El usuario pertenece a la Clase {resultado['clase_predicha']}")
                    st.info(f"Nivel de Certeza (Probabilidad): {resultado['probabilidad_clase']:.2%}")
                else:
                    error_msg = res.json().get("detail", "Error desconocido")
                    st.error(f"Error en la API: {error_msg}")
                    
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con el servicio de Machine Learning. ¿Están encendidos los contenedores?")
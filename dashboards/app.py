import streamlit as st
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt

st.title("Segmentación de Usuarios")

# Obtiene los datos para la visualización
respuesta = requests.get(
    "http://ml_service:8000/dashboard-data"
)

payload = respuesta.json()

data = pd.DataFrame(payload["usuarios"])
metricas = payload["metricas"]
centroides = pd.DataFrame(payload["centroides"])

# Muestra las métricas del modelo
st.subheader("Métricas del modelo")

# Filtro interactivo por segmento
st.markdown("**Filtrar por Cluster:**")
lista_clusters = sorted(data["cluster"].unique())
clusters_seleccionados = []

columnas_filtros = st.columns(len(lista_clusters))

for i, cluster in enumerate(lista_clusters):
    with columnas_filtros[i]:
        # value=True mantiene los clusters seleccionados por defecto al cargar
        if st.checkbox(f"Cluster {cluster}", value=True):
            clusters_seleccionados.append(cluster)

if not clusters_seleccionados:
    st.warning("Ningún cluster seleccionado.")
    data_filtrada = pd.DataFrame(columns=data.columns) # Data vacía
else:
    # Creamos la data filtrada basada en la selección
    data_filtrada = data[data["cluster"].isin(clusters_seleccionados)]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Silhouette Score Global",
        f"{metricas['silhouette_score']:.3f}"
    )

with col2:
    st.metric(
        "Clusters Seleccionados",
        len(clusters_seleccionados)
    )

with col3:
    st.metric(
        "Usuarios en Selección",
        len(data_filtrada)
    )

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
# Aplica un gradiente de color a la tabla para convertirla en un mapa de calor visual
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

# Muestra los cluster usando dos variables
# Carga el modelo
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
# Carga data escalada
scaler = pickle.load(open("models/scaler.pkl", "rb"))
centroides = pd.DataFrame(
    payload["centroides"]
)
st.subheader("Visualización de segmentos usando 2 características")

# Escoge solo columnas que existan en ambas tablas y excluye identificadores
columnas_numericas = [col for col in data.select_dtypes(include=['float64', 'int64']).columns 
                      if col in centroides.columns and col not in ['cluster', 'id_cliente']]

col1, col2 = st.columns(2)
with col1:
    columna_x = st.selectbox("Selecciona Eje X", columnas_numericas, index=columnas_numericas.index('horas_consumo_mensual'))
with col2:
    columna_y = st.selectbox("Selecciona Eje Y", columnas_numericas, index=columnas_numericas.index('gasto_mensual'))

fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(
    data[columna_x],
    data[columna_y],
    c=data["cluster"],
    alpha=0.7,
    s=40
)

ax.scatter(
    centroides[columna_x],
    centroides[columna_y],
    marker="X",
    s=250,
    edgecolor="black",
    linewidth=2
)

ax.set_xlabel(columna_x, fontsize=14, fontweight="bold")
ax.set_ylabel(columna_y, fontsize=14, fontweight="bold")

ax.set_title(
    f"Clusters según {columna_x} y {columna_y}", fontsize=16, fontweight="bold"
)

ax.grid(True)

st.pyplot(fig)
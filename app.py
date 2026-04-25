import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

#PRIMERO CONFIGURAMOS NUESTRA PAGINA Y EL ESTILO
st.set_page_config(
    page_title="Optimizador de Carga Inteligente PVPC",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

#CARGAMOS LOS DATOS Y EL MODELO PREDICITIVO
@st.cache_resource
def load_assets():
    
    model = joblib.load('modelo_lgbm.pkl')
    X_future = pd.read_parquet('X_test.parquet').iloc[-24:]
    return model, X_future

try:
    model, X_future = load_assets()
except Exception as e:
    st.error(f"Error al cargar los archivos: {e}. Asegúrate de que 'modelo_lgbm.pkl' y 'X_test.parquet' estén en la misma carpeta.")
    st.stop()

#QUE EL USUARIO CARGE LOS DATOS
st.sidebar.header("Configuración de Carga")
capacidad = st.sidebar.number_input("Capacidad de la batería (kWh)", min_value=1.0, value=50.0, step=1.0)
soc_actual = st.sidebar.slider("Nivel de carga actual (%)", 0, 100, 20)
soc_objetivo = st.sidebar.slider("Nivel de carga deseado (%)", 0, 100, 80)
potencia_cargador = st.sidebar.number_input("Potencia del cargador (kW)", min_value=1.1, max_value=22.0, value=7.4, step=0.1)

#CALCULAMOS LA ENERGIA QUE HACE FALTA Y EL TIEMPO
energia_a_cargar = capacidad * (soc_objetivo - soc_actual) / 100
horas_necesarias = int(np.ceil(energia_a_cargar / potencia_cargador))

#USAMOS EL ALGORITMO DE PREDICCION
X_input = X_future.select_dtypes(include=['number'])
precios_predichos = model.predict(X_input)


df_precios = pd.DataFrame({
    'Hora': [f"{h}:00" for h in range(24)],
    'Precio': precios_predichos
})

#BUSCAMOS EL MINIMO Y EL MÁXIMO
costes_ventana = []
for i in range(len(df_precios) - horas_necesarias + 1):
    ventana = df_precios['Precio'].iloc[i : i + horas_necesarias]
    coste_total = ventana.sum() * (energia_a_cargar / horas_necesarias) 
    costes_ventana.append(coste_total)

idx_mejor = np.argmin(costes_ventana)
idx_peor = np.argmax(costes_ventana)

coste_optimo = costes_ventana[idx_mejor]
coste_maximo = costes_ventana[idx_peor]
ahorro = coste_maximo - coste_optimo
ahorro_perc = (ahorro / coste_maximo) * 100

# CUERPO DESHBOARD
st.title("Optimizador de carga")
st.markdown(f"Objetivo: Cargar **{energia_a_cargar:.2f} kWh** (aprox. **{horas_necesarias} horas** de conexión).")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mejor Hora de Inicio", df_precios['Hora'].iloc[idx_mejor])
with col2:
    st.metric("Coste Estimado", f"{coste_optimo:.2f} €", f"-{ahorro:.2f} € vs peor caso")
with col3:
    st.metric("Ahorro Potencial", f"{ahorro_perc:.1f} %", delta_color="normal")

#GRAFICOS DE PRECIOS Y TABLA DE DETALLE
st.subheader("Curva de Precios Predicha (Próximas 24h)")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_precios['Hora'], df_precios['Precio'], color='#007bff', marker='o', label='Precio Predicho (€/kWh)')
# Resaltar la mejor ventana
ax.axvspan(idx_mejor, idx_mejor + horas_necesarias - 1, color='green', alpha=0.2, label='Ventana Óptima')
# Resaltar la peor ventana
ax.axvspan(idx_peor, idx_peor + horas_necesarias - 1, color='red', alpha=0.1, label='Ventana más Cara')

ax.set_ylabel("Precio (€/kWh)")
ax.set_xticks(range(24))
ax.set_xticklabels(df_precios['Hora'], rotation=45)
ax.legend()
st.pyplot(fig)

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.subheader("📋 Planificación Detallada")
    st.write(f"- **Hora de conexión recomendada:** {df_precios['Hora'].iloc[idx_mejor]}")
    st.write(f"- **Hora de finalización estimada:** {idx_mejor + horas_necesarias}:00")
    st.write(f"- **Energía total a transferir:** {energia_a_cargar:.2f} kWh")
    
with c2:
    st.subheader("💰 Comparativa de Costes")
    data_comp = {
        "Escenario": ["Optimizado (Tu Modelo)", "Peor Momento (Pico)", "Diferencia/Ahorro"],
        "Coste Total (€)": [f"{coste_optimo:.2f} €", f"{coste_maximo:.2f} €", f"{ahorro:.2f} €"]
    }
    st.table(pd.DataFrame(data_comp))

st.info("**Nota del TFM:** Este dashboard utiliza un modelo LightGBM con un R² de 0.9202 para predecir los precios del mercado PVPC y optimizar el gasto económico del usuario.")
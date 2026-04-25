# Predicción de Precios PVPC y Carga de Vehículo Eléctrico

Este repositorio tiene el Trabajo Fin de Máster (TFM) que está enfocado a la predicción del Precio Voluntario para el Pequeño Consumidor (PVPC) y la optimización de la demanda energética para la carga de vehículos electricos

## Demo en Vivo
Puedes ver la aplicación funcionando aquí: 
[https://huggingface.co/spaces/jcsoriano-data/tfm-pvpc-final-v3]

##  Estructura del Proyecto
El flujo de trabajo está organizado de forma secuencial para facilitar su comprensión:

1.  **Ingesta de Datos**: Scripts `01` y `02` para la creación de datasets desde fuentes oficiales (REE).
2.  **Limpieza y Preprocesamiento**: Scripts `03`, `04` y `05` donde se tratan nulos, duplicados y outliers.
3.  **Feature Engineering**: Script `06` para la creación de variables temporales y específicas.
4.  **Modelado**:
    * `09_prophet.py`: Modelo aditivo para series temporales.
    * `10_LightGBM.py`: Algoritmo de Gradient Boosting optimizado.
    * `11_GRU.py` y `12_GRU2.py`: Redes Neuronales Recurrentes (Gated Recurrent Units).
5.  **Comprobación**: Se comprueba que el modelo seleccionado no a caído en  que los res(`13_test_modelo.py`).
6. **Despliegue**: Interfaz interactiva desarrollada en **Streamlit** (`app.py`)

##  Tecnologías utilizadas
* **Lenguaje:** Python 3.x
* **Machine Learning:** Scikit-learn, LightGBM.
* **Deep Learning:** TensorFlow / Keras (Modelos GRU).
* **Time Series:** Prophet.
* **Dashboard:** Streamlit.
* **Análisis de datos:** Pandas, Numpy, Matplotlib, Seaborn.

##  Instalación
Si deseas ejecutar este proyecto localmente:

1. Clona el repositorio:
   ```bash
   git clone [https://github.com/JCSorianoMeseguer/TFM-PVPC-VEHICULO_ELECTRICO.git](https://github.com/JCSorianoMeseguer/TFM-PVPC-VEHICULO_ELECTRICO.git)

## En la consolo 
1 pip install -r requirements.txt
2. streamlit run 13_test_modelo.py

© 2026 - Desarrollado por Jose Carlos Soriano Meseguer como parte del Máster en Data Science.

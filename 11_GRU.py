import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization

def secuencias( X, y, salto_tiempo= 24):

    Xs, ys= [],[]
    for i in range(len(X)-salto_tiempo):
        Xs.append(X[i:(i + salto_tiempo)])
        ys.append(y[i:(i+ salto_tiempo)])
    return np.array(Xs), np.array(ys)

#CARGAMOS LAS VARIABLES DEPENDIENTES E INDEPENDIENTES

train_b= pd.read_parquet('X_train.parquet')
y_train_b = pd.read_parquet('y_train.parquet').values
val_b= pd.read_parquet('X_val.parquet')
y_val = pd.read_parquet('y_val.parquet').values.flatten()

for df in [train_b, val_b]:
    if 'FECHA' in df.columns:
        df.set_index('FECHA', inplace=True)
    df = df.select_dtypes(include=[np.number])

#ESCALAMOS LAS VARIABLES PARA QUE SEAN COMPARADAS DE FORMA ADECUADA POR GRU

scarler_X= RobustScaler()
scaler_y = RobustScaler()

X_train_scaler= scarler_X.fit_transform(train_b)
X_val_scaler= scarler_X.transform(val_b)

y_train_scaler= scaler_y.fit_transform(y_train_b.reshape(-1,1))
y_val_scaler= scaler_y.transform(y_val.reshape(-1,1))

#CREAMOS SECUENCIAS PARA GRU

X_train_gru, y_train_gru = secuencias(X_train_scaler, y_train_scaler)
X_val_gru, y_val_gru = secuencias(X_val_scaler, y_val_scaler)

#PESOS TEMPORALES, PARA QUE EL ALGORITMO LE DE MAS IMPORTANCIA A LOS PRECIOS ACTUALES

peso = np.linspace(0.1,1, len(y_train_gru))

#DEFINIMOS NUESTRO GRU

modelo_gru= Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train_gru.shape[1], X_train_gru.shape[2])),
    BatchNormalization(),
    Dropout(0.2),
    GRU(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
    ])

modelo_gru.compile(optimizer='adam', loss='mae')

#ENTRENAMIENTO
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = modelo_gru.fit(
    X_train_gru, y_train_gru,
    validation_data=(X_val_gru, y_val_gru),
    sample_weight= peso,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

#GRAFICA DE APRENDIZAJE

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida Entrenamiento (Pesada)')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.axvline(x=len(history.history['loss'])-11, color='r', linestyle='--', label='Punto de Convergencia')
plt.title('Evolución del Aprendizaje (Loss Curve)')
plt.xlabel('Épocas')
plt.ylabel('MAE (Escalado)')
plt.legend()
plt.grid(True)
plt.show()

#DESESCALAMOS EL MODELO Y CALCULAMOS METRICAS

y_pred_scale = modelo_gru.predict(X_val_gru)

y_pred = scaler_y.inverse_transform(y_pred_scale)

y_real = y_val[24:]

mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
r2 = r2_score(y_real, y_pred)


metricas_df = pd.DataFrame({
    'Modelo': ['GRU_Ponderada_Robust'],
    'MAE': [mae],
    'RMSE': [rmse],
    'MAPE_%': [mape],
    'R2': [r2]
})

plt.figure(figsize=(12, 6))


#VEMOS LA PREDICCION RESPECTO AL PRECIO REAL
plt.plot(y_real[:200], label='Precio Real PVPC', color='#1f77b4', linewidth=2, alpha=0.8)
plt.plot(y_pred[:200], label='Predicción GRU', color='#d62728', linestyle='--', linewidth=2)


plt.title('Modelo GRU 64 neuronas: Precio Real vs. Predicción (Muestra de Validación)', fontsize=14, fontweight='bold')
plt.xlabel('Tiempo (Horas)', fontsize=12)
plt.ylabel('Precio PVPC (€/MWh)', fontsize=12)
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.6)


plt.tight_layout()
plt.savefig('grafica_gru_pvpc.png', dpi=300) # Guarda la imagen con calidad para el Word/LaTeX del TFM
plt.show()

metricas_df.to_csv('metricas_gru.csv', index=False, sep=';')




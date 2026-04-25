import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

#CARGAMOS LAS VARIABLES DEPENDIENTES E INDEPENDIENTES

train_b= pd.read_parquet('X_train.parquet')
y_train_b = pd.read_parquet('y_train.parquet').values
val_b= pd.read_parquet('X_val.parquet')

# Añadimos los regresores necesarios
regresores = [
    'PRECIO_SEMANA_ANTERIOR', 'PRECIO_AYER', 'MEDIA_24H', 
    'POSICION_RELATIVA', 'VOLATILIDAD_6', 'DIA_SIN', 
    'DIA_COS', 'MEDIA_TRIMESTRAL', 'RANGO_PUNTA', 
    'MEDIA_SEMANA', 'DIFERENCIA_SEMANA', 'HORA_COS', 
    'HORA_SIN', 'PRE_2021', 'DEMANDA_PREVISTA'
]
train= pd.DataFrame()

train['ds'] = pd.date_range(start='2014-05-01 00:00:00', periods=len(train_b), freq='h')
train['y'] = y_train_b.flatten()
for col in regresores:
    train[col] = train_b[col].values



# Preparar validación igual
val = pd.DataFrame()
ultima_fecha_train = train['ds'].iloc[-1]
val['ds'] = pd.date_range(start=ultima_fecha_train + pd.Timedelta(hours=1), periods=len(val_b), freq='h')
for col in regresores:
    val[col] = val_b[col].values


print(train['ds'].head())
print(val['ds'].head())

'''CONFIGURAMOS EL MODELO 
Desactivamos la estacionalidad automática, para hacer que el modelo
capte mejor las estacionalidades, proponiendo unas más estrictas'''

model = Prophet(
    daily_seasonality=False, 
    weekly_seasonality=False, 
    yearly_seasonality=False,
    changepoint_prior_scale=0.5, 
    seasonality_prior_scale=10.0
)

model.add_seasonality(name='diaria', period=1, fourier_order=15)
model.add_seasonality(name='semanal', period=7, fourier_order=3)
model.add_seasonality(name='anual', period=365.25, fourier_order=10)

#AÑADIMOS LOS RANGOS PARA QUE EL MODELO ENTIENDA MEJOR ESTE PARAMETRO

for col in regresores:
    model.add_regressor(col)
model.add_country_holidays(country_name='ES')

model.fit(df=train)

forecast = model.predict(val)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

y_real = pd.read_parquet('y_val.parquet').values.flatten()
y_pred = forecast['yhat'].values

mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(12,6))
plt.plot(val['ds'][:168], y_real[:168], label='Real (PVPC)', color='blue')
plt.plot(val['ds'][:168], y_pred[:168], label='Predicción (Prophet)', color='red', linestyle='--')
plt.title('Comparativa PVPC: Real vs Predicción (1 semana)')
plt.legend()
plt.show()

model.plot_components(forecast)
plt.show()

mape = np.mean(np.abs((y_real - y_pred) / (+ 1e-10 + y_real))) * 100



tabla_resultados = pd.DataFrame({
    'Métrica': ['MAE (Error Absoluto Medio)', 
                'RMSE (Raíz del Error Cuadrático Medio)', 
                'MAPE (Error Porcentual)', 
                'R² (Coeficiente de Determinación)'],
    'Valor Prophet': [
        f"{mae:.4f} €/kWh", 
        f"{rmse:.4f} €/kWh", 
        f"{mape:.2f} %", 
        f"{r2:.4f}"
    ]
})

print(tabla_resultados)

tabla_resultados.to_csv('metricas_prophet.csv', index=False)
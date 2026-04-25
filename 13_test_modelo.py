import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CARGAMOS LOS DATOS DEL TEST
X_test_b = pd.read_parquet('X_test.parquet')
y_test = pd.read_parquet('y_test.parquet').values.flatten()

X_test = X_test_b.select_dtypes(include=['number'])

#CARGAMOS EL MODELO LIGHTGBM Y VEMOS LAS ESTADISTICAS

modelo_lgbm = joblib.load('modelo_lgbm.pkl')

y_pred = modelo_lgbm.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

df_resultados_test = pd.DataFrame({
    'Métrica': ['MAE', 'RMSE', 'MAPE (%)', 'R² Score'],
    'Valor en Test (Datos No Vistos)': [
        f"{mae:.4f}", 
        f"{rmse:.4f}", 
        f"{mape:.2f}%", 
        f"{r2:.4f}"
    ]
})

df_resultados_test.to_csv('resultados_finales_test_LGBM.csv', index=False)

print("--- TABLA DE RESULTADOS FINALES (TEST) ---")
print(df_resultados_test)

plt.figure(figsize=(12, 5))
plt.plot(y_test[:168], label='Real (1 semana)', color='blue')
plt.plot(y_pred[:168], label='Predicción', color='red', linestyle='--')
plt.title('Rendimiento del Modelo en la primera semana de Test')
plt.legend()
plt.show()






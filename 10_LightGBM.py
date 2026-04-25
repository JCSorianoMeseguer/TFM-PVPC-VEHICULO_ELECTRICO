import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import joblib

#CARGAMOS LAS VARIABLES DEPENDIENTES E INDEPENDIENTES

train_b= pd.read_parquet('X_train.parquet')
y_train_b = pd.read_parquet('y_train.parquet').values
val_b= pd.read_parquet('X_val.parquet')

#PREPARAMOS LAS VARIABLES Y EL MODEL 

X_train = train_b.select_dtypes(include=['number'])
X_val = val_b.select_dtypes(include=['number'])
y_train = y_train_b.flatten()


y_val = pd.read_parquet('y_val.parquet').values.flatten()

peso_tiempo= np.linspace(0.1,1,(len(y_train)))


train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'mae',         
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,         
    'feature_fraction': 0.9,
    'verbose': -1
}

model_lgb = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)


y_pred_lgb = model_lgb.predict(X_val)

#CALCUILAMOS METRICAS LightGBM
mae_lgb = mean_absolute_error(y_val, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
mape_lgb = np.mean(np.abs((y_val - y_pred_lgb) / y_val)) * 100
r2_lgb = r2_score(y_val, y_pred_lgb)


comparativa = pd.DataFrame({
    'Métrica': ['MAE', 'RMSE', 'MAPE (%)', 'R²'],
    'LightGBM': [mae_lgb, rmse_lgb, mape_lgb, r2_lgb]
})

print("--- TABLA COMPARATIVA DE MODELOS ---")
print(comparativa.to_string(index=False))

comparativa.to_csv('metricas_ligthGBM.csv', index= False)


plt.figure(figsize=(10, 6))
lgb.plot_importance(model_lgb, importance_type='gain', precision=2)
plt.title("Importancia de las variables (Ganancia)")
plt.tight_layout()
plt.show()

#VEMOS LA PREDICCION RESPECTO AL PRECIO REAL
plt.plot(y_val[:200], label='Precio Real (PVPC)', color='royalblue', linewidth=2, alpha=0.8)
plt.plot(y_pred_lgb[:200], label='Predicción LightGBM', color='crimson', linestyle='--', linewidth=2)

plt.title(f'Comparativa: Precio Real vs Predicho', fontsize=14)
plt.xlabel('Tiempo (Horas)', fontsize=12)
plt.ylabel('Precio (€/MWh)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#GUARDAMOS EL MODELO PARA HACER EL TEST

joblib.dump(model_lgb, 'modelo_lgbm.pkl')

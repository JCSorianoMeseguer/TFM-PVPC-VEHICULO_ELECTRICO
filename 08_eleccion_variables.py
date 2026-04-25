import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb




#Cargamos el archivo
df = pd.read_parquet('pvpc_fin.parquet')


#Vamosa  ver la correlacion entre variables
cols_interes = ['PRECIO_KWh', 'DEMANDA_PREVISTA', 'FUTUROS', 'MEDIA_24H', 'MEDIA_SEMANA','MEDIA_TRIMESTRAL','POSICION_RELATIVA', 'VOLATILIDAD_6', 'PRECIO_AYER','PRECIO_SEMANA_ANTERIOR', 'DIFERENCIA_SEMANA']

correlacion = df[cols_interes].corr()

'''plt.figure(figsize=(10, 8))
sns.heatmap(correlacion, annot=True, cmap='RdBu', fmt=".2f")
plt.title("Correlación de Pearson - Variables Clave")
plt.show()'''

#Vamos a ver las variables que mas peso tienen para quedarnos con estas
#Creamos las targets
'''X= df.drop(columns= [ 'FECHA', 'PRECIO_KWh'])
y= df['PRECIO_KWh']

#Usamos el algoritmo LightGBM

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    importance_type='gain',
    random_state=42
)
model.fit(X, y)'''

importances = pd.DataFrame({
    'Variable': X.columns,
    'Ganancia_Total': model.feature_importances_
})

total_gain = importances['Ganancia_Total'].sum()
importances['Importancia_%'] = (importances['Ganancia_Total'] / total_gain) * 100


importances = importances.sort_values(by='Ganancia_Total', ascending=False).reset_index(drop=True)

#Hacemos un gráfico
'''plt.figure(figsize=(12, 8))
sns.barplot(x='Importancia_%', y='Variable', data=importances, palette='magma')
plt.title('Importancia Relativa de las Variables (en %)', fontsize=15)
plt.xlabel('Porcentaje de Ganancia (%)', fontsize=12)
plt.ylabel('Variables del Dataset', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()'''

#Creamos una tabla para la memoria


importances.to_csv('tabla importancia variables.csv', sep=';', encoding= 'utf-8')

#Creamos un nuevo archivo con las varibles seleccinadas

variables_seleccionadas = [
    'PRECIO_SEMANA_ANTERIOR', 'PRECIO_AYER', 'MEDIA_24H', 
    'POSICION_RELATIVA', 'VOLATILIDAD_6', 'DIA_SIN', 
    'DIA_COS', 'MEDIA_TRIMESTRAL', 'RANGO_PUNTA', 
    'MEDIA_SEMANA', 'DIFERENCIA_SEMANA', 'HORA_COS', 
    'HORA_SIN', 'PRE_2021', 'DEMANDA_PREVISTA'
]

# Creamos el dataset final de trabajo

df_modelo = df[['FECHA', 'PRECIO_KWh'] + variables_seleccionadas].copy()

df_modelo.to_parquet('pvpc_seleccion.parquet', engine='pyarrow')
df_modelo.to_csv('pvpc_seleccion.csv', sep=';', encoding= 'utf-8')
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from statsmodels.tsa.stattools import adfuller

#CARGAMOS EL ARCHIVO
ruta= os.path.dirname(__file__)
ruta_archivo= os.path.join(ruta, 'pvpc_bruto.parquet')
df = pd.read_parquet(ruta_archivo)

#TEST DE DICKEY-FULLER AUMENTADA (ADF)
resultado_adf = adfuller(df['PRECIO_KWh'])

print(f'Estadístico ADF: {resultado_adf[0]:.4f}')
print(f'p-valor: {resultado_adf[1]:.4e}')
print('Valores Críticos:')
for clave, valor in resultado_adf[4].items():
    print(f'   {clave}: {valor:.4f}')

datos_adf = {
    'Métrica': [
        'Estadístico ADF', 
        'p-valor', 
        'Retardos usados', 
        'Número de observaciones', 
        'Valor Crítico (1%)', 
        'Valor Crítico (5%)', 
        'Valor Crítico (10%)'
    ],
    'Valor': [
        resultado_adf[0],
        resultado_adf[1],
        resultado_adf[2],
        resultado_adf[3],
        resultado_adf[4]['1%'],
        resultado_adf[4]['5%'],
        resultado_adf[4]['10%']
    ]
}

df_adf_tabla = pd.DataFrame(datos_adf)

# Añadimos una columna de interpretación rápida para tu memoria
df_adf_tabla['Resultado'] = ""
df_adf_tabla.loc[1, 'Resultado'] = "Estacionaria" if resultado_adf[1] <= 0.05 else "No Estacionaria"
df_adf_tabla.to_csv('tabla_test_adf.csv', sep=';', index=False, decimal=',')


#FUNCIONES

def clasificador(fila):

    if fila['FESTIVO']== 1:
        return 'FESTIVO'
    elif fila['FIN_SEMANA']==1:
        return 'FIN_SEMANA'
    else:
        return 'LAVORAL'

#VEMOS LOS ESTADISTICOS

estadistica= df.describe()
print('*****RESUMEN ESTADÍSTICO******')
print(estadistica)
print('ceros en demanda prevista', (df['DEMANDA_PREVISTA']== 0).sum())

estadistica.to_csv('estadisticos.csv', sep=';', decimal=',', encoding= 'utf-8')


#LIMPIAMOS LOS 0 DE LA COLUMNA DEMANDA_PREVISTA
df_l = df[df['DEMANDA_PREVISTA']>0].copy()

print(f'Registros eliminados: {len(df) - len(df_l)}')

#VEMOS SI HAY NULOS
nulos= df_l.isnull().sum()
print('***NULOS***')
print((nulos))

nulos.to_csv('nulos.csv', sep=';', decimal=',', encoding= 'utf-8')
df_l.to_parquet('pvpc_brutos_sin_nulos.parquet')

#BOXPLOT PARA VER SI HAY OUTLIERS PRECIO

if not pd.api.types.is_datetime64_any_dtype(df.index):
    df.index = pd.to_datetime(df.index)

df_g= df_l.copy()
df_g['HORA']= df_g.index.hour

'''plt.figure(figsize= (15,8))
sns.boxplot(x='HORA', y= 'PRECIO_MWh', data= df_g, palette= 'Spectral')

plt.title('Distribucion precio €/MWh (outliers)')
plt.xlabel('Hora')
plt.ylabel('Precio (€/MWh)')
plt.grid(axis= 'y', linestyle ='--', alpha= 0.5)
plt.show()'''

#BOXPLOT PARA VER SI HAY OUTLIERS EN LA DEMANDA

df_g['FIN_SEMANA']= df_g.index.weekday.map(lambda x: 1 if x>=5 else 0)

festivos= holidays.Spain()
df_g['FESTIVO'] = df_g.index.map(lambda x: 1 if x in festivos else 0)

df_g['CATEGORIA_DIA']= df_g.apply(clasificador, axis=1)
'''plt.figure(figsize= (15,8))
sns.boxplot(x='HORA', y='DEMANDA_PREVISTA', hue= 'CATEGORIA_DIA', data= df_g, palette= 'Set2' )

plt.title('Oultiers demandad prevista')
plt.xlabel('Hora')
plt.ylabel('Demanada_prevista')
plt.legend( title= 'Tipo de dia')
plt.grid(axis= 'y', linestyle ='--', alpha= 0.4)
plt.show()'''


'''plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_g, x='DEMANDA_PREVISTA', y='PRECIO_MWh', alpha=0.3, palette='viridis')

plt.title('Relación Demanda vs Precio (Identificando significancia de outliers)')
plt.axhline(y=0.3, color='r', linestyle='--', label='Umbral de precio extremo')
plt.xlabel('Demanda Prevista ')
plt.ylabel('Precio PVPC ')
plt.legend()
plt.show()'''

#LIMPIAMOS LOS 0 DE LA COLUMNA DEMANDA_PREVISTA
df_l = df[df['DEMANDA_PREVISTA']>0].copy()

print(f'Registros eliminados: {len(df) - len(df_l)}')

#VEMOS LA RELACION ENTRE PRECIO, DEMANDA, FIN DE SEMANA Y FESTIVO
'''cols_interes = ['PRECIO_MWh', 'DEMANDA_PREVISTA', 'FIN_SEMANA', 'FESTIVO']
corr_matrix = df_g[cols_interes].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación: ¿Qué explica el precio?')
plt.show()'''


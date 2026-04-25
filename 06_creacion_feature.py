import pandas as pd
import numpy as np 
import os
import holidays
from datetime import time



#DESCARGA DE CSV
ruta = os.path.dirname(__file__)
ruta_archivo= os.path.join(ruta, 'pvpc_sin_outliers.parquet')

df_pen= pd.read_parquet(ruta_archivo)



print(f"Columnas detectadas: {len(df_pen.columns)}")
print("Nombres actuales:", df_pen.columns.tolist())
print(df_pen.info())



#PONER DECIMALES CON PUNTO
df_pen['PRECIO_MWh'] = (df_pen['PRECIO_MWh'].replace(',', '.', regex=True)).apply(pd.to_numeric, errors= 'coerce')

#FORMATO FECHA Y HORA 

df_pen['FECHA']= df_pen.index.date
df_pen['HORA']= df_pen.index.hour
df_pen['DIA_SEMANA']= df_pen.index.dayofweek

df_pen['DIA_SIN']= np.sin(2*np.pi*df_pen['DIA_SEMANA']/7)
df_pen['DIA_COS']= np.cos(2*np.pi* df_pen['DIA_SEMANA']/7)


df_pen['HORA_SIN']= np.sin(2*np.pi*df_pen['HORA']/24)
df_pen['HORA_COS']= np.cos(2*np.pi*df_pen['HORA']/24)
print(df_pen.head())


#FESTIVOS y FIN DE SEMANA EN MODO BINOMIAL
festivo= holidays.Spain()
df_pen['FESTIVO'] = df_pen['FECHA'].isin(list(holidays.Spain(years=range(2014, 2027)).keys())).astype(int)

df_pen['FIN_SEMANA']= df_pen['DIA_SEMANA'].isin([5,6]).astype(int)

#CAMBIO LEY 6/2021
fecha_2021= pd.to_datetime('2021-06-01').date()
df_pen['PRE_2021']= df_pen['FECHA'].apply(lambda x: 1 if x >= fecha_2021 else 0)

df_pen= df_pen[(df_pen['PRE_2021']==0) & (df_pen['DESCRIPCION'].str.contains('Término de facturación de energía activa del PVPC vehículo eléctrico'))| (df_pen['PRE_2021']==1)]


#PONER TODA LOS CONSUMO EN SU RANGO VALLE, PUNTA, LLANO
Rango=[ df_pen['HORA'].isin([10, 11, 12, 13, 18, 19, 20, 21]), #PUNTA
       df_pen['HORA'].isin([8, 9, 14, 15, 16, 17, 22, 23]),# LLANO
       df_pen['HORA']<8 ##VALLE
       ]
nombre=['PUNTA', 'LLANO', 'VALLE']
df_pen['RANGO']= np.select(Rango, nombre, default= 'Desconocido')
df_pen.loc[df_pen['DIA_SEMANA']>=5, 'RANGO']= 'VALLE'
df_pen.loc[df_pen['FESTIVO']==1, 'RANGO']='VALLE'
df_pen= pd.get_dummies(df_pen, columns= ['RANGO'], prefix= 'RANGO', dtype= int)


#CALCULAR FUTUROS
df_pen['FECHA'] = pd.to_datetime(df_pen['FECHA'])
PESO= [
    df_pen['FECHA']<pd.Timestamp('2024-01-01'),
    ((df_pen['FECHA']>=pd.Timestamp('2024-01-01')) | (df_pen['FECHA']< pd.Timestamp('2025-01-01'))),
    ((df_pen['FECHA']>=pd.Timestamp('2025-01-01'))| (df_pen['FECHA']< pd.Timestamp('2026-01-01'))),
    df_pen['FECHA']>pd.Timestamp('2026-01-01')
]
porcentaje= [0, 0.25, 0.4, 0.55]
df_pen['FUTUROS']= np.select(PESO, porcentaje, default= np.nan)


#PRECIO KWh
df_pen['PRECIO_KWh']= df_pen['PRECIO_MWh']/1000

#ORDENAR INDICES 
df_pen= df_pen.sort_values(by='FECHA_HORA', ascending= True)

#MEDIAS MOVILES y DESVIACIONES
df_pen['MEDIA_24H']= df_pen['PRECIO_KWh'].rolling(window=24).mean()
df_pen['MEDIA_SEMANA']= df_pen['PRECIO_KWh'].rolling(window=168).mean()
df_pen['MEDIA_TRIMESTRAL']= df_pen['PRECIO_KWh'].rolling(window= 2160).mean()
df_pen['VOLATILIDAD_6']= df_pen['PRECIO_KWh'].rolling(window= 6).std()


#VEMOS COMO ESTA EL MOMENTO RESPECTO AL DIA
dia_min= df_pen.groupby('FECHA')['PRECIO_KWh'].transform('min')
dia_max= df_pen.groupby('FECHA')['PRECIO_KWh'].transform('max')
df_pen['POSICION_RELATIVA']= np.where(
    (dia_max-dia_min) > 0, 
    (df_pen['PRECIO_KWh'] - dia_min) / (dia_max-dia_min), 
    0
)


#CREAMOS DOS NUEVAS COLUMNAS, LA DEL PRECIO DE LA LUZ EL DIA ANTERIOR Y OTRA QUE ES EL PRECIO DE LA LUZ EL MISMO DIA DE LA SEMANA PASADA

df_pen['PRECIO_AYER']= df_pen['PRECIO_KWh'].shift(24)
df_pen['PRECIO_SEMANA_ANTERIOR']= df_pen['PRECIO_KWh'].shift(168)
df_pen['DIFERENCIA_SEMANA'] = df_pen['PRECIO_AYER']- df_pen['PRECIO_SEMANA_ANTERIOR']


borrar= ['DESCRIPCION', 'ZONA', 'PRECIO_MWh', 'HORA', 'DIA_SEMANA']

df_final= df_pen.drop(columns= borrar)

df_final= df_final.dropna()

#ORDENAMOS LAS COLUMNAS PARA UNA VISIÓN MAS CLARA
orden= ['FECHA','PRECIO_KWh', 'DEMANDA_PREVISTA']

orden_2=[]
for col in df_final:
    if col not in orden:
        orden_2.append(col)


df_final= df_final[orden +orden_2]

print(df_final.head())


#TRANSFORMAR EL DATAFRAME EN PARQUET
df_final.to_parquet('pvpc_fin.parquet', engine='pyarrow')
df_final.to_csv('df_final.csv', sep=';', encoding= 'utf-8')

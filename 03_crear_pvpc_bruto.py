import pandas as pd
import numpy as np 
import os




#DESCARGA DE CSV
ruta = os.path.dirname(__file__)
ruta_archivo= os.path.join(ruta, 'pvpc.csv')
ruta_demanda= os.path.join(ruta, 'demanda.csv')

df = pd.read_csv(ruta_archivo, sep=';', decimal=',', encoding='utf-8', index_col=0)
print(f"Columnas detectadas: {len(df.columns)}")
print("Nombres actuales:", df.columns.tolist())
print(df.info())

df_demanda= pd.read_csv(ruta_demanda, sep=';', decimal='.', encoding='utf-8', index_col=0)
print(f'columnas detectadas demanda {len(df_demanda.columns)}')
print('Nombre actuales', df_demanda.columns.tolist())
print(df_demanda.info())

#MODIFICACION NOMBRE DF
df.columns=[ 'DESCRIPCION', 'ID_GEO', 'ZONA', 'PRECIO_MWh', 'FECHA_HORA']

#SELECCIÓN PENINSULA
df['ZONA']= df['ZONA'].replace(r'^\s*$', 'Península', regex= True).fillna('Península')
print(df.info())


df_pen= df[df['ZONA']== 'Península']
df_pen= df_pen.drop(columns=['ID_GEO'])


df_pen= df_pen[(df_pen['DESCRIPCION'].str.contains('Término de facturación de energía activa del PVPC vehículo eléctrico'))| (df_pen['DESCRIPCION'].str.contains('Término de facturación de energía activa del PVPC 2.0TD Península'))]

print(df_pen.info())

#JUNTAR df_pen Y df_demanda
df_pen['FECHA_HORA']= pd.to_datetime(df_pen['FECHA_HORA'], utc=True, format='ISO8601').dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
df_demanda['FECHA_HORA'] = pd.to_datetime(df_demanda['FECHA_HORA'], utc=True, format='ISO8601').dt.tz_convert('Europe/Madrid').dt.tz_localize(None)

df_pen.set_index('FECHA_HORA', inplace= True)
df_demanda.set_index('FECHA_HORA', inplace=True)
comparacion = df_pen.merge(df_demanda, left_index=True, right_index= True, how= 'outer', indicator= True)
print(f'valores distintos {comparacion[comparacion['_merge']!= 'both']}')

df_final = comparacion.drop(columns= ['_merge'])
df_final['PRECIO_MWh'] = (df_final['PRECIO_MWh'].replace(',', '.', regex=True)).apply(pd.to_numeric, errors= 'coerce')
print(df_final.info())

df_final.to_parquet('pvpc_bruto.parquet')
df_final.to_csv('pvpc_bruto.csv', sep=';', encoding='utf-8')

import pandas as pd
import numpy as np
import glob
import os

#BUSCAMOS Y JUNTAMOS LOS DIFERENTES CSV PARA CREAR UNO 
archivo = './'
archivo_csv = glob.glob(os.path.join('r20*.csv'))
print('archivos encontrados', len(archivo_csv))

lista_df= []
for archivo in archivo_csv:
    df = pd.read_csv(archivo, sep=';', decimal=',', encoding='utf-8')
    lista_df.append(df)

if lista_df:
    df_final = pd.concat(lista_df, ignore_index=True)
    
        # GUARDAMOS EL DATAFRAME EN UN CSV
    df_final.to_csv('demanda_bruta.csv', index=False, sep=';', decimal=',', encoding='utf-8')
    print("¡Unión completada! Archivo guardado como 'demanda.csv'")
else:
    print("No se encontraron archivos CSV para unir.")
df = pd.read_csv('demanda_bruta.csv', sep=';', decimal=',', encoding='utf-8')
print(df.info())

#ORDENAMOS EL CSV POR EL INDICE DE TIEMPO
df= df.sort_values(by='datetime', ascending= True)
df =df.reset_index(drop=True)

#TENEMOS UNA LISTA LARGA Y LA TRANSFORMAMOS EN DOS COLUMNAS
df['datetime']= pd.to_datetime(df['datetime'], utc= True)
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df= df.pivot_table(index= 'datetime', columns= 'name', values= 'value').reset_index()
print(df.info())

#CAMBIAMOS LOS NOMBRES Y ELIMINAMOS LO QUE NO NOS INTERESA, VEMOS SI HAY FECHAS REPETIDAS POR EL CAMBIO DE HORARIO
df.columns = ['FECHA_HORA', 'DEMANDA_REAL', 'DEMANDA_PREVISTA']

df = df.drop(columns= ['DEMANDA_REAL'])
print(f'valores duplicados {df['FECHA_HORA'].duplicated().sum()}')


#COMO HAY HUECOS LOS RELLENAMOS INTERPOLANDO
filas_nulas= df[df.isnull().any(axis=1)]
print(filas_nulas)

df['DEMANDA_PREVISTA']= df['DEMANDA_PREVISTA'].interpolate(method= 'linear')

print(f'valores con nulos {df[df.isnull().any(axis=1)]}')

#COMPROBAMOS Y CORREGIMOS ERRORES EN LOS DATOS
print(df.describe())
print('ceros en demanda prevista', (df['DEMANDA_PREVISTA']== 0).sum())





print(df.head())

df.to_csv('demanda.csv', sep=';', encoding='UTF-8')


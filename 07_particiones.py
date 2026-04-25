import os
import pandas as pd

def separar(data, target, eliminar):
    y = data[target]
    cols_to_drop = [target] + eliminar
    X = data.drop(columns=cols_to_drop)
    return X, y

ruta = os.path.dirname(__file__)
ruta_archivo= os.path.join(ruta, 'pvpc_seleccion.parquet')

df= pd.read_parquet(ruta_archivo)

#ASEGURAMOS QUE ESTA ORDENADO POR FECHAS

df['FECHA']= pd.to_datetime(df['FECHA'])
df = df.sort_values('FECHA').reset_index(drop=True)

#HACEMOS LAS PARTICIONES PARA ENTRENAR NUESTRO MODELOS

train = df[df['FECHA'] <='2024-12-31']
val = df[(df['FECHA'] >'2024-12-31') & (df['FECHA']<= '2025-06-30')]
test = df[df['FECHA']> '2025-06-30']

print(f"Train: hasta {train['FECHA'].max()} - ({len(train)} filas)")
print(f"Val:   de {val['FECHA'].min()} a {val['FECHA'].max()} - ({len(val)} filas)")
print(f"Test:  desde {test['FECHA'].min()} - ({len(test)} filas)")

#SEPARAMOS LO TARGET Y VALORES PARA LA PREDICCION
eliminar= ['FECHA']
X_train, y_train = separar(train, 'PRECIO_KWh', eliminar )
X_val, y_val     = separar(val, 'PRECIO_KWh', eliminar)
X_test, y_test   = separar(test, 'PRECIO_KWh', eliminar)

#GUARDAMOS LAS VARIABLES INDEPENDIENTES
X_train.to_parquet('X_train.parquet')
X_val.to_parquet('X_val.parquet')
X_test.to_parquet('X_test.parquet')

# GUARDAMOS LA VARIABLE DEPENDIENTE
y_train.to_frame().to_parquet('y_train.parquet')
y_val.to_frame().to_parquet('y_val.parquet')
y_test.to_frame().to_parquet('y_test.parquet')


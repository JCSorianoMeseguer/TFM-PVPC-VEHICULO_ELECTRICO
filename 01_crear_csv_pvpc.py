import pandas as pd
import glob
import os

#BUSCAMOS Y JUNTAMOS LOS DIFERENTES CSV PARA CREAR UNO 
act = './'
archivos_csv= glob.glob(os.path.join(act, '20*.csv'))
print('archivos encontrados', len(archivos_csv))

lista_df= []

for archivo in archivos_csv:

    df = pd.read_csv(archivo, sep=';', decimal=',', encoding='utf-8')
    lista_df.append(df)



    
if lista_df:
    df_final = pd.concat(lista_df, ignore_index=True)
    
    # GUARDAMOS EL DATAFRAME EN UN CSV
    df_final.to_csv('pvpc.csv', index=False, sep=';', decimal=',', encoding='utf-8')
    print("¡Unión completada! Archivo guardado como 'pvpc_completo_2014_2026.csv'")
else:
    print("No se encontraron archivos CSV para unir.")

#COMPROBAMOS EL CSV QUE HEMOS CREADO
df = pd.read_csv(pvpc, sep=';', decimal==',', encoding='utf-8')
print(pd.info())



print(df_final.tail(100))
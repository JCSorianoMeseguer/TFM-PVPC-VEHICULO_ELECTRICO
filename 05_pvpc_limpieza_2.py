import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

#ABRIMOS EL ARCHIVO
ruta= os.path.dirname(__file__)
ruta_archivo= os.path.join(ruta, 'pvpc_brutos_sin_nulos.parquet')
df = pd.read_parquet(ruta_archivo)

#IDENTIFICAMOS LOS VALORES QUE TIENEN UN VALOR MUY BAJO

valores_a= df[df['DEMANDA_PREVISTA']< 15000].copy()
valores_a['HORA']= valores_a.index.hour

print(valores_a.columns)

print(valores_a[[ 'DEMANDA_PREVISTA', 'PRECIO_MWh', ]].sort_values(by= 'DEMANDA_PREVISTA'))

tabla_outliers = valores_a[[ 'DEMANDA_PREVISTA', 'PRECIO_MWh']].reset_index()

#CREAMOS UNA TABLA PARA EXPLICARLA EN LA MEMORIA

'''fig, ax = plt.subplots(figsize=(8, 2)
ax.axis('off')
tabla = ax.table(cellText=tabla_outliers.values, 
                 colLabels=tabla_outliers.columns, 
                 loc='center', 
                 cellLoc='center')

tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.2, 1.2) 

plt.title("Tabla: Identificación de registros anómalos en demanda", y=1.1)
plt.savefig("tabla_outliers_demanda.png", bbox_inches='tight', dpi=300)
plt.show()'''

lista_error= [pd.Timestamp('2015-04-29 07:00:00'),
              pd.Timestamp('2015-04-29 08:00:00'),
              pd.Timestamp('2025-04-28 12:00:00')]

for idx in lista_error:
    diana = idx - pd.Timedelta(days= 7)
    if diana in df.index:
        n_valor= df.loc[diana, 'DEMANDA_PREVISTA']
        df.loc[idx, 'DEMANDA_PREVISTA']= n_valor
        print(f'Corregido {idx}, con valor de {diana}: {n_valor}')

#CALCULAMOS LOS CUARTILES PARA EL PRECIO PARA VER SI HAY OUTLIERS PARA VER QUE SE HACE CON ELLOS

Q1= df['PRECIO_MWh'].quantile(0.25)
Q3= df['PRECIO_MWh'].quantile(0.75)

IQR= Q3-Q1

umbral_sup= Q3+ IQR * 1.5
umbral_inf= Q1- IQR*1.5

o_precio= (df['PRECIO_MWh']> umbral_sup)|(df['PRECIO_MWh']< umbral_inf)
total_o = o_precio.sum()
menores_cero = df[df['PRECIO_MWh']<0]
print(f"El umbral estadístico de outlier es: {umbral_sup:.4f} €/MWh")
print(f'El umbral estadístico inferior es: {umbral_inf: .4f} €/MWh')
print(f"Número de registros considerados outliers: {total_o}")
print(f'Estor outliers son el {((total_o/len(df))*100):.3}%')
print(f'Valores inferiores a 0 en el precio {len(menores_cero)}')

#VAMOS A VER LOS OUTLIERS DE PRECIO SI SON TENDENCIA 

outliers= df[df['PRECIO_MWh']> umbral_sup].copy()

outliers['año']= outliers.index.year
outliers['mes']= outliers.index.month

print(f'resumen por años: {outliers.groupby('año').size()}')

df_unico = df[~df.index.duplicated(keep='first')].copy()


df_unico['año'] = df_unico.index.year


'''plt.figure(figsize=(12, 6))
sns.boxplot(data=df_unico, x='año', y='PRECIO_MWh', palette="viridis")
plt.axhline(y=umbral_sup, color='red', linestyle='--', label=f'Umbral Outlier ({umbral_sup:.2f})')
plt.title('Distribución Anual de Precios PVPC', fontsize=14)
plt.ylabel('€ / MWh')
plt.xlabel('Año')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()'''

print(df.loc['2015-04-29 07:00:00'])

df.to_parquet('pvpc_sin_outliers.parquet')
df.to_csv('pvpc_sin_outliers.csv', sep=';', encoding='utf-8')



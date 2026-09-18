# ==============================================================================
# MACHINE LEARNING
# Práctica Nro 1: Apertura de archivos  12/09/2026
# se debe instalar openpyxl para poder leer archivos xlsx en la consola de Python
# pip install openpyxl
# ==============================================================================
# Load CSV Using Pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Ejecuta 'cls' si estás en Windows, o 'clear' si estás en Mac o Linux
os.system('cls' if os.name == 'nt' else 'clear')

filename = "data/processed/hotel_bookings.csv"

data = pd.read_csv(filename) 
print("------------------------------------------------------------")
print("Forma del DataFrame:")
print(data.shape)
print("------------------------------------------------------------")
print("Información del DataFrame:")
print(data.info())
print("------------------------------------------------------------")
print("Tipos de datos:")
print(data.dtypes)
print("------------------------------------------------------------")
print("Valores nulos por columna:")
nulos_por_columna = data.isnull().sum()
print(nulos_por_columna[nulos_por_columna>0])  # Muestra solo las columnas con valores nulos
print("------------------------------------------------------------")
print("Total de valores nulos:")
print(data.isnull().sum().sum())
print("------------------------------------------------------------")
print("Reemplazando los valores nulos en la columna 'children' con 0...")
data['children'] = data['children'].fillna(0)
print("Valores nulos por columna después de reemplazar los valores nulos en la columna 'children' con 0:")
nulos_por_columna = data.isnull().sum()
print(nulos_por_columna[nulos_por_columna>0])  # Muestra solo las columnas con valores nulos
print("------------------------------------------------------------")
print("Total de valores nulos despues de reemplazar:")
print(data.isnull().sum().sum())

print("------------------------------------------------------------")
print("Reemplazando los valores nulos en la columna 'agent' con 0...")
data['agent'] = data['agent'].fillna(0)
print("Valores nulos por columna después de reemplazar los valores nulos en la columna 'agent' con 0:")
nulos_por_columna = data.isnull().sum()
print(nulos_por_columna[nulos_por_columna>0])  # Muestra solo las columnas con valores nulos
print("------------------------------------------------------------")
print("Total de valores nulos despues de reemplazar agent:")
print(data.isnull().sum().sum())


print("------------------------------------------------------------")
print("Reemplazando los valores nulos en la columna 'country' con OTRO...")
data['country'] = data['country'].fillna("OTRO")
print("Valores nulos por columna después de reemplazar los valores nulos en la columna 'country' con OTRO:")
nulos_por_columna = data.isnull().sum()
print(nulos_por_columna[nulos_por_columna>0])  # Muestra solo las columnas con valores nulos
print("------------------------------------------------------------")
print("Total de valores nulos despues de reemplazar country:")
print(data.isnull().sum().sum())
print("------------------------------------------------------------")
# Selección y extracción de características prioritarias 
columns_to_keep = ['hotel', 'lead_time', 'customer_type', 'adr']
df_subset = data[columns_to_keep].dropna().copy()
print("------------------------------------------------------------")
print("Subconjunto de datos con características prioritarias:")
print(df_subset.head(10))  # Muestra las primeras 10 filas del subconjunto de datos

print("------------------------------------------------------------")
print("Gráfico de Cajas")
print("------------------------------------------------------------")
#Configurar el tamaño general de la figura
plt.figure(figsize=(12, 5))
# Primer subgráfico: Boxplot para 'lead_time'
plt.subplot(1, 2, 1) # (1 fila, 2 columnas, posición 1)
sns.boxplot(y=df_subset['lead_time'], color='skyblue')
plt.title('Gráfico de Cajas: Lead Time (Antelación)')
plt.ylabel('Días')

#Segundo subgráfico: Boxplot para 'adr'
plt.subplot(1, 2, 2) # (1 fila, 2 columnas, posición 2)
sns.boxplot(y=df_subset['adr'], color='lightgreen')
plt.title('Gráfico de Cajas: ADR (Tarifa Promedio Diaria)')
plt.ylabel('Euros')

#  Ajustar los espacios para que no se superpongan los textos
plt.tight_layout()

# Guardar la imagen (opcional) y mostrarla en pantalla
plt.savefig('boxplot_subset.png')
plt.show()

print("------------------------------------------------------------")
print("Transformación de variables categóricas nominales (One-Hot Encoding)")
print("------------------------------------------------------------")

# Transformación de variables categóricas nominales (One-Hot Encoding)
df_transformed = pd.get_dummies(df_subset, columns=['hotel', 'customer_type'], drop_first=True)

print("------------------------------------------------------------")
print("Variables transformadas hotel y customer_type:")
print(df_transformed.head(20).drop(["lead_time","adr"], axis=1))  # Muestra las primeras 20 filas del DataFrame transformado
print("------------------------------------------------------------")

# Instanciación de métodos de escalado
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Normalización y Estandarización matemática
df_transformed['lead_time_minmax'] = min_max_scaler.fit_transform(df_transformed[['lead_time']])
df_transformed['adr_zscore'] = standard_scaler.fit_transform(df_transformed[['adr']])

print(df_transformed[['lead_time_minmax','lead_time', 'adr_zscore','adr']].describe())
print("------------------------------------------------------------")
print("Terminación de las variables transformadas")
print("------------------------------------------------------------")
# Exportación de visualizaciones gráficas
plt.figure(figsize=(18, 5))

# Histograma
plt.subplot(1, 3, 1)
sns.histplot(df_transformed['lead_time_minmax'], bins=30, kde=True, color='skyblue')
plt.title("Histograma: Lead Time (Min-Max)")

# Diagrama de Caja
plt.subplot(1, 3, 2)
sns.boxplot(y=df_transformed['adr_zscore'], color='lightgreen')
plt.title("Gráfico de Cajas: ADR (Z-Score)")


# Dispersión
plt.subplot(1, 3, 3)
sns.scatterplot(x='lead_time_minmax', y='adr_zscore', data=df_transformed, alpha=0.3, color='coral')
plt.title("Dispersión: Lead Time vs ADR")

plt.tight_layout()
plt.savefig("visualizaciones_transformacion.png")
plt.show()
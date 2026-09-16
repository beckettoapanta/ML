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
print("------------------------------------------------------------")
print("Cantidad de textos vacíos por columna:")
textos_vacios = (data == "").sum()
print(textos_vacios[textos_vacios > 0])
print("------------------------------------------------------------")

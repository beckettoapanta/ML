# ==============================================================================
# MACHINE LEARNING
# Práctica Nro 1: Apertura de archivos  12/09/2026
# ==============================================================================
# Load XLS   Using Pandas
import pandas as pd
import os
import openpyxl 
# Ejecuta 'cls' si estás en Windows, o 'clear' si estás en Mac o Linux
os.system('cls' if os.name == 'nt' else 'clear')
filename = "data/processed/1000-Registros-de-ventas.xlsx"
data = pd.read_excel(filename, sheet_name="Ventas")   
print("------------------------------------------------------------")
print("Primeras 10 filas del DataFrame:\n")
print(data.head(10))            
print("------------------------------------------------------------") 
print("Muestra la forma del DataFrame (filas, columnas):")
print(data.shape)         
print("------------------------------------------------------------")
print("Estadísticas descriptivas del DataFrame:")
print(data.describe())        
print("------------------------------------------------------------")
print("Información del DataFrame:")
print(data.info())           # Tipos de datos, valores nulos, etc.)      
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("------------------------------------------------------------")
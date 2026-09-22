# ==============================================================================
# MACHINE LEARNING
# Evidencia AA3-EV01:  RReporte de visualización de datos. 21/09/2026
# 
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
# ==============================================================================

# Ejecuta 'cls' si estás en Windows, o 'clear' si estás en Mac o Linux
os.system('cls' if os.name == 'nt' else 'clear')
# ==============================================================================
# Cargar el dataset limpio y estandarizado

filename = "data/processed/supermarket_sales.csv"
df = pd.read_csv(filename) 
print(df.info())
print(df.describe())
print(df.head(20))  # Mostrar las primeras 20 filas para verificar la carga del dataset
print(df.columns)
# ==============================================================================
# Exploración de datos y visualización

plt.figure(figsize=(10, 5))
sns.histplot(df['revenue'], bins=30, kde=True)
plt.title('Distribución del Total de Ventas')
plt.xlabel('Total')
plt.ylabel('Frecuencia')
plt.savefig('dist_total.png')
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
sns.boxplot(x='branch', y='revenue', data=df)
plt.title('Total de Ventas por Sucursal (Detección de Atípicos)')
plt.savefig('box_branch.png')
plt.show()
plt.close
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.savefig('corr.png')
plt.show()
plt.close()
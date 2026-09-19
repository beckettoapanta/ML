# ==============================================================================
# MACHINE LEARNING
# Evidencia AA2-EV02:  Reporte de transformación de datos. 18/09/2026
# Código limpieza, estandarización numérica y codificación categórica
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

filename = "data/processed/hotel_bookingsDef.csv"
df_final = pd.read_csv(filename) 
print(df_final.head(20))  # Mostrar las primeras 20 filas para verificar la carga del dataset

# Aplicar Análisis de Componentes Principales
pca = PCA()
pca.fit(df_final)

# Calcular la varianza explicada acumulada
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

# Proyectar en 2 componentes y extraer cargas (Loadings)
df_pca_2d = pd.DataFrame(pca.transform(df_final)[:, :2], columns=['PC1', 'PC2'])
loadings = pd.DataFrame(pca.components_.T[:, :5], index=df_final.columns, columns=[f'PC{i+1}' for i in range(5)])

# Configurar el panel general de visualización
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico A: Varianza Explicada Acumulada (Scree Plot)
axes[0].plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o', linestyle='--')
axes[0].axhline(y=0.75, color='red', linestyle='-', label='Umbral Recomendado (75%)')
axes[0].set_title('Gráfica de Varianza Explicada')
axes[0].set_xlabel('Número de Componentes Principales')
axes[0].set_ylabel('Varianza Acumulada')
axes[0].legend()

# Gráfico B: Mapa de Calor (Top 10 variables con mayor peso en PC1 y PC2)
top_vars = loadings['PC1'].abs().sort_values(ascending=False).head(10).index
sns.heatmap(loadings.loc[top_vars, ['PC1', 'PC2']], annot=True, cmap='coolwarm', ax=axes[1], fmt='.2f')
axes[1].set_title('Mapa de Calor: Cargas y Pesos (Loadings)')

# Gráfico C: Diagrama de Dispersión en Espacio Reducido
muestra = df_pca_2d.sample(5000, random_state=42) # Muestreo para evitar saturación visual
sns.scatterplot(x='PC1', y='PC2', data=muestra, alpha=0.5, color='#2980B9', ax=axes[2])
axes[2].set_title('Diagrama de Dispersión (PC1 vs PC2)')

plt.tight_layout()
plt.show()
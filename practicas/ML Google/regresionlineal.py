#=============================================================================
# Regresión Lineal Múltiple con Gráficos Separados e Integrados
# Autor: Becket Toapanta date: 2026-09-24
#=============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
#=============================================================================
# 1. Preparación previa (Simulación de datos y modelo)
#=============================================================================
# 1. Creamos un DataFrame simulado con tus 5 atributos y la variable objetivo (mpg)
np.random.seed(42)
n = 100
data = {
    'Pounds': np.random.uniform(2000, 5000, n),
    'Displacement': np.random.uniform(70, 450, n),
    'Acceleration': np.random.uniform(8, 25, n),
    'Cylinders': np.random.choice([4, 6, 8], n),
    'Horsepower': np.random.uniform(50, 250, n),
    'MPG': np.random.uniform(10, 45, n) # Variable real a predecir
}
df = pd.DataFrame(data)

# 2. Definimos variables e introducimos el modelo
X = df[['Pounds', 'Displacement', 'Acceleration', 'Cylinders', 'Horsepower']]
y = df['MPG']

modelo = LinearRegression()
modelo.fit(X, y)
df['MPG_Predicho'] = modelo.predict(X) # Aquí guardamos y'
#=============================================================================
# Código para Gráficos Separados (Análisis Individual)
#=============================================================================

# Lista con los nombres exactos de tus columnas de atributos
atributos = ['Pounds', 'Displacement', 'Acceleration', 'Cylinders', 'Horsepower']

# Creamos una figura con espacio para 5 gráficos (2 filas, 3 columnas)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten() # Aplana la matriz para iterar fácilmente

for i, col in enumerate(atributos):
    sns.regplot(data=df, x=col, y='MPG', ax=axes[i], color='blue', scatter_kws={'alpha':0.5})
    axes[i].set_title(f'MPG vs {col}')
    axes[i].set_ylabel('Millas por Galón (MPG)')

# Eliminamos el sexto gráfico que queda vacío en la cuadrícula
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

#=============================================================================
# 3. Código para Gráfico Integrado (Rendimiento del Modelo Completo)
#=============================================================================

plt.figure(figsize=(7, 6))

# Gráfico de dispersión de Reales vs Predichos
sns.scatterplot(data=df, x='MPG', y='MPG_Predicho', alpha=0.7, color='green')

# Línea de referencia perfecta (Identidad de 45 grados)
max_val = max(df['MPG'].max(), df['MPG_Predicho'].max())
min_val = min(df['MPG'].min(), df['MPG_Predicho'].min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Predicción Perfecta')

plt.title('Gráfico Integrado: Valores Reales vs. Predichos ($y\'$)')
plt.xlabel('Millas por Galón Reales')
plt.ylabel('Millas por Galón Predichas ($y\'$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

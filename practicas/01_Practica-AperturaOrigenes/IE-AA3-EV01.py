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

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# Checkear valores faltantes en el dataset
missing = df.isnull().sum().sum()
print(f"Total valores faltantes: {missing}")
# ==============================================================================
# Exploración de datos y visualización

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x='revenue',
    bins=30,
    kde=True,
    color="#1fb482",       # Azul corporativo limpio
    edgecolor="white",     # Separación elegante entre barras
    linewidth=1.2,
    alpha=0.75,            # Transparencia 
    line_kws={"linewidth": 2.5, "color": "#d62728"} # KDE en un tono contraste sutil
)
# ==============================================================================
# HISTOGRAMA DE VENTAS: Personalización de ejes y etiquetas
# ==============================================================================
# Personalizar títulos con jerarquía visual (Negrita y Tamaños)
plt.title('Distribución del Total de Ventas', fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
plt.xlabel('Total de Ventas ($)', fontsize=12, labelpad=10, color='#34495e')
plt.ylabel('Frecuencia de Transacciones', fontsize=12, labelpad=10, color='#34495e')
# Limpiar el exceso de líneas (Remover bordes superior y derecho)
sns.despine(left=True, bottom=True)
# Ajustar detalles de los ejes para que no se corten
plt.tight_layout()

# Guardar y mostrar
plt.savefig('dist_total_profesional.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()
# ==============================================================================
# DIAGRAMA DE CAJAS (BOXPLOT): Personalización de estilo y colores
# ==============================================================================
# Configurar un tema de fondo limpio y moderno
sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "grid.alpha": 0.5})

# Definir dimensiones de la figura con alta resolución
plt.figure(figsize=(10, 6), dpi=100)

# Crear el Boxplot estilizado
sns.boxplot(
    x='branch', 
    y='revenue', 
    data=df,
    palette="Blues_d",          # Degradado de azul elegante y profesional
    linewidth=1.5,              # Grosor de las líneas de las cajas
    fliersize=7,                # Tamaño de los puntos atípicos
    flierprops={
        "markerfacecolor": "#e74c3c",  # Color rojo llamativo para resaltar los atípicos
        "markeredgecolor": "white",    # Borde blanco para mayor contraste
        "marker": "o"
    },
    width=0.6                   # Espaciado óptimo entre cajas
)

# Personalizar textos y títulos (Jerarquía visual)
plt.title('Total de Ventas por Sucursal\n(Detección de Valores Atípicos)', fontsize=15, fontweight='bold', pad=15, color='#2c3e50')
plt.xlabel('Sucursal (Branch)', fontsize=12, labelpad=10, color='#34495e')
plt.ylabel('Ingresos (Revenue, $)', fontsize=12, labelpad=10, color='#34495e')

# Formatear las etiquetas de los ejes para mejor lectura
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Eliminar bordes para un diseño minimalista "editorial"
sns.despine(left=True, bottom=True)

# Ajustar elementos para evitar recortes
plt.tight_layout()

# Guardar en alta calidad y mostrar
plt.savefig('box_branch_profesional.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# ==============================================================================
# DIAGRAMA DE CORRELACIÓN: Personalización de estilo y colores
# ==============================================================================

# Seleccionar numéricas y filtrar las que tengan varianza cero (evita filas/columnas vacías)
numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.loc[:, numeric_df.var() > 0] 

# Calcular la matriz de correlación
corr_matrix = numeric_df.corr()

# Crear una máscara para ocultar la mitad superior (triángulo superior)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Configurar la figura con excelente resolución
plt.figure(figsize=(10, 8), dpi=100)

# Dibujar el mapa de calor estilizado
sns.heatmap(
    corr_matrix,
    mask=mask,                  # Aplica la máscara triangular
    annot=True,                 # Muestra los números
    fmt=".2f",                  # Dos decimales
    cmap="vlag",                # Paleta divergente elegante (azul a rojo, centrada en 0)
    vmax=1.0, vmin=-1.0,        # Límites perfectos para correlación
    center=0,
    square=True,                # Forzar a que las celdas sean cuadrados perfectos
    linewidths=1.5,             # Líneas finas de separación entre celdas
    cbar_kws={"shrink": 0.7, "label": "Coeficiente de Correlación"}, # Personalizar barra lateral
    annot_kws={"size": 10, "weight": "bold"} # Números claros y legibles
)

# Personalizar títulos y etiquetas cuidando los márgenes
plt.title('Matriz de Correlación de Variables', fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
plt.xticks(rotation=45, ha='right', fontsize=11, color='#34495e')
plt.yticks(rotation=0, fontsize=11, color='#34495e')

# Ajustar automáticamente los márgenes para que NADA se corte
plt.tight_layout()

# Guardar y mostrar
plt.savefig('matriz_correlacion_profesional.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()




# ==============================================================================
# PARTICIONAMIENTO DE DATOS: División en Entrenamiento, Validación y Prueba
# ==============================================================================
# Porcentajes de partición de la base  (70% train, 15% val, 15% test)

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

# Mostramos la partición
print(f"Train: {len(train_df)} ({len(train_df)/len(df):.0%})")
print(f"Validation: {len(val_df)} ({len(val_df)/len(df):.0%})")
print(f"Test: {len(test_df)} ({len(test_df)/len(df):.0%})")

# ==============================================================================
# Visualicemos la distribución de las particiones en función de una variable 
#  clave (por ejemplo, «revenue»)
# ==============================================================================
# Configurar un entorno estético minimalista
sns.set_theme(style="white", rc={"grid.linestyle": "--", "grid.alpha": 0.3})

# Inicializar la figura con alta resolución
plt.figure(figsize=(11, 6), dpi=100)

# Definir una paleta de colores corporativa y clara
colors = {
    'train': '#1f77b4',   # Azul formal
    'val': '#e67e22',     # Naranja cálido
    'test': '#2ecc71'     # Verde equilibrado
}

# Graficar cada densidad de forma individual con parámetros optimizados
sns.kdeplot(
    data=train_df['revenue'], 
    label='Entrenamiento (70%)', 
    fill=True, 
    alpha=0.18,           # Transparencia equilibrada para ver las capas de fondo
    color=colors['train'], 
    linewidth=2,          # Línea superior más definida
    antialiased=True
)

sns.kdeplot(
    data=val_df['revenue'], 
    label='Validación (15%)', 
    fill=True, 
    alpha=0.18, 
    color=colors['val'], 
    linewidth=2,
    antialiased=True
)

sns.kdeplot(
    data=test_df['revenue'], 
    label='Prueba (15%)', 
    fill=True, 
    alpha=0.18, 
    color=colors['test'], 
    linewidth=2,
    antialiased=True
)

# Agregar una cuadrícula sutil en el eje X para facilitar la lectura de valores
plt.grid(axis='x', color='gray', linestyle=':', alpha=0.5)

# 6. Jerarquía tipográfica y etiquetas limpias
plt.title('Distribución de la Variable Revenue en las Particiones', fontsize=15, fontweight='bold', pad=20, color='#2c3e50')
plt.xlabel('Revenue ($)', fontsize=12, labelpad=12, color='#34495e')
plt.ylabel('Densidad', fontsize=12, labelpad=12, color='#34495e')

# Formatear la leyenda para que se vea ordenada y moderna
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10, loc='upper right')

# Eliminar bordes para un diseño "despejado"
sns.despine(top=True, right=True, left=False, bottom=False)

# Ajustar márgenes para evitar cortes accidentales
plt.tight_layout()

# Guardar en alta calidad para el informe final
plt.savefig('partition_dist_profesional.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()



# ==============================================================================
# PIPELINE DE PREPROCESAMIENTO: Preparación de datos para modelado
# ==============================================================================


#  Limpieza estructural (Eliminamos ID, fechas crudas y variables con colinealidad perfecta)
# Nota: 'revenue' también se excluye aquí porque suele ser la variable a predecir (Target)
columnas_a_eliminar = ['invoice_id', 'date', 'time', 'cogs', 'gross_income', '5pct_markup', 'revenue']
X = df.drop(columns=columnas_a_eliminar)

# Clasificación de las columnas restantes
columnas_numericas = ['unit_cost', 'quantity', 'rating']
columnas_categoricas = [
    'branch', 'city', 'customer_type', 
    'gender_customer', 'product_line', 'payment_method'
]

# Definición de las tuberías (Pipelines) individuales
procesador_numerico = Pipeline(steps=[
    ('escalado', StandardScaler()) # Convierte los números a escala Z-Score
])

procesador_categorico = Pipeline(steps=[
    # Convierte el texto a 1s y 0s. drop='first' evita la multicolinealidad
    ('codificacion', OneHotEncoder(drop='first', sparse_output=False)) 
])

# Ensamblaje del preprocesador maestro
preprocesador = ColumnTransformer(transformers=[
    ('num', procesador_numerico, columnas_numericas),
    ('cat', procesador_categorico, columnas_categoricas)
])

# Ejecución: Ajustar (aprender medias/categorías) y Transformar los datos
X_procesado = preprocesador.fit_transform(X)

# Reconstruir a un DataFrame de Pandas para visualizar el resultado
nombres_columnas = preprocesador.get_feature_names_out()
df_procesado = pd.DataFrame(X_procesado, columns=nombres_columnas)

# Mostrar resultados
print(f"Dimensiones originales para el modelo: {X.shape}")
print(f"Dimensiones finales tras el Pipeline: {df_procesado.shape}")
print("\nVista previa de los datos listos para IA:")
print(df_procesado.head(3).round(3))

#------------------------------------------------------------
#          EXPORTACION DE DATOS LIMPIOS Y TRANSFORMADOS
#------------------------------------------------------------

print("------------------------------------------------------------")
archivo_salida = 'data/processed/supermarket_salesdef.csv'
df_procesado.to_csv(archivo_salida, index=False)

print(f"Dimensiones finales tras limpieza y transformación: {df_procesado.shape}")
print(f"Archivo '{archivo_salida}' generado exitosamente y listo para elModelado.")

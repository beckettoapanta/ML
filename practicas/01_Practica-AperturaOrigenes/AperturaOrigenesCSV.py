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
print("Primeras 5 filas del DataFrame:")
print(data.head(5))
print("------------------------------------------------------------")
print("Forma del DataFrame:")
print(data.shape)
print("------------------------------------------------------------")
print("Columnas del DataFrame:")
print(data.columns)
print("------------------------------------------------------------")
print("Descripción del DataFrame:")
print(data.describe())
print("------------------------------------------------------------")
print("Información del DataFrame:")
print(data.info())
print("------------------------------------------------------------")
print("Tipos de datos:")
print(data.dtypes)
print("------------------------------------------------------------")
print("Valores nulos por columna:")
print(data.isnull().sum())
print("------------------------------------------------------------")
print("Total de valores nulos:")
print(data.isnull().sum().sum())
print("------------------------------------------------------------")
print("Número de valores únicos por columna:")
print(data.nunique())   
print("------------------------------------------------------------")
print("opcion 1:Conteo de valores en la columna 'MARCA':")
print(data['MARCA'].value_counts())
print("------------------------------------------------------------")
print("Opción 2:Conteo de valores en la columna 'MARCA':")
print(data.groupby('MARCA').size())
print("------------------------------------------------------------")
print(" Pairwise Pearson correlation of columns, excluding NA/null values:")
correlations = data.corr( method='pearson', numeric_only=True)  # Calcula la correlación de Pearson entre las columnas numéricas
print(correlations)
print("------------------------------------------------------------")  
print("Correlacion por pares de variables:") 
# Desapilar (unstack) para convertir la matriz en una serie de pares de variables
corr_pairs = correlations.unstack()
# Filtrar los valores según tu condición (mayor a 0.75 o menor a -0.75)
# Excluimos el 1.0 perfecto para no mostrar la correlación de una variable consigo misma
strong_corrs = corr_pairs[(abs(corr_pairs) > 0.75) & (corr_pairs != 1.0)]
print("Pares de variables con correlación fuerte (> 0.75 o < -0.75):")
print(strong_corrs)
print("------------------------------------------------------------")  
print("Sesgo para cada atributo:") 
print(data.skew(numeric_only=True))  # Calcula el sesgo para cada columna numérica


print("------------------------------------------------------------")  
print("VISUALIZACIONES:")
print("------------------------------------------------------------") 

  
print("------------------------------------------------------------")  
print("Histogramas de cada atributo:") 
#histograms = data.hist(figsize=(12, 10), bins=20, grid = False ) 
# Un ejemplo combinando varios parámetros de personalización
histograms = data.hist(
    figsize=(12, 10), 
    bins=10, 
    color="#21c45f",     # Color azul personalizado
    edgecolor='white',   # Bordes blancos para separar las barras
    grid=False,          # Quita las líneas de fondo
    xrot=45,             # Rota los números del eje X para que no se encimen
    alpha=0.8            # Ligera transparencia estética
)

# figsize=(12, 10) establece el tamaño de la figura, y bins=20 define el número de contenedores para los histogramas    
plt.suptitle('Distribución de mis Atributos', fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajusta el diseño para que no se superpongan los subplots
# rect=[0, 0, 1, 0.95] asegura que el título principal no se superponga con los subplots
plt.show()  # Muestra los histogramas   
print("------------------------------------------------------------")  
print("Visualizar los histogramas a linea de cada atributo: Densidad de cada atributo")
print("Densidad de cada atributo:") 
data.plot(kind='density', subplots=True, layout=(len(data.columns)//3+1, 3), figsize=(12, 10))
plt.suptitle('Densidad de mis Atributos', fontsize=16, fontweight='bold', y=0.95)
plt.show()  # Muestra los histogramas 

print("------------------------------------------------------------")  
print("Visualizar Boxplot de cada atributo:")
#data.boxplot(figsize=(12, 10), grid=False, patch_artist=True, boxprops=dict(facecolor="#21c45f", color='black'), medianprops=dict(color='red'))
#plt.suptitle('Boxplot de mis Atributos', fontsize=16, fontweight='bold', y=0.95)
plots = data.plot(kind='box', subplots=True, layout=(len(data.columns)//3+1, 3), figsize=(12, 10), patch_artist=True, boxprops=dict(facecolor="#21c45f", color='black'), medianprops=dict(color='red'))
plt.suptitle('Boxplot de mis Atributos', fontsize=16, fontweight='bold', y=0.95)    
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()  # Muestra los boxplots 


print("------------------------------------------------------------")  
print("Visualizar Matriz de correlación con un mapa de calor:")

sns.heatmap(
    correlations,
    annot=True,  # Muestra los números de la correlación dentro de cada cuadro
    cmap="coolwarm",  # Escala de color (azul = negativa, rojo = positiva)
    fmt=".2f",  # Limita los números a 2 decimales
    linewidths=0.5,  # Añade una línea de separación entre los cuadros
    vmin=-1,  # Fija el valor mínimo de la escala en -1
    vmax=1,  # Fija el valor máximo de la escala en 1
)
#  Añadir título y ajustar espacios para que no se corte
plt.title("Matriz de Correlación", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
# Mostrar la gráfica
plt.show()
print("------------------------------------------------------------")  
print("Genera una cuadrícula de dispersión para todas las variables numéricas")

# 1. Definimos tu variable objetivo (el eje Y principal)
# Cambia 'STATUS' por la columna que más te interese analizar si prefieres otra
variable_y = "EDAD"

# 2. Lista automática de las columnas numéricas que se van a graficar en el eje X
columnas_x = [
    col
    for col in data.select_dtypes(include=["number"]).columns
    if col != variable_y
]

# 3. Configuramos una cuadrícula limpia (ej. 3 columnas de gráficos por fila)
n_cols = 3
n_filas = (len(columnas_x) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_filas, n_cols, figsize=(16, 4 * n_filas))
axes = axes.flatten()  # Aplana la matriz de gráficos para recorrerla fácilmente

# 4. Dibujamos un gráfico de dispersión claro para cada atributo
for i, col in enumerate(columnas_x):
    sns.scatterplot(
        data=data,
        x=col,
        y=variable_y,
        ax=axes[i],
        alpha=0.6,
        color="#1f77b4",
        edgecolor="w",
    )
    axes[i].set_title(f"Relación: {col} vs {variable_y}", fontsize=11, weight="bold")
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel(variable_y, fontsize=10)
    axes[i].grid(True, linestyle="--", alpha=0.5)

# 5. Ocultamos los recuadros vacíos si el total de gráficos no es múltiplo de 3
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajustamos espacios para que ningún texto ni etiqueta se superponga
plt.suptitle('Matriz de Dispersión', fontsize=16, fontweight='bold', y=0.95)   
 
plt.tight_layout(rect=[0, 0, 1, 0.95])
# hspace=0.5 aumenta la distancia vertical entre las filas de gráficos
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()

print("------------------------------------------------------------")  
# ==============================================================================
# MACHINE LEARNING
# Práctica Nro 1: Apertura de archivos  12/09/2026
# ==============================================================================
library("dplyr")

# Limpiar la consola en R (Equivalente a os.system en Python)
cat("\014") 

# Definir la ruta del archivo
getwd()
setwd("C:/GitHub-Local/ML/data/processed")
filename <- "long_format.csv"

# Leer el archivo CSV

data <- read.csv("wide_format.csv", sep = ",", dec = ".", header = TRUE)

cat("------------------------------------------------------------\n")
cat("Primeras 5 filas del DataFrame:\n")
print(head(data, 5))

cat("------------------------------------------------------------\n")
cat("Forma del DataFrame:\n")
print(dim(data)) # Devuelve un vector con c(filas, columnas)

cat("------------------------------------------------------------\n")
cat("Columnas del DataFrame:\n")
print(colnames(data))

cat("------------------------------------------------------------\n")
cat("Descripción del DataFrame:\n")
print(summary(data)) # Resumen estadístico básico

cat("------------------------------------------------------------\n")
cat("Información del DataFrame:\n")
print(str(data)) # Muestra la estructura interna, tipos y filas (Equivalente a info y dtypes)

cat("------------------------------------------------------------\n")
cat("Tipos de datos:\n")
print(sapply(data, class)) # Muestra la clase/tipo de cada columna

cat("------------------------------------------------------------\n")
cat("Valores nulos por columna:\n")
print(colSums(is.na(data)))

cat("------------------------------------------------------------\n")
cat("Cantidad NA que hay en la data:\n")
print(sum(is.na(data)))   

cat("------------------------------------------------------------\n")
cat("Número de valores únicos por columna:\n")
print(sapply(data, function(x) length(unique(x))))

cat("------------------------------------------------------------\n")
cat("opcion 1: Conteo de valores en la columna 'MARCA':\n")
print(table(data$MARCA))
cat("------------------------------------------------------------\n")

cat("Opción 2:Conteo de valores en la columna 'MARCA':\n")
# Opción A (Usando dplyr):
print(data %>% count(MARCA))

cat("------------------------------------------------------------\n")
cat(" Pairwise Pearson correlation of columns, excluding NA/null values:\n")

# Filtramos solo las columnas numéricas y calculamos la correlación de Pearson
# 'use = "complete.obs"' excluye los valores NA/null de forma similar a Python
columnas_numericas <- data[sapply(data, is.numeric)]
correlations <- cor(columnas_numericas, method = "pearson", use = "complete.obs")
print(correlations)

cat("------------------------------------------------------------\n")
cat("Correlacion por pares de variables:\n")

# Convertir la matriz de correlación en una lista de pares (equivalente a unstack)
corr_pairs <- as.data.frame(as.table(correlations))

# Renombramos las columnas para que sea más claro
colnames(corr_pairs) <- c("Variable_1", "Variable_2", "Correlacion")

# Filtrar los valores según tu condición (mayor a 0.75 o menor a -0.75)
# Excluimos el 1.0 perfecto usando 'Correlacion != 1'
strong_corrs <- subset(corr_pairs, abs(Correlacion) > 0.5 & Correlacion != 1.0)

cat("Pares de variables con correlación fuerte (> 0.5 o < -0.5):\n")
print(strong_corrs)



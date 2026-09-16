# ==============================================================================
# MACHINE LEARNING
# Práctica Nro 1: Apertura de archivos  12/09/2026
# ==============================================================================
# Load XLS Using readxl
# install.packages("readxl")
library(readxl)

# Ejecuta 'cls' si estás en Windows, o 'clear' si estás en Mac o Linux
if (.Platform$OS.type == "windows") {
  shell("cls")
} else {
  system("clear")
}

# Definir la ruta del archivo
getwd()
setwd("C:/GitHub-Local/ML/data/processed")
filename <- "1000-Registros-de-ventas.xlsx"
data <- read_excel(filename, sheet = "Ventas")

cat("------------------------------------------------------------\n")
cat("Primeras 10 filas del DataFrame:\n\n")
print(head(data, 10))

cat("------------------------------------------------------------\n")
cat("Muestra la forma del DataFrame (filas, columnas):\n")
print(dim(data)) # Devuelve un vector con [filas, columnas]

cat("------------------------------------------------------------\n")
cat("Estadísticas descriptivas del DataFrame:\n")
print(summary(data)) # Equivalente a data.describe()

cat("------------------------------------------------------------\n")
cat("Información del DataFrame:\n")
print(str(data)) # Equivalente a data.info() (Tipos de datos, estructura, etc.)

cat("------------------------------------------------------------\n")
cat("------------------------------------------------------------\n")
cat("------------------------------------------------------------\n")

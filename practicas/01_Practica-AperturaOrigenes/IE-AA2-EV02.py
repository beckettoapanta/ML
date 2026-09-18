# ==============================================================================
# MACHINE LEARNING
# Evidencia AA2-EV02:  Reporte de transformación de datos. 18/09/2026
# ==============================================================================
# Load CSV Using Pandas
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Ejecuta 'cls' si estás en Windows, o 'clear' si estás en Mac o Linux
os.system('cls' if os.name == 'nt' else 'clear')

filename = "data/processed/hotel_bookings.csv"

data = pd.read_csv(filename) 
print("------------------------------------------------------------")
print(f"Dimensiones originales: {data.shape}")
#"------------------------------------------------------------"
#"             DEPURACION DE DATOS"
#"------------------------------------------------------------"

# Eliminar columnas con exceso de nulos y baja relevancia predictiva (IDs)
df_clean = data.drop(columns=['company', 'agent'])

# Imputar la columna 'country' (0.4% de nulos) con una categoría nueva
df_clean['country'] = df_clean['country'].fillna('OTRO')

# Imputar la columna 'children' (4 nulos) con 0 (ausencia de niños)
df_clean['children'] = df_clean['children'].fillna(0)

#"------------------------------------------------------------"
#"             TRATAMIENTO DE VALORES ATÍPICOS"
#"------------------------------------------------------------"
# Eliminar el error masivo de facturación detectado en el Boxplot y Z-Score
df_clean = df_clean[df_clean['adr'] < 5000]

# Filtrar posibles tarifas negativas (errores de sistema o reembolsos mal digitados)
df_clean = df_clean[df_clean['adr'] >= 0]
#"------------------------------------------------------------")
#"            TRANSFORMACION DE VARIABLES CATEGÓRICAS")
#"------------------------------------------------------------")
# Aplicar One-Hot Encoding con drop_first=True para evitar la trampa de variables ficticias
df_transformed = pd.get_dummies(
    df_clean, 
    columns=['hotel', 'customer_type'], 
    drop_first=True
)

#------------------------------------------------------------
#         ESCALAMIENTO DE VARIABLES NUMERICAS")
#------------------------------------------------------------

# Aislar estrictamente las columnas numéricas que miden magnitudes o cantidades.
# Se excluye los IDs, fechas, y las columnas binarias generadas por One-Hot Encoding.
columnas_numericas = [
    'lead_time', 
    'stays_in_weekend_nights', 
    'stays_in_week_nights',
    'adults', 
    'children', 
    'babies', 
    'previous_cancellations',
    'previous_bookings_not_canceled', 
    'booking_changes',
    'days_in_waiting_list', 
    'adr', 
    'required_car_parking_spaces',
    'total_of_special_requests'
]

# Instanciar el algoritmo de estandarización
scaler = StandardScaler()

# Aplicar el escalado matemático
# Sobrescribimos las columnas originales con sus versiones estandarizadas
df_clean[columnas_numericas] = scaler.fit_transform(df_clean[columnas_numericas])

# Verificación técnica
# Comprobamos que la media (mean) sea ~0 y la desviación estándar (std) sea ~1
print(df_clean[columnas_numericas].describe().round(3))



#"------------------------------------------------------------")
#"         EXPORTACION DE DATOS LIMPIOS Y TRANSFORMADOS")
#"------------------------------------------------------------")

print("------------------------------------------------------------")
archivo_salida = 'data/processed/hotel_bookingsDef.csv'
df_transformed.to_csv(archivo_salida, index=False)

print(f"Dimensiones finales tras limpieza y transformación: {df_transformed.shape}")
print(f"Archivo '{archivo_salida}' generado exitosamente y listo para PCA/Modelado.")
print("------------------------------------------------------------")
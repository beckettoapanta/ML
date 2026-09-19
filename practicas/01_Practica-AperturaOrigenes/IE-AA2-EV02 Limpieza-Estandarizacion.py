# ==============================================================================
# MACHINE LEARNING
# Evidencia AA2-EV02:  Reporte de transformación de datos. 18/09/2026
# Código limpieza, estandarización numérica y codificación categórica
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
#------------------------------------------------------------
#             DEPURACION DE DATOS"
#------------------------------------------------------------

# Eliminar columnas con exceso de nulos y baja relevancia predictiva (IDs)
df_clean = data.drop(columns=['company', 'agent'])

# Imputar la columna 'country' (0.4% de nulos) con una categoría nueva
df_clean['country'] = df_clean['country'].fillna('OTRO')

# Imputar la columna 'children' (4 nulos) con 0 (ausencia de niños)
df_clean['children'] = df_clean['children'].fillna(0)

#------------------------------------------------------------
#             TRATAMIENTO DE VALORES ATÍPICOS"
#------------------------------------------------------------
# Eliminar el error masivo de facturación detectado en el Boxplot y Z-Score
df_clean = df_clean[df_clean['adr'] < 5000]

# Filtrar posibles tarifas negativas (errores de sistema o reembolsos mal digitados)
df_clean = df_clean[df_clean['adr'] >= 0]


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

#------------------------------------------------------------
#            TRANSFORMACION DE VARIABLES CATEGÓRICAS")
#------------------------------------------------------------
# Definir la lista de todas las variables categóricas estratégicas
# (Se asume que 'country' ya fue imputado con "OTRO" en el paso de limpieza)
columnas_categoricas = [
    'hotel',
    'meal',
    'country', 
    'market_segment',
    'distribution_channel',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'customer_type'
]

# NOTA DE NEGOCIO Y MODELADO: 
# No incluimos 'reservation_status' en esta lista. Si EL objetivo es predecir si un cliente
# va a cancelar (variable 'is_canceled'), incluir su estatus final de reserva le daría 
# la respuesta al algoritmo antes de tiempo. Esto se conoce como "Fuga de Datos" (Data Leakage) 
# y arruinaría la validez predictiva del modelo.

# Aplicar la transformación matemática (One-Hot Encoding)
df_transformed = pd.get_dummies(
    df_clean, 
    columns=columnas_categoricas, 
    drop_first=True
)

# 3. Forzar que los valores True/False se conviertan estrictamente en 1 y 0 (Opcional pero recomendado para algunos algoritmos)
# df_transformed = df_transformed.astype(int)


#------------------------------------------------------------
#          DEPURACIÓN DE DATOS POR REDUNANCIA 
#------------------------------------------------------------

# Definir la lista de columnas redundantes o peligrosas para el modelo
columnas_a_eliminar = [
    'reservation_status',       # Fuga de datos: Revela directamente si el cliente canceló o hizo check-out.
    'reservation_status_date',  # Fuga de datos: La fecha en la que se confirmó la cancelación o salida.
    'arrival_date_year',        # Ruido temporal: El año específico no ayuda a generalizar patrones futuros.
    'arrival_date_week_number', # Redundancia: La temporalidad ya se puede capturar con 'arrival_date_month'.
    'arrival_date_day_of_month', # Ruido: El día exacto (ej. 15 vs 16) aporta poco peso frente al 'lead_time'.
    'arrival_date_month'
]

# Ejecutar la eliminación (Drop)
df_depurado = df_transformed.drop(columns=columnas_a_eliminar, errors='ignore')

#------------------------------------------------------------
#          EXPORTACION DE DATOS LIMPIOS Y TRANSFORMADOS
#------------------------------------------------------------

print("------------------------------------------------------------")
archivo_salida = 'data/processed/hotel_bookingsDef.csv'
df_depurado.to_csv(archivo_salida, index=False)

print(f"Dimensiones finales tras limpieza y transformación: {df_transformed.shape}")
print(f"Archivo '{archivo_salida}' generado exitosamente y listo para PCA/Modelado.")
print("------------------------------------------------------------")


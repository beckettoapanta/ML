ml/
├── .gitignore               # Archivos que Git debe ignorar (datasets grandes, credenciales, etc.)
├── README.md                # Presentación del repositorio e índice de tus prácticas
├── requirements.txt         # Librerías necesarias para ejecutar tus códigos (pandas, scikit-learn...)
│
├── data/                    # NUNCA subas archivos gigantes a GitHub (usa datos de prueba o enlaces)
│   ├── raw/                 # Datos originales, puros y sin modificar (lectura obligatoria)
│   └── processed/           # Datos ya limpios, transformados y listos ("Tidy Data")
│
├── notebooks/               # Laboratorio de experimentación y análisis exploratorio (EDA)
│   ├── 01_eda_practica1.ipynb
│   └── 02_modelado_practica1.ipynb
│
├── src/                     # Código fuente modular y reutilizable (scripts .py)
│   ├── __init__.py
│   ├── data_processing.py   # Funciones para limpiar y transformar datos
│   └── evaluation.py        # Funciones para calcular métricas (Accuracy, F1-score...)
│
├── models/                  # Modelos ya entrenados y guardados (archivos .pkl, .joblib, .h5)
│   └── modelo_practica1.pkl
│
└── practicas/               # (Opcional) Carpetas numeradas si prefieres separar por proyectos individuales
    ├── 01_prediccion_precios/
    └── 02_clasificacion_imagenes/

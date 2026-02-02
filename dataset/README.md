# Documentación del Dataset

Este directorio contiene los archivos de datos utilizados y generados durante el proceso de entrenamiento y evaluación del modelo de detección de riesgo de diabetes.

## Contenido del Directorio

### 1. Archivos de Datos

| Archivo | Descripción | Razón de Existencia |
|---------|-------------|---------------------|
| **`dataset.csv`** | Archivo original con los datos crudos (en inglés). | Es la fuente primaria de información antes de cualquier procesamiento. |
| **`modificado.csv`** | Dataset completo preprocesado. Contiene las columnas renombradas al español, filtradas por relevancia y con la variable objetivo binarizada. | Sirve como punto de control intermedio para verificar la transformación completa de los datos antes de la división. |
| **`train.csv`** | Subconjunto de entrenamiento (80% de los datos). | Utilizado exclusivamente para entrenar el modelo `CatBoostClassifier`. |
| **`test.csv`** | Subconjunto de prueba (20% de los datos). | Utilizado exclusivamente para evaluar el rendimiento del modelo con datos no vistos (validación). |

### 2. Directorios Adicionales

- **`informe/`**: Contiene un reporte HTML (`informe_preprocesamiento.html`) y gráficos generados automáticamente que describen las estadísticas y distribuciones de las variables seleccionadas.

---

## Proceso de Generación

Los archivos `modificado.csv`, `train.csv` y `test.csv` son generados automáticamente por el script `scripts/preprocesamiento/preprocesamiento.py`. El flujo de transformación es el siguiente:

1.  **Carga**: Se lee el archivo original `dataset.csv`.
2.  **Traducción**: Se renombran las columnas de inglés a español para facilitar la interpretación (ej. `HighBP` -> `hipertension`).
3.  **Selección de Características**: Se conservan únicamente las columnas definidas como "factores individuales" relevantes para el modelo (ej. edad, IMC, hábitos), descartando información no utilizada.
4.  **Binarización del Objetivo**: La variable `estado_diabetes` (originalmente 0=Sano, 1=Prediabetes, 2=Diabetes) se transforma a binaria:
    - `0`: Sin riesgo.
    - `1`: Con riesgo (agrupa Prediabetes y Diabetes).
5.  **División (Split)**:
    - Se realiza una partición estratificada para mantener la proporción de clases.
    - **80%** para Entrenamiento (`train.csv`).
    - **20%** para Prueba (`test.csv`).
6.  **Exportación**: Se guardan los DataFrames resultantes en formato CSV.

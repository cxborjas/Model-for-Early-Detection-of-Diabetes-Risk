# Documentación del Módulo de Preprocesamiento# preprocesamiento.py



Este módulo (`preprocesamiento.py`) es el encargado de transformar los datos crudos originales en conjuntos de datos limpios y estructurados, listos para ser utilizados en el entrenamiento y validación del modelo de detección de riesgo de diabetes.Script encargado de la preparación del dataset original (BRFSS 2015) para el modelado predictivo del riesgo de diabetes.



## Funcionalidades Principales✔ Renombra columnas del inglés al español  

✔ Selecciona solo variables relevantes (factores individuales)  

El script realiza una serie de transformaciones secuenciales sobre el dataset original:✔ Binariza la variable objetivo (0 = sano, 1 = riesgo/prediabetes/diabetes)  

✔ Exporta `dataset/modificado.csv` listo para entrenamiento  

1.  **Carga de Datos**: Lee el archivo original `dataset/dataset.csv`.

2.  **Traducción de Variables**: Renombra las columnas del inglés original a nombres descriptivos en español (ej. `HighBP` -> `hipertension`, `BMI` -> `imc`).
3.  **Selección de Características**: Filtra el dataset para conservar únicamente las variables relevantes para el modelo, enfocándose en factores de riesgo individuales.
4.  **Binarización de la Variable Objetivo**: Transforma la variable `estado_diabetes` para simplificar el problema a una clasificación binaria:
    *   `0`: Sin riesgo (Sano).
    *   `1`: Con riesgo (Incluye prediabetes y diabetes).
5.  **División de Datos (Split)**: Separa los datos en dos conjuntos independientes utilizando muestreo estratificado para mantener la proporción de clases:
    *   **Entrenamiento (80%)**: Para entrenar el modelo.
    *   **Prueba (20%)**: Para evaluar el rendimiento final.
6.  **Generación de Informes**: Crea un reporte HTML automático con estadísticas descriptivas y gráficos de distribución de las variables procesadas.

## Variables del Modelo

El script selecciona y procesa las siguientes variables para el modelo predictivo:

| Variable Original | Variable Procesada | Descripción |
| :--- | :--- | :--- |
| `Diabetes_012` | **`estado_diabetes`** | Variable objetivo (0: Sano, 1: Riesgo). |
| `BMI` | **`imc`** | Índice de Masa Corporal. |
| `Age` | **`rango_edad`** | Categoría de edad (1-13). |
| `Sex` | **`sexo`** | Sexo biológico (0: Mujer, 1: Hombre). |
| `PhysActivity` | **`actividad_fisica_reciente`** | Actividad física en los últimos 30 días (0: No, 1: Sí). |
| `Fruits` | **`consumo_frutas`** | Consume fruta al menos una vez al día (0: No, 1: Sí). |
| `Veggies` | **`consumo_verduras`** | Consume verdura al menos una vez al día (0: No, 1: Sí). |
| `Smoker` | **`fumador_historico`** | Ha fumado al menos 100 cigarrillos en su vida (0: No, 1: Sí). |
| `HvyAlcoholConsump` | **`consumo_alcohol_elevado`** | Consumo excesivo de alcohol (0: No, 1: Sí). |
| `GenHlth` | **`salud_general`** | Autopercepción de salud (1: Excelente - 5: Pobre). |
| `PhysHlth` | **`dias_mala_salud_fisica`** | Días con mala salud física en el último mes. |
| `MentHlth` | **`dias_mala_salud_mental`** | Días con mala salud mental en el último mes. |
| `DiffWalk` | **`dificultad_caminar`** | Dificultad seria para caminar o subir escaleras (0: No, 1: Sí). |

## Entradas y Salidas

### Entradas
*   `dataset/dataset.csv`: Archivo CSV con los datos crudos originales.

### Salidas
El script genera los siguientes archivos:

1.  **Datasets Procesados**:
    *   `dataset/modificado.csv`: Dataset completo preprocesado.
    *   `dataset/train.csv`: Conjunto de entrenamiento (80%).
    *   `dataset/test.csv`: Conjunto de prueba (20%).

2.  **Informe de Calidad de Datos** (`dataset/informe/`):
    *   `informe_preprocesamiento.html`: Reporte interactivo con estadísticas y metadatos.
    *   `dist_estado_diabetes.png`: Gráfico de barras de la variable objetivo.
    *   `hist_imc.png`: Histograma de la distribución del IMC.

## Ejecución

Para ejecutar el proceso de limpieza y preparación de datos:

```bash
python scripts/preprocesamiento/preprocesamiento.py
```

Al finalizar, el script mostrará en consola la distribución de clases en los conjuntos de entrenamiento y prueba, confirmando que se ha mantenido el balance original.

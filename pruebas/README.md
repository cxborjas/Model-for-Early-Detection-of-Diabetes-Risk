# Resultados de Búsqueda de Hiperparámetros

Este directorio contiene los resultados de las pruebas exhaustivas (Grid Search) realizadas para optimizar el modelo de detección de riesgo de diabetes.

## Archivo: `grid_search_results.csv`

Este archivo CSV contiene el registro detallado de todas las combinaciones de hiperparámetros probadas y sus métricas de rendimiento correspondientes.

### Descripción de las Columnas

| Columna | Descripción |
| :--- | :--- |
| **iterations** | Número de árboles de decisión entrenados en el ensamble. |
| **learning_rate** | Tasa de aprendizaje utilizada para actualizar los pesos en cada iteración. |
| **depth** | Profundidad máxima de los árboles de decisión. |
| **l2_leaf_reg** | Coeficiente de regularización L2 para evitar el sobreajuste. |
| **recall** | Sensibilidad del modelo (Tasa de Verdaderos Positivos). Métrica principal priorizada. |
| **roc_auc** | Área bajo la curva ROC. Indica la capacidad de discriminación global del modelo. |
| **balance_score** | Promedio entre Recall y ROC AUC, utilizado para seleccionar el mejor equilibrio. |

### Interpretación

Cada fila representa un experimento único con una configuración específica. El objetivo de este archivo es permitir el análisis comparativo para seleccionar la configuración que maximice el **Recall** sin sacrificar excesivamente la precisión global (**ROC AUC**).

La configuración óptima seleccionada se basa en el valor más alto de `balance_score` o `recall`, según la prioridad clínica del modelo.

# Documentación del Módulo de Entrenamiento

Este módulo (`entrenamiento.py`) es el responsable de entrenar, validar y evaluar el modelo de aprendizaje automático para la detección temprana del riesgo de diabetes. Utiliza el algoritmo **CatBoostClassifier** y está diseñado siguiendo principios de programación orientada a objetos para facilitar su mantenimiento y escalabilidad.

## Estructura del Código

El script se organiza principalmente en dos clases:

### 1. Clase `ConfiguracionModelo`
Es una `dataclass` que almacena todos los hiperparámetros y configuraciones del modelo. Esto permite modificar el comportamiento del entrenamiento sin alterar la lógica interna.

***iteraciones**: Número de árboles de decisión a entrenar (por defecto: 150).
***tasa_aprendizaje**: Velocidad a la que el modelo aprende de los errores (por defecto: 0.07).
***profundidad**: Profundidad máxima de los árboles (por defecto: 6).
***reg_l2_hoja**: Coeficiente de regularización L2 (por defecto: 3).
***semilla_aleatoria**: Semilla para garantizar la reproducibilidad de los resultados.
***pesos_clases**: Diccionario para manejar el desbalance de clases (da más peso a la clase minoritaria positiva).

### 2. Clase `DetectorRiesgoDiabetes`
Es la clase principal que orquesta todo el flujo de trabajo.

#### Métodos Principales:

*   **`__init__`**: Inicializa el detector, configura las rutas de salida y establece la semilla aleatoria.
*   **`cargar_datos()`**: Lee los archivos `train.csv` y `test.csv` desde el directorio `dataset/`. Separa las características (X) de la variable objetivo (y).
*   **`entrenar(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)`**: Configura e inicia el entrenamiento del modelo CatBoost. Utiliza métricas personalizadas como AUC y Recall durante el proceso.
*   **`optimizar_umbral(y_verdadero, y_proba)`**: Busca el umbral de decisión óptimo que maximiza el equilibrio entre sensibilidad y especificidad (Índice de Youden). Esto es crucial en modelos médicos para ajustar qué tan "estricto" es el modelo al clasificar un caso como positivo.
*   **`_calcular_metricas(...)`**: Genera un diccionario con métricas clave: Sensibilidad (Recall), ROC AUC, Puntaje de Balance y la matriz de confusión desglosada (VP, VN, FP, FN).
*   **`generar_graficos(...)`**: Crea visualizaciones detalladas del rendimiento:
    *   Curvas de evolución de ROC AUC durante el entrenamiento.
    *   Matriz de confusión visual.
    *   Curva ROC con el umbral óptimo marcado.
*   **`ejecutar()`**: Método maestro que ejecuta secuencialmente todos los pasos: carga, entrenamiento, optimización, evaluación, guardado de artefactos y generación de reportes.

## Entradas y Salidas

### Entradas Requeridas
El script espera encontrar la siguiente estructura de carpetas en la raíz del proyecto:
*   `dataset/train.csv`: Datos de entrenamiento.
*   `dataset/test.csv`: Datos de prueba.

### Salidas Generadas
Todos los resultados se guardan automáticamente en la carpeta `resultados/`:

1.  **`modelo.pkl`**: Archivo binario con el modelo entrenado, el umbral óptimo y las métricas. Listo para ser usado en producción.
2.  **`historial_entrenamiento.csv` y `historial_prueba.csv`**: Datos crudos de la evolución del aprendizaje paso a paso.
3.  **Gráficos (.png)**:
    *   `evolucion_entrenamiento.png` / `evolucion_prueba.png`: Progreso del aprendizaje.
    *   `matriz_confusion.png`: Desempeño en clasificación de clases.
    *   `curva_roc_prueba.png`: Capacidad de discriminación del modelo.

## Ejecución

Para ejecutar el entrenamiento manualmente desde la terminal:

```bash
python scripts/entrenamiento/entrenamiento.py
```

El script imprimirá en consola un reporte detallado del proceso, incluyendo la distribución de datos, el progreso del entrenamiento y las métricas finales comparativas.

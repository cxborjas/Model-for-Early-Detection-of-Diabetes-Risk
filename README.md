## Autores / Authors

**Autor 1:** Jesús Ernesto González Torres  
**Institución/Dependencia/País:** Universidad Estatal de Bolívar, Departamento de Tecnologías de la Información y Comunicación, Ecuador.  
**Correo electrónico:** jegonzalez@mailes.ueb.edu.ec  
**ORCID:** [https://orcid.org/0009-0007-0693-0320](https://orcid.org/0009-0007-0693-0320)

**Autor 2:** Alexander Ufredo Alegría Chaves  
**Institución/Dependencia/País:** Universidad Estatal de Bolívar, Departamento de Tecnologías de la Información y Comunicación, Ecuador.  
**Correo electrónico:** alalegria@mailes.ueb.edu.ec  
**ORCID:** [https://orcid.org/0009-0001-3700-2515](https://orcid.org/0009-0001-3700-2515)

**Autor 3:** Diana Magali Alegría Camino  
**Institución/Dependencia/País:** Instituto Superior Tecnológico “El Libertador”, Carrera de Desarrollo de Software, Ecuador.  
**Correo electrónico:** dalegria@istel.edu.ec  
**ORCID:** [https://orcid.org/0009-0002-3670-9479](https://orcid.org/0009-0002-3670-9479)

**Autor 4:** Verónica Elizabeth Sánchez Aguiar  
**Institución/Dependencia/País:** Instituto Superior Tecnológico “El Libertador”, Carrera de Electrónica, Ecuador.  
**Correo electrónico:** vsanchez@istel.edu.ec  
**ORCID:** [https://orcid.org/0009-0003-4415-9566](https://orcid.org/0009-0003-4415-9566)

**Correo electrónico del autor de correspondencia:** jegonzalez@mailes.ueb.edu.ec

### Revisor Técnico / Technical Reviewer
**Nombre:** Claudio Borja Saltos  
**Rol:** Revisor del código para el modelo / Model Code Reviewer
**ORCID:** [https://orcid.org/0009-0008-6938-9399](https://orcid.org/0009-0008-6938-9399)

---

## Descripcion del Proyecto

Sistema integral de Machine Learning para la detección temprana del riesgo de diabetes tipo 2 utilizando algoritmos supervisados de aprendizaje automático. El proyecto incluye desde el preprocesamiento de datos hasta una aplicación de escritorio completamente funcional para evaluación en tiempo real.

### Caracteristicas Principales

- **Preprocesamiento automatizado** de datos con limpieza y transformación
- **Entrenamiento de modelo** Algoritmos CatBoost
- **Optimización de hiperparametros** mediante Grid Search
- **Evaluacion completa** con métricas de rendimiento y visualizaciones
- **Aplicacion de escritorio** con interfaz gráfica intuitiva (PyQt5)
- **Generacion de reportes** detallados con recomendaciones personalizadas

## Objetivo

Desarrollar un modelo predictivo que permita identificar tempranamente a personas en riesgo de desarrollar diabetes tipo 2, basándose en factores individuales como:
- Indice de Masa Corporal (IMC)
- Edad y sexo biológico
- Habitos de estilo de vida (actividad física, alimentación, tabaquismo)
- Estado de salud general y mental
- Movilidad física

## Estructura del Proyecto

```
.
├── dataset/                      # Datos del proyecto
│   ├── dataset.csv              # Dataset original
│   ├── modificado.csv           # Dataset preprocesado
│   ├── train.csv                # Conjunto de entrenamiento
│   └── test.csv                 # Conjunto de prueba
│
├── scripts/                      # Scripts de procesamiento y aplicación
│   ├── preprocesamiento/        # Scripts de limpieza de datos
│   │   ├── preprocesamiento.py  # Script principal de preprocesamiento
│   │   └── README.md            # Documentación del preprocesamiento
│   │
│   ├── entrenamiento/           # Scripts de entrenamiento
│   │   ├── entrenamiento.py     # Script principal de entrenamiento
│   │   └── README.md            # Documentación del entrenamiento
│   │
│   └── app/                     # Aplicación de escritorio
│       ├── app.py               # Aplicación principal PyQt5
│       └── README.md            # Documentación de la aplicación
│
├── resultados/                   # Resultados del entrenamiento
│   ├── modelo.pkl               # Modelo entrenado serializado
│   ├── matriz_confusion.png     # Matriz de confusión
│   ├── curva_roc_prueba.png     # Curva ROC
│   ├── evolucion_entrenamiento.png  # Métricas de entrenamiento
│   ├── evolucion_prueba.png     # Métricas de validación
│   ├── historial_entrenamiento.csv  # Log de entrenamiento
│   └── historial_prueba.csv     # Log de prueba
│
├── pruebas/                      # Resultados de experimentación
│   └── grid_search_results.csv  # Resultados de búsqueda de hiperparámetros
│
├── .gitignore                    # Archivos ignorados por Git
├── requerimientos.txt            # Dependencias del proyecto
└── README.md                     # Este archivo
```

## Instalacion y Configuracion

### Prerrequisitos

- Python 3.7 o superior
- pip (gestor de paquetes de Python)
- Git (opcional)

### Pasos de Instalacion
**Crear entorno virtual**
```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requerimientos.txt
```

## Dependencias Principales

| Libreria | Version | Proposito |
|----------|---------|-----------|
| pandas | >=2.3.0 | Manipulación de datos |
| numpy | >=2.3.0 | Operaciones numéricas |
| scikit-learn | >=1.7.0 | Algoritmos de ML |
| catboost | >=1.2.0 | Modelo de gradient boosting |
| matplotlib | >=3.10.0 | Visualizaciones |
| joblib | >=1.5.0 | Serialización de modelos |
| PyQt5 | >=5.15.11 | Interfaz gráfica |

Ver `requerimientos.txt` para la lista completa.

## Pipeline del Proyecto

### 1. Preprocesamiento de Datos

**Script**: `scripts/preprocesamiento/preprocesamiento.py`

Proceso automatizado que incluye:
- Carga de datos desde CSV
- Limpieza de valores faltantes o inconsistentes
- Codificación de variables categóricas
- Normalización/estandarización de características
- División estratificada en conjuntos de entrenamiento y prueba
- Guardado de datasets procesados

**Entrada**: `dataset/dataset.csv`
**Salida**: `dataset/modificado.csv`, `dataset/train.csv`, `dataset/test.csv`

**Ejecucion**:
```bash
python scripts/preprocesamiento/preprocesamiento.py
```

### 2. Entrenamiento del Modelo

**Script**: `scripts/entrenamiento/entrenamiento.py`

Proceso de entrenamiento que incluye:
- Carga de datos preprocesados
- Entrenamiento de múltiples modelos (CatBoost, XGBoost, Random Forest, etc.)
- Optimización de hiperparámetros mediante Grid Search
- Validación cruzada estratificada
- Selección del mejor umbral de decisión
- Evaluación con múltiples métricas
- Generación de visualizaciones
- Serialización del modelo final

**Entrada**: `dataset/train.csv`, `dataset/test.csv`
**Salida**: Archivos en `resultados/`

**Ejecucion**:
```bash
python scripts/entrenamiento/entrenamiento.py
```

**Metricas Evaluadas**:
- Precision (Precisión)
- Recall (Sensibilidad)
- ROC-AUC
- Matriz de Confusión

### 3. Aplicacion de Evaluacion

**Script**: `scripts/app/app.py`

Aplicación de escritorio con interfaz gráfica que permite:
- Ingreso de datos del paciente mediante formulario
- Cálculo automático del IMC
- Predicción en tiempo real del riesgo
- Visualización de resultados con barras de progreso
- Análisis detallado de factores de riesgo/protectores
- Recomendaciones personalizadas por prioridad
- Navegación intuitiva entre vistas

**Entrada**: Datos del usuario + `resultados/modelo.pkl`
**Salida**: Reporte visual interactivo

**Ejecucion**:
```bash
# Linux/Mac
.venv/bin/python scripts/app/app.py

# Windows
.venv\Scripts\python scripts\app\app.py
```

## Caracteristicas del Modelo

### Variables de Entrada

| Variable | Tipo | Descripcion |
|----------|------|-------------|
| IMC | Continua | Indice de Masa Corporal (calculado) |
| Rango de Edad | Ordinal | Grupo etario (1-13) |
| Sexo | Binaria | 0: Femenino, 1: Masculino |
| Actividad Fisica | Binaria | Ejercicio regular (0: No, 1: Si) |
| Consumo de Frutas | Binaria | Consumo diario (0: No, 1: Si) |
| Consumo de Verduras | Binaria | Consumo diario (0: No, 1: Si) |
| Fumador Historico | Binaria | Fuma o ha fumado (0: No, 1: Si) |
| Consumo de Alcohol | Binaria | Consumo frecuente (0: No, 1: Si) |
| Salud General | Ordinal | Percepción de salud (1-5) |
| Dias Mala Salud Fisica | Continua | Dias en último mes (0-30) |
| Dias Mala Salud Mental | Continua | Dias en último mes (0-30) |
| Dificultad para Caminar | Binaria | Problemas de movilidad (0: No, 1: Si) |

### Variable Objetivo

- **Diabetes**: Variable binaria (0: Sin diabetes, 1: Con diabetes o prediabetes)

**CatBoost** (Modelo principal)
   - Manejo nativo de variables categóricas
   - Resistente al overfitting
   - Alta precisión en datos tabulares


## Resultados y Metricas

Los resultados del entrenamiento se guardan en `resultados/`:

### Visualizaciones Generadas

1. **Matriz de Confusion** (`matriz_confusion.png`)
   - Verdaderos/Falsos Positivos y Negativos
   - Visualización de aciertos y errores

2. **Curva ROC** (`curva_roc_prueba.png`)
   - Relación Sensibilidad vs Especificidad
   - Area bajo la curva (AUC)

### Archivos de Historial

- `historial_entrenamiento.csv`: Métricas por época en conjunto de entrenamiento
- `historial_prueba.csv`: Métricas por época en conjunto de prueba
- `pruebas/grid_search_results.csv`: Resultados de búsqueda de hiperparámetros

## Uso de la Aplicacion

### Formulario de Evaluacion

1. **Informacion Personal**
   - Ingresar peso (kg) y altura (cm)
   - Seleccionar rango de edad
   - Indicar sexo biológico

2. **Habitos de Vida**
   - Actividad física regular
   - Consumo de frutas y verduras
   - Tabaquismo
   - Consumo de alcohol

3. **Estado de Salud**
   - Salud general percibida
   - Días con mala salud física/mental
   - Dificultades de movilidad

4. **Evaluacion**
   - Clic en "Evaluar Riesgo"
   - Visualización de resultados

### Pantalla de Resultados

**Columna 1: Conclusion**
- Resultado de la predicción (Riesgo Alto/Bajo)
- Probabilidad calculada
- Factores identificados
- Acción principal recomendada

**Columna 2: Analisis Detallado**
- Barras de progreso por factor
- Nivel de riesgo por parámetro (Alto/Medio/Bajo)
- Interpretación visual con colores

**Columna 3: Recomendaciones**
- Lista priorizada de acciones
- Prioridad Alta
- Prioridad Media
- Prioridad Baja

## Configuracion y Personalizacion

### Modificar Hiperparametros

Editar en `scripts/entrenamiento/entrenamiento.py`:
```python
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    # ...
}
```

### Ajustar Umbrales de Riesgo

Editar en `scripts/app/app.py`, funciones `_generar_*`:
```python
if imc >= 30:  # Cambiar umbral de obesidad
    # ...
```

### Personalizar Interfaz

Modificar estilos CSS inline en `scripts/app/app.py`:
```python
widget.setStyleSheet("""
    QPushButton {
        background-color: #2E86AB;
        /* agregar estilos personalizados */
    }
""")
```

## Pruebas y Validacion

### Ejecutar Pruebas del Modelo

```bash
python scripts/entrenamiento/entrenamiento.py
```

Revisa las métricas en:
- Terminal (salida estándar)
- `resultados/historial_*.csv`
- Gráficas en `resultados/*.png`

### Validar Aplicacion

```bash
.venv/bin/python scripts/app/app.py
```

Probar con datos de ejemplo y verificar:
- Cálculo correcto del IMC
- Predicciones consistentes
- Visualización de resultados
- Navegación entre pantallas

## Limitaciones y Consideraciones

1. **Uso Orientativo**: Los resultados son preliminares y no reemplazan diagnóstico médico profesional
2. **Datos de Entrenamiento**: El modelo está limitado por la calidad y representatividad de los datos de entrenamiento
3. **Factores No Considerados**: No incluye antecedentes familiares directos ni factores genéticos
4. **Validación Clínica**: Requiere validación en entornos clínicos antes de uso médico
5. **Actualización Periódica**: El modelo debe reentrenarse con datos actualizados regularmente

## Privacidad y Seguridad

- Los datos ingresados no se almacenan ni transmiten
- La evaluación se realiza localmente en la máquina del usuario
- No requiere conexión a internet para funcionar
- Cumple con principios de privacidad by design

## Solucion de Problemas

### Error: "No se pudo cargar el modelo"
```bash
# Verificar que existe el archivo
ls -la resultados/modelo.pkl

# Reentrenar si es necesario
python scripts/entrenamiento/entrenamiento.py
```

### Error: Modulos no encontrados
```bash
# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Reinstalar dependencias
pip install -r requerimientos.txt
```

### Problemas de rendimiento
- Reducir iteraciones en Grid Search
- Usar subconjunto de datos para pruebas
- Ajustar parámetros de CatBoost

## Licencia

Este proyecto es parte de un trabajo académico/investigación sobre detección temprana de diabetes.

---

**Nota Importante**: Esta herramienta es de apoyo educativo e investigativo. Cualquier decisión médica debe ser tomada por profesionales de la salud calificados.

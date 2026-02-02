# Evaluador de Riesgo Temprano de Diabetes

## Descripción

Aplicación de escritorio desarrollada con PyQt5 que evalúa el riesgo de diabetes tipo 2 utilizando un modelo de aprendizaje automático entrenado. La aplicación proporciona una interfaz gráfica intuitiva para ingresar datos del paciente y obtener una evaluación personalizada del riesgo junto con recomendaciones específicas.

## Características

- **Interfaz gráfica moderna**: Diseño limpio y profesional con navegación intuitiva
- **Evaluación en tiempo real**: Predicción instantánea basada en modelo de ML optimizado
- **Análisis completo**: Visualización detallada de factores de riesgo y protectores
- **Recomendaciones personalizadas**: Sugerencias específicas según el perfil del usuario
- **Adaptación automática**: Se ajusta a diferentes resoluciones de pantalla

## Requisitos del Sistema

### Software
- Python 3.7 o superior
- Sistema operativo: Windows, Linux o macOS
- Entorno virtual Python (recomendado)

### Dependencias
```
PyQt5 >= 5.15.0
pandas >= 1.3.0
numpy >= 1.21.0
joblib >= 1.0.0
scikit-learn >= 1.0.0
```

## Instalación

1. **Clonar o descargar el proyecto**
```bash
cd "ruta/al/proyecto"
```

2. **Crear y activar entorno virtual**
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

## Ejecución

### Linux/Mac
```bash
cd "Modelo para la detección temprana del riesgo de diabetes usando algoritmos supervisados de aprendizaje automático basados en factores individuales"
.venv/bin/python scripts/app/app.py
```

### Windows
```bash
cd "Modelo para la detección temprana del riesgo de diabetes usando algoritmos supervisados de aprendizaje automático basados en factores individuales"
.venv\Scripts\python scripts\app\app.py
```

## Uso de la Aplicación

### 1. Pantalla Principal - Formulario

La aplicación inicia con un formulario dividido en tres secciones:

#### Información Personal
- **Peso (kg)**: Peso corporal en kilogramos (rango: 30-200)
- **Altura (cm)**: Altura en centímetros (rango: 100-250)
- **Rango de edad**: Seleccionar grupo etario (18-80+)
- **Sexo biológico**: Femenino o Masculino

#### Hábitos y Estilo de Vida
- **Ejercicio regular**: ¿Realiza actividad física regularmente?
- **Consumo de frutas**: ¿Consume frutas diariamente?
- **Consumo de verduras**: ¿Consume verduras diariamente?
- **Tabaquismo**: ¿Fuma o ha fumado?
- **Consumo de alcohol**: ¿Consume alcohol frecuentemente?

#### Estado de Salud
- **Salud general**: Percepción personal del estado de salud (Excelente a Mala)
- **Días con mala salud física**: Frecuencia en el último mes
- **Días con estrés/malestar emocional**: Frecuencia en el último mes
- **Dificultad para caminar**: ¿Tiene dificultad para caminar o subir escaleras?

### 2. Botones de Acción

- **Limpiar**: Resetea todos los campos del formulario
- **Evaluar Riesgo**: Procesa los datos y genera el reporte
- **Salir**: Cierra la aplicación

### 3. Pantalla de Resultados

Tras evaluar, se muestra una pantalla dividida en tres columnas:

#### Conclusión del Análisis
- Resultado de la predicción (Riesgo Alto/Bajo)
- Probabilidad calculada por el modelo
- Factores identificados (riesgo o protectores)
- Acción recomendada principal

#### Análisis Detallado
Visualización con barras de progreso de:
- IMC (Índice de Masa Corporal)
- Edad
- Actividad Física
- Alimentación
- Tabaquismo
- Consumo de Alcohol
- Salud General
- Salud Física
- Salud Mental
- Movilidad

Cada parámetro muestra su nivel de riesgo mediante colores:
- **Verde**: Bajo riesgo / Saludable
- **Amarillo**: Riesgo medio / Atención
- **Rojo**: Alto riesgo / Crítico

#### Recomendaciones
Lista priorizada de acciones específicas:
- **Prioridad Alta** (rojo): Acciones urgentes
- **Prioridad Media** (amarillo): Mejoras importantes
- **Prioridad Baja** (verde): Mantenimiento preventivo

### 4. Navegación

- **Volver al Formulario**: Regresa a la pantalla de ingreso de datos
- **Salir de la Aplicación**: Cierra el programa

## Estructura del Código

### Clase Principal: `EvaluadorRiesgoDiabetes`

```python
class EvaluadorRiesgoDiabetes(QMainWindow):
    """Aplicación de evaluación de riesgo de diabetes usando ML."""
```

#### Métodos Principales

**Inicialización y Configuración**
- `__init__()`: Constructor principal
- `_configurar_geometria()`: Configura tamaño y posición de la ventana
- `cargar_modelo()`: Carga el modelo ML desde disco
- `inicializar_interfaz()`: Construye la interfaz gráfica

**Interfaz de Usuario**
- `_crear_grupo()`: Crea grupos visuales en el formulario
- `_crear_campo_numero()`: Genera campos numéricos (peso, altura)
- `_crear_campo_combo()`: Genera selectores desplegables

**Funcionalidad Principal**
- `limpiar_formulario()`: Resetea todos los campos
- `evaluar_riesgo()`: Ejecuta el modelo de predicción
- `mostrar_resultado()`: Genera y muestra el reporte completo

**Generación de Reportes**
- `_generar_conclusion_mejorada()`: Crea el resumen ejecutivo
- `_generar_analisis_visual()`: Genera barras de análisis de factores
- `_generar_recomendaciones_mejoradas()`: Crea lista priorizada de recomendaciones

**Utilidades**
- `_obtener_categoria_imc()`: Calcula categoría de IMC

## Modelo de Machine Learning

### Archivo del Modelo
- **Ubicación**: `resultados/modelo.pkl`
- **Formato**: Pickle serializado con joblib
- **Contenido**: Diccionario con modelo entrenado e información del umbral

### Características Utilizadas
El modelo utiliza las siguientes características:
- IMC (calculado automáticamente)
- Rango de edad (codificado 1-13)
- Sexo (0: Femenino, 1: Masculino)
- Actividad física reciente (binario)
- Consumo de frutas (binario)
- Consumo de verduras (binario)
- Fumador histórico (binario)
- Consumo de alcohol elevado (binario)
- Salud general (escala 1-5)
- Días de mala salud física (0-30)
- Días de mala salud mental (0-30)
- Dificultad para caminar (binario)

### Interpretación de Resultados

**Riesgo Alto**
- Probabilidad ≥ umbral óptimo (determinado en entrenamiento)
- Requiere atención médica inmediata
- Exámenes recomendados: Glucemia en ayunas y HbA1c

**Riesgo Bajo**
- Probabilidad < umbral óptimo
- Mantener hábitos saludables
- Chequeos preventivos anuales

## Personalización

### Modificar Estilos

Los estilos CSS están definidos inline en cada widget. Para modificarlos:

```python
widget.setStyleSheet("""
    QPushButton {
        background-color: #2E86AB;
        color: white;
        /* agregar más estilos */
    }
""")
```

### Ajustar Umbrales de Riesgo

Los umbrales están definidos en las funciones `_generar_*`:

```python
if imc >= 30:  # Obesidad
    # ...
elif imc >= 25:  # Sobrepeso
    # ...
```

### Agregar Nuevos Campos

1. Crear el campo en `inicializar_interfaz()`
2. Almacenarlo en `self.campos`
3. Actualizar el procesamiento en `evaluar_riesgo()`
4. Reentrenar el modelo con la nueva característica

## Solución de Problemas

### Error: "No se pudo cargar el modelo"
- Verificar que existe `resultados/modelo.pkl`
- Verificar permisos de lectura del archivo
- Comprobar versión de scikit-learn compatible

### Error: "ModuleNotFoundError: No module named 'PyQt5'"
```bash
pip install PyQt5
```

### Advertencia: "QSocketNotifier: Can only be used with threads started with QThread"
- Es una advertencia de Qt, no afecta la funcionalidad
- Se puede ignorar de forma segura

### La ventana no se maximiza correctamente
- Verificar configuración del gestor de ventanas
- El código usa `self.showMaximized()` automáticamente

## Mejores Prácticas de Uso

1. **Datos Precisos**: Ingresar información real y actualizada
2. **Consulta Médica**: Los resultados son orientativos, no reemplazan diagnóstico médico
3. **Reevaluación**: Realizar nuevas evaluaciones cada 3-6 meses
4. **Seguimiento**: Mantener registro de resultados históricos

## Limitaciones

- El modelo está entrenado con datos históricos específicos
- No considera factores genéticos o antecedentes familiares directos
- No reemplaza exámenes clínicos de laboratorio
- Requiere datos completos para predicción precisa

## Mantenimiento

### Actualizar el Modelo
1. Entrenar nuevo modelo con datos actualizados
2. Guardar como `resultados/modelo.pkl`
3. Verificar compatibilidad de características
4. Probar con casos conocidos

### Logs y Debug
Para habilitar modo debug, agregar al inicio de `main()`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Licencia

Este proyecto es parte de un trabajo académico/investigación sobre detección temprana de diabetes.

## Contacto y Soporte

Para reportar problemas o sugerencias:
- Crear un issue en el repositorio del proyecto
- Documentar claramente el error o mejora propuesta
- Incluir información del sistema y versiones de dependencias

## Versión

**Versión actual**: 1.0.0  
**Última actualización**: Noviembre 2025  
**Compatible con**: Python 3.7+, PyQt5 5.15+

---

**Nota Importante**: Esta aplicación es una herramienta de evaluación preliminar y no sustituye la consulta médica profesional. Ante cualquier preocupación sobre su salud, consulte con un profesional de la salud calificado.

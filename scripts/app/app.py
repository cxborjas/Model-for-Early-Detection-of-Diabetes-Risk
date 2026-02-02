#!/usr/bin/env python3

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QComboBox, QSpinBox, QPushButton, QMessageBox,
    QGroupBox, QScrollArea, QFrame, QDialog, QStackedWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class EvaluadorRiesgoDiabetes(QMainWindow):
    """Aplicación de evaluación de riesgo de diabetes usando ML."""
    
    def __init__(self):
        super().__init__()
        self.ruta_base = Path(__file__).parent.parent.parent
        self.modelo_info = None
        self._configurar_geometria()
        self.cargar_modelo()
        self.inicializar_interfaz()
    
    def _configurar_geometria(self):
        """Configura la resolución de pantalla."""
        from PyQt5.QtWidgets import QDesktopWidget
        desktop = QDesktopWidget()
        self.screen_geometry = desktop.screenGeometry()
        self.screen_width = self.screen_geometry.width()
        self.screen_height = self.screen_geometry.height()
        
    def cargar_modelo(self):
        """Carga el modelo de ML desde el disco."""
        ruta_modelo = self.ruta_base / "resultados" / "modelo.pkl"
        try:
            self.modelo_info = joblib.load(ruta_modelo)
        except Exception as e:
            QMessageBox.critical(
                None, 
                "Error", 
                f"No se pudo cargar el modelo:\n{str(e)}"
            )
            sys.exit(1)
    
    def inicializar_interfaz(self):
        """Inicializa la interfaz gráfica de usuario."""
        self.setWindowTitle("Evaluador de Riesgo Temprano de Diabetes")
        self.showMaximized()
        
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_main = QVBoxLayout(widget_central)
        layout_main.setContentsMargins(0, 0, 0, 0)
        
        # Crear el StackedWidget para manejar las vistas
        self.stack = QStackedWidget()
        layout_main.addWidget(self.stack)
        
        # --- VISTA 1: FORMULARIO ---
        self.vista_formulario = QWidget()
        layout_formulario_vista = QVBoxLayout(self.vista_formulario)
        
        titulo = QLabel("Evaluador de Riesgo de Diabetes")
        titulo.setFont(QFont("Arial", 18, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("color: #2E86AB; padding: 15px; background-color: #f0f8ff; border-radius: 5px;")
        layout_formulario_vista.addWidget(titulo)
        
        info_modelo = QLabel(f"""
        <div style='text-align: center;'>
        <b>Modelo:</b> CatBoost Classifier | 
        <b>Sensibilidad:</b> {self.modelo_info['metricas']['sensibilidad']*100:.2f}% | 
        <b>ROC AUC:</b> {self.modelo_info['metricas']['roc_auc']*100:.2f}% | 
        <b>Umbral:</b> {self.modelo_info['umbral_optimo']*100:.2f}%
        </div>
        """)
        info_modelo.setStyleSheet("background-color: #e8f4f8; padding: 8px; border-radius: 5px; font-size: 11px;")
        layout_formulario_vista.addWidget(info_modelo)
        
        layout_campos = QHBoxLayout()
        self.campos = {}
        
        # Columna 1
        columna1 = QVBoxLayout()
        
        grupo_antropometrico = self._crear_grupo("Información Personal")
        layout_antropometrico = QVBoxLayout()
        
        layout_peso, widget_peso = self._crear_campo_numero(
            "Peso (kg):", 30, 200, 0, 0, "Su peso corporal en kilogramos"
        )
        self.peso_widget = widget_peso
        layout_antropometrico.addLayout(layout_peso)
        
        layout_altura, widget_altura = self._crear_campo_numero(
            "Altura (cm):", 100, 250, 0, 0, "Su altura en centímetros"
        )
        self.altura_widget = widget_altura
        layout_antropometrico.addLayout(layout_altura)
        
        layout_edad, widget_edad = self._crear_campo_combo(
            "¿Cuál es su rango de edad?",
            ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
             "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        )
        self.campos['rango_edad'] = widget_edad
        layout_antropometrico.addLayout(layout_edad)
        
        layout_sexo, widget_sexo = self._crear_campo_combo(
            "Sexo biológico:",
            ["Femenino", "Masculino"],
            [0, 1]
        )
        self.campos['sexo'] = widget_sexo
        layout_antropometrico.addLayout(layout_sexo)
        
        grupo_antropometrico.setLayout(layout_antropometrico)
        columna1.addWidget(grupo_antropometrico)
        columna1.addStretch()
        layout_campos.addLayout(columna1)
        
        # Columna 2
        columna2 = QVBoxLayout()
        
        grupo_habitos = self._crear_grupo("Hábitos y Estilo de Vida")
        layout_habitos = QVBoxLayout()
        
        layout_actividad, widget_actividad = self._crear_campo_combo(
            "¿Hace ejercicio regularmente?",
            ["No", "Sí"],
            [0, 1]
        )
        self.campos['actividad_fisica_reciente'] = widget_actividad
        layout_habitos.addLayout(layout_actividad)
        
        layout_frutas, widget_frutas = self._crear_campo_combo(
            "¿Come frutas diariamente?",
            ["No", "Sí"],
            [0, 1]
        )
        self.campos['consumo_frutas'] = widget_frutas
        layout_habitos.addLayout(layout_frutas)
        
        layout_verduras, widget_verduras = self._crear_campo_combo(
            "¿Come verduras diariamente?",
            ["No", "Sí"],
            [0, 1]
        )
        self.campos['consumo_verduras'] = widget_verduras
        layout_habitos.addLayout(layout_verduras)
        
        layout_fumador, widget_fumador = self._crear_campo_combo(
            "¿Fuma o ha fumado?",
            ["No", "Sí"],
            [0, 1]
        )
        self.campos['fumador_historico'] = widget_fumador
        layout_habitos.addLayout(layout_fumador)
        
        layout_alcohol, widget_alcohol = self._crear_campo_combo(
            "¿Consume alcohol frecuentemente?",
            ["No", "Sí"],
            [0, 1]
        )
        self.campos['consumo_alcohol_elevado'] = widget_alcohol
        layout_habitos.addLayout(layout_alcohol)
        
        grupo_habitos.setLayout(layout_habitos)
        columna2.addWidget(grupo_habitos)
        columna2.addStretch()
        layout_campos.addLayout(columna2)
        
        # Columna 3
        columna3 = QVBoxLayout()
        
        grupo_salud = self._crear_grupo("Estado de Salud")
        layout_salud = QVBoxLayout()
        
        layout_salud_gral, widget_salud_gral = self._crear_campo_combo(
            "¿Cómo es su salud general?",
            ["Excelente", "Muy buena", "Buena", "Regular", "Mala"],
            [1, 2, 3, 4, 5]
        )
        self.campos['salud_general'] = widget_salud_gral
        layout_salud.addLayout(layout_salud_gral)
        
        layout_dias_fisica, widget_dias_fisica = self._crear_campo_combo(
            "Días con mala salud física (último mes):",
            ["Ninguno", "1-7 días", "8-14 días", "15-21 días", "22-30 días"],
            [0, 5, 10, 18, 26]
        )
        self.campos['dias_mala_salud_fisica'] = widget_dias_fisica
        layout_salud.addLayout(layout_dias_fisica)
        
        layout_dias_mental, widget_dias_mental = self._crear_campo_combo(
            "Días con estrés/malestar emocional (último mes):",
            ["Ninguno", "1-7 días", "8-14 días", "15-21 días", "22-30 días"],
            [0, 5, 10, 18, 26]
        )
        self.campos['dias_mala_salud_mental'] = widget_dias_mental
        layout_salud.addLayout(layout_dias_mental)
        
        layout_dificultad, widget_dificultad = self._crear_campo_combo(
            "¿Dificultad para caminar/subir escaleras?",
            ["No", "Sí"],
            [0, 1]
        )
        self.campos['dificultad_caminar'] = widget_dificultad
        layout_salud.addLayout(layout_dificultad)
        
        grupo_salud.setLayout(layout_salud)
        columna3.addWidget(grupo_salud)
        columna3.addStretch()
        layout_campos.addLayout(columna3)
        
        layout_formulario_vista.addLayout(layout_campos)
        
        layout_botones = QHBoxLayout()
        layout_botones.addStretch()
        
        btn_limpiar = QPushButton("Limpiar")
        btn_limpiar.setCursor(Qt.PointingHandCursor)
        btn_limpiar.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                min-width: 130px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        btn_limpiar.clicked.connect(self.limpiar_formulario)
        layout_botones.addWidget(btn_limpiar)
        
        layout_botones.addSpacing(10)
        
        btn_evaluar = QPushButton("Evaluar Riesgo")
        btn_evaluar.setCursor(Qt.PointingHandCursor)
        btn_evaluar.setStyleSheet("""
            QPushButton {
                background-color: #2E86AB;
                color: white;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: #1a5276;
            }
        """)
        btn_evaluar.clicked.connect(self.evaluar_riesgo)
        layout_botones.addWidget(btn_evaluar)
        
        layout_botones.addSpacing(10)
        
        btn_salir_form = QPushButton("Salir")
        btn_salir_form.setCursor(Qt.PointingHandCursor)
        btn_salir_form.setStyleSheet("""
            QPushButton {
                background: white;
                color: #c0392b;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #e74c3c;
                border-radius: 5px;
                min-width: 110px;
            }
            QPushButton:hover {
                background: #ffe6e6;
                border-color: #c0392b;
            }
            QPushButton:pressed {
                background: #fadbd8;
            }
        """)
        btn_salir_form.clicked.connect(QApplication.instance().quit)
        layout_botones.addWidget(btn_salir_form)
        
        layout_botones.addStretch()
        layout_formulario_vista.addLayout(layout_botones)
        
        # --- VISTA 2: RESULTADOS (Inicialmente vacía) ---
        self.vista_resultados = QWidget()
        self.layout_resultados = QVBoxLayout(self.vista_resultados)
        
        # Agregar vistas al stack
        self.stack.addWidget(self.vista_formulario)
        self.stack.addWidget(self.vista_resultados)
        
        # Mostrar formulario inicialmente
        self.stack.setCurrentIndex(0)
        
    def _crear_grupo(self, titulo):
        grupo = QGroupBox(titulo)
        grupo.setFont(QFont("Arial", 11, QFont.Bold))
        grupo.setStyleSheet("""
            QGroupBox {
                border: 2px solid #d3d3d3;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 15px;
                background-color: #f9f9f9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #2E86AB;
            }
        """)
        return grupo
    
    def _crear_campo_numero(self, etiqueta, minimo, maximo, valor_defecto, decimales, tooltip=""):
        from PyQt5.QtWidgets import QDoubleSpinBox
        
        layout = QVBoxLayout()
        
        lbl = QLabel(etiqueta)
        lbl.setFont(QFont("Arial", 10))
        layout.addWidget(lbl)
        
        spinbox = QSpinBox() if decimales == 0 else QDoubleSpinBox()
        if decimales == 0:
            spinbox.setMinimum(int(minimo))
            spinbox.setMaximum(int(maximo))
            spinbox.setValue(int(valor_defecto))
        else:
            spinbox.setMinimum(minimo)
            spinbox.setMaximum(maximo)
            spinbox.setValue(valor_defecto)
            spinbox.setDecimals(decimales)
        spinbox.setToolTip(tooltip)
        spinbox.setStyleSheet("""
            QSpinBox, QDoubleSpinBox {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        layout.addWidget(spinbox)
        
        return layout, spinbox
    
    def _crear_campo_combo(self, etiqueta, opciones, valores):
        layout = QVBoxLayout()
        
        lbl = QLabel(etiqueta)
        lbl.setFont(QFont("Arial", 10))
        layout.addWidget(lbl)
        
        combo = QComboBox()
        for i, opcion in enumerate(opciones):
            combo.addItem(opcion, valores[i])
        combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        layout.addWidget(combo)
        
        return layout, combo
    
    def limpiar_formulario(self):
        respuesta = QMessageBox.question(
            self, "Confirmar", "¿Desea limpiar todos los campos?",
            QMessageBox.Yes | QMessageBox.No
        )
        if respuesta == QMessageBox.Yes:
            self.peso_widget.setValue(70)
            self.altura_widget.setValue(170)
            for nombre, widget in self.campos.items():
                if isinstance(widget, QComboBox):
                    widget.setCurrentIndex(0)
                else:
                    widget.setValue(widget.minimum())
    
    def evaluar_riesgo(self):
        try:
            datos = {}
            for nombre, widget in self.campos.items():
                if isinstance(widget, QComboBox):
                    datos[nombre] = widget.currentData()
                else:
                    datos[nombre] = widget.value()
            
            peso = self.peso_widget.value()
            altura_cm = self.altura_widget.value()
            altura_m = altura_cm / 100.0
            imc = peso / (altura_m ** 2)
            datos['imc'] = round(imc, 1)
            
            df_entrada = pd.DataFrame([datos])
            df_entrada = df_entrada[self.modelo_info['nombres_caracteristicas']]
            
            probabilidad = self.modelo_info['modelo'].predict_proba(df_entrada)[0, 1]
            umbral = self.modelo_info['umbral_optimo']
            prediccion = 1 if probabilidad >= umbral else 0
            
            self.mostrar_resultado(prediccion, probabilidad, umbral, imc, datos)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al realizar la evaluación:\n{str(e)}")
    
    def mostrar_resultado(self, prediccion, probabilidad, umbral, imc, datos):
        # Limpiar layout anterior de resultados si existe
        if self.layout_resultados.count() > 0:
            # Eliminar widgets anteriores
            while self.layout_resultados.count():
                item = self.layout_resultados.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        
        categoria_imc = self._obtener_categoria_imc(imc)
        
        layout = self.layout_resultados
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Encabezado con resultado principal
        header_layout = QHBoxLayout()
        
        if prediccion == 1:
            titulo_label = QLabel("RIESGO DETECTADO DE DIABETES")
            titulo_label.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e74c3c, stop:1 #c0392b);
                color: white; 
                padding: 20px; 
                font-size: 24px; 
                font-weight: bold; 
                border-radius: 8px;
                border: 2px solid #a93226;
            """)
        else:
            titulo_label = QLabel("RIESGO BAJO DE DIABETES")
            titulo_label.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #27ae60, stop:1 #229954);
                color: white; 
                padding: 20px; 
                font-size: 24px; 
                font-weight: bold; 
                border-radius: 8px;
                border: 2px solid #1e8449;
            """)
        
        titulo_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(titulo_label)
        
        # Panel de métricas clave
        metricas_widget = QWidget()
        metricas_widget.setStyleSheet("""
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 10px;
        """)
        metricas_layout = QVBoxLayout()
        metricas_widget.setLayout(metricas_layout)
        
        probabilidad_label = QLabel(f"<b style='font-size: 16px;'>Probabilidad</b><br><span style='font-size: 28px; color: {'#e74c3c' if prediccion == 1 else '#27ae60'};'><b>{probabilidad*100:.1f}%</b></span>")
        probabilidad_label.setAlignment(Qt.AlignCenter)
        metricas_layout.addWidget(probabilidad_label)
        
        umbral_label = QLabel(f"<span style='font-size: 11px; color: #6c757d;'>Umbral: {umbral*100:.1f}%</span>")
        umbral_label.setAlignment(Qt.AlignCenter)
        metricas_layout.addWidget(umbral_label)
        
        imc_label = QLabel(f"<span style='font-size: 11px; color: #6c757d;'>IMC: {imc:.1f} ({categoria_imc})</span>")
        imc_label.setAlignment(Qt.AlignCenter)
        metricas_layout.addWidget(imc_label)
        
        metricas_widget.setFixedWidth(200)
        header_layout.addWidget(metricas_widget)
        
        layout.addLayout(header_layout)
        
        # Separador decorativo
        separador_1 = QLabel()
        separador_1.setStyleSheet("background-color: #dee2e6; min-height: 2px; max-height: 2px;")
        layout.addWidget(separador_1)
        
        # Layout principal de 3 columnas
        layout_contenido = QHBoxLayout()
        layout_contenido.setSpacing(15)
        
        # COLUMNA 1: Conclusión del Análisis (35%)
        columna_izquierda = QWidget()
        columna_izquierda.setStyleSheet("""
            background-color: white;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
        """)
        layout_col_izq = QVBoxLayout()
        columna_izquierda.setLayout(layout_col_izq)
        
        titulo_conclusion = QLabel("CONCLUSIÓN DEL ANÁLISIS")
        titulo_conclusion.setFont(QFont("Arial", 13, QFont.Bold))
        titulo_conclusion.setStyleSheet("color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px;")
        layout_col_izq.addWidget(titulo_conclusion)
        
        conclusion_html = self._generar_conclusion_mejorada(prediccion, probabilidad, datos, imc)
        label_conclusion = QLabel(conclusion_html)
        label_conclusion.setWordWrap(True)
        label_conclusion.setTextFormat(Qt.RichText)
        layout_col_izq.addWidget(label_conclusion)
        
        layout_col_izq.addStretch()
        layout_contenido.addWidget(columna_izquierda, 35)
        
        # COLUMNA 2: Análisis de Factores (35%)
        columna_centro = QWidget()
        columna_centro.setStyleSheet("""
            background-color: white;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
        """)
        layout_col_centro = QVBoxLayout()
        columna_centro.setLayout(layout_col_centro)
        
        titulo_analisis = QLabel("ANÁLISIS DETALLADO")
        titulo_analisis.setFont(QFont("Arial", 13, QFont.Bold))
        titulo_analisis.setStyleSheet("color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px;")
        layout_col_centro.addWidget(titulo_analisis)
        
        analisis_html = self._generar_analisis_visual(datos, imc, prediccion)
        label_analisis = QLabel(analisis_html)
        label_analisis.setWordWrap(True)
        label_analisis.setTextFormat(Qt.RichText)
        
        scroll_analisis = QScrollArea()
        scroll_analisis.setWidget(label_analisis)
        scroll_analisis.setWidgetResizable(True)
        scroll_analisis.setFrameShape(QFrame.NoFrame)
        scroll_analisis.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        layout_col_centro.addWidget(scroll_analisis)
        layout_contenido.addWidget(columna_centro, 35)
        
        # COLUMNA 3: Recomendaciones (30%)
        columna_derecha = QWidget()
        columna_derecha.setStyleSheet("""
            background-color: white;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
        """)
        layout_col_der = QVBoxLayout()
        columna_derecha.setLayout(layout_col_der)
        
        titulo_recomendaciones = QLabel("RECOMENDACIONES")
        titulo_recomendaciones.setFont(QFont("Arial", 13, QFont.Bold))
        titulo_recomendaciones.setStyleSheet("color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px;")
        layout_col_der.addWidget(titulo_recomendaciones)
        
        recomendaciones_html = self._generar_recomendaciones_mejoradas(datos, imc, prediccion)
        label_recomendaciones = QLabel(recomendaciones_html)
        label_recomendaciones.setWordWrap(True)
        label_recomendaciones.setTextFormat(Qt.RichText)
        
        scroll_recomendaciones = QScrollArea()
        scroll_recomendaciones.setWidget(label_recomendaciones)
        scroll_recomendaciones.setWidgetResizable(True)
        scroll_recomendaciones.setFrameShape(QFrame.NoFrame)
        scroll_recomendaciones.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        layout_col_der.addWidget(scroll_recomendaciones)
        layout_contenido.addWidget(columna_derecha, 30)
        
        layout.addLayout(layout_contenido)
        
        # Separador decorativo inferior
        separador_2 = QLabel()
        separador_2.setStyleSheet("background-color: #dee2e6; min-height: 2px; max-height: 2px; margin-top: 10px;")
        layout.addWidget(separador_2)
        
        # Botones en la parte inferior con estilo mejorado
        frame_botones = QFrame()
        frame_botones.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-top: 1px solid #dee2e6;
                padding: 10px;
            }
        """)
        layout_botones = QHBoxLayout(frame_botones)
        layout_botones.setContentsMargins(20, 10, 20, 10)
        
        info_label = QLabel("Tip: Revise cada sección para entender mejor su perfil de riesgo")
        info_label.setStyleSheet("color: #6c757d; font-size: 12px; font-style: italic;")
        layout_botones.addWidget(info_label)
        
        layout_botones.addStretch()

        btn_volver = QPushButton("Volver al Formulario")
        btn_volver.setCursor(Qt.PointingHandCursor)
        btn_volver.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                min-width: 180px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #21618c);
                border: 1px solid #1a5276;
            }
            QPushButton:pressed {
                background: #1a5276;
                padding-top: 14px;
                padding-bottom: 10px;
            }
        """)
        btn_volver.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout_botones.addWidget(btn_volver)
        
        layout_botones.addSpacing(15)

        btn_salir = QPushButton("Salir de la Aplicación")
        btn_salir.setCursor(Qt.PointingHandCursor)
        btn_salir.setStyleSheet("""
            QPushButton {
                background: white;
                color: #c0392b;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #e74c3c;
                border-radius: 6px;
                min-width: 180px;
            }
            QPushButton:hover {
                background: #ffe6e6;
                border-color: #c0392b;
            }
            QPushButton:pressed {
                background: #fadbd8;
            }
        """)
        btn_salir.clicked.connect(QApplication.instance().quit)
        layout_botones.addWidget(btn_salir)
        
        layout.addWidget(frame_botones)
        
        # Cambiar a la vista de resultados
        self.stack.setCurrentIndex(1)
    
    def _obtener_categoria_imc(self, imc):
        if imc < 18.5:
            return "Bajo peso"
        elif imc < 25:
            return "Peso normal"
        elif imc < 30:
            return "Sobrepeso"
        else:
            return "Obesidad"
    
    def _generar_conclusion(self, prediccion, probabilidad, datos, imc):
        """Genera una conclusión clara basada en el resultado del modelo"""
        umbral = self.modelo_info['umbral_optimo']
        
        conclusion_html = "<div style='background-color: #fff9e6; padding: 15px; border-left: 4px solid #f39c12; margin: 10px 0; border-radius: 3px;'>"
        conclusion_html += "<h3 style='margin-top: 0; color: #e67e22;'>Conclusión del Análisis</h3>"
        
        if prediccion == 1:
            # Contar factores de riesgo
            factores_riesgo = []
            if imc >= 30:
                factores_riesgo.append("obesidad")
            elif imc >= 25:
                factores_riesgo.append("sobrepeso")
            
            if datos['rango_edad'] >= 9:
                factores_riesgo.append("edad avanzada")
            
            if datos['actividad_fisica_reciente'] == 0:
                factores_riesgo.append("sedentarismo")
            
            if datos['fumador_historico'] == 1:
                factores_riesgo.append("tabaquismo")
            
            if datos['salud_general'] >= 4:
                factores_riesgo.append("salud general deteriorada")
            
            if datos['dias_mala_salud_fisica'] >= 15:
                factores_riesgo.append("problemas de salud física frecuentes")
            
            conclusion_html += f"""
            <p style='font-size: 13px; line-height: 1.6;'>
            El modelo de inteligencia artificial ha identificado un <b style='color: #e74c3c;'>riesgo elevado 
            de diabetes</b> con una probabilidad del <b>{probabilidad*100:.2f}%</b>, que supera el umbral 
            de decisión del {umbral*100:.2f}%.
            </p>
            """
            
            if len(factores_riesgo) > 0:
                conclusion_html += f"""
                <p style='font-size: 13px; line-height: 1.6;'>
                Se han detectado <b>{len(factores_riesgo)} factores de riesgo principales</b>: 
                {', '.join(factores_riesgo)}. La combinación de estos factores aumenta significativamente 
                la probabilidad de desarrollar diabetes tipo 2.
                </p>
                """
            
            conclusion_html += """
            <p style='font-size: 13px; line-height: 1.6;'>
            <b style='color: #c0392b;'>Es importante que consulte con un profesional de la salud 
            lo antes posible</b> para realizar exámenes específicos de glucosa en sangre y recibir 
            orientación médica personalizada.
            </p>
            """
        else:
            # Contar factores protectores
            factores_protectores = []
            if imc >= 18.5 and imc < 25:
                factores_protectores.append("peso saludable")
            
            if datos['actividad_fisica_reciente'] == 1:
                factores_protectores.append("actividad física regular")
            
            if datos['consumo_frutas'] == 1 and datos['consumo_verduras'] == 1:
                factores_protectores.append("dieta saludable")
            
            if datos['fumador_historico'] == 0:
                factores_protectores.append("no fumador")
            
            if datos['salud_general'] <= 2:
                factores_protectores.append("excelente salud general")
            
            conclusion_html += f"""
            <p style='font-size: 13px; line-height: 1.6;'>
            El modelo de inteligencia artificial ha determinado un <b style='color: #27ae60;'>riesgo bajo 
            de diabetes</b> con una probabilidad del <b>{probabilidad*100:.2f}%</b>, que está por debajo 
            del umbral de decisión del {umbral*100:.2f}%.
            </p>
            """
            
            if len(factores_protectores) > 0:
                conclusion_html += f"""
                <p style='font-size: 13px; line-height: 1.6;'>
                Se han identificado <b>{len(factores_protectores)} factores protectores</b>: 
                {', '.join(factores_protectores)}. Estos hábitos y características reducen significativamente 
                el riesgo de desarrollar diabetes.
                </p>
                """
            
            # Identificar áreas de mejora si las hay
            areas_mejora = []
            if imc >= 25:
                areas_mejora.append("control de peso")
            if datos['actividad_fisica_reciente'] == 0:
                areas_mejora.append("actividad física")
            if datos['consumo_frutas'] == 0 or datos['consumo_verduras'] == 0:
                areas_mejora.append("alimentación")
            
            if len(areas_mejora) > 0:
                conclusion_html += f"""
                <p style='font-size: 13px; line-height: 1.6;'>
                Aunque su riesgo es bajo, podría <b>optimizar aún más su salud</b> mejorando en: 
                {', '.join(areas_mejora)}. Esto ayudará a mantener el riesgo bajo a largo plazo.
                </p>
                """
            else:
                conclusion_html += """
                <p style='font-size: 13px; line-height: 1.6;'>
                <b style='color: #27ae60;'>¡Felicitaciones!</b> Sus hábitos de vida son excelentes. 
                Continúe con estos buenos hábitos y realice chequeos médicos preventivos regularmente.
                </p>
                """
        
        conclusion_html += "</div>"
        return conclusion_html
    
    def _generar_tabla_datos_horizontal(self, datos, imc):
        """Genera una tabla HTML horizontal compacta con los valores del modelo"""
        tabla_html = """
        <div style='background-color: #f9f9f9; padding: 8px; border-radius: 5px; font-size: 10px;'>
        <table style='width: 100%; border-collapse: collapse;'>
        """
        
        # Fila 1: IMC, Edad, Sexo, Actividad
        tabla_html += f"""
        <tr>
            <td style='padding: 3px; font-weight: bold;'>IMC:</td>
            <td style='padding: 3px;'>{imc:.1f}</td>
            <td style='padding: 3px; font-weight: bold;'>Edad:</td>
            <td style='padding: 3px;'>{datos['rango_edad']}</td>
            <td style='padding: 3px; font-weight: bold;'>Sexo:</td>
            <td style='padding: 3px;'>{datos['sexo']}</td>
        </tr>
        <tr>
            <td style='padding: 3px; font-weight: bold;'>Act.Física:</td>
            <td style='padding: 3px;'>{datos['actividad_fisica_reciente']}</td>
            <td style='padding: 3px; font-weight: bold;'>Frutas:</td>
            <td style='padding: 3px;'>{datos['consumo_frutas']}</td>
            <td style='padding: 3px; font-weight: bold;'>Verduras:</td>
            <td style='padding: 3px;'>{datos['consumo_verduras']}</td>
        </tr>
        <tr>
            <td style='padding: 3px; font-weight: bold;'>Fumador:</td>
            <td style='padding: 3px;'>{datos['fumador_historico']}</td>
            <td style='padding: 3px; font-weight: bold;'>Alcohol:</td>
            <td style='padding: 3px;'>{datos['consumo_alcohol_elevado']}</td>
            <td style='padding: 3px; font-weight: bold;'>Salud Gral:</td>
            <td style='padding: 3px;'>{datos['salud_general']}</td>
        </tr>
        <tr>
            <td style='padding: 3px; font-weight: bold;'>Días S.Física:</td>
            <td style='padding: 3px;'>{datos['dias_mala_salud_fisica']}</td>
            <td style='padding: 3px; font-weight: bold;'>Días S.Mental:</td>
            <td style='padding: 3px;'>{datos['dias_mala_salud_mental']}</td>
            <td style='padding: 3px; font-weight: bold;'>Dif.Caminar:</td>
            <td style='padding: 3px;'>{datos['dificultad_caminar']}</td>
        </tr>
        </table>
        <p style='font-size: 9px; color: #666; margin: 5px 0 0 0; font-style: italic;'>
        Valores binarios: 0=No, 1=Sí | Salud General: 1=Excelente, 5=Mala
        </p>
        </div>
        """
        return tabla_html
    
    def _generar_conclusion_mejorada(self, prediccion, probabilidad, datos, imc):
        """Genera conclusión con diseño mejorado"""
        umbral = self.modelo_info['umbral_optimo']
        
        html = "<div style='padding: 10px; line-height: 1.6;'>"
        
        if prediccion == 1:
            factores_riesgo = []
            if imc >= 30: factores_riesgo.append("Obesidad (IMC ≥30)")
            elif imc >= 25: factores_riesgo.append("Sobrepeso (IMC 25-30)")
            if datos['rango_edad'] >= 9: factores_riesgo.append("Edad avanzada")
            if datos['actividad_fisica_reciente'] == 0: factores_riesgo.append("Sedentarismo")
            if datos['fumador_historico'] == 1: factores_riesgo.append("Tabaquismo")
            if datos['salud_general'] >= 4: factores_riesgo.append("Salud deteriorada")
            if datos['dias_mala_salud_fisica'] >= 15: factores_riesgo.append("Problemas físicos frecuentes")
            
            html += f"""
            <p style='background-color: #ffe6e6; padding: 12px; border-radius: 5px; border-left: 4px solid #e74c3c;'>
            <b style='color: #c0392b; font-size: 14px;'>Situación de Riesgo</b><br>
            <span style='font-size: 12px; color: #555;'>
            El modelo ha detectado un riesgo elevado basándose en la presencia de {len(factores_riesgo)} factores de riesgo significativos.
            </span>
            </p>
            """
            
            if factores_riesgo:
                html += "<p style='font-size: 12px; margin: 10px 0;'><b>Factores identificados:</b></p><ul style='font-size: 11px; margin: 5px 0 10px 15px; line-height: 1.5;'>"
                for factor in factores_riesgo:
                    html += f"<li>{factor}</li>"
                html += "</ul>"
            
            html += f"""
            <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; font-size: 11px; margin-top: 10px;'>
            <b>Acción Inmediata Requerida:</b><br>
            Es fundamental que consulte con un médico para realizar exámenes específicos de glucosa en sangre (glucemia en ayunas y HbA1c) y recibir orientación profesional.
            </div>
            """
        else:
            factores_protectores = []
            if 18.5 <= imc < 25: factores_protectores.append("Peso saludable")
            if datos['actividad_fisica_reciente'] == 1: factores_protectores.append("Ejercicio regular")
            if datos['consumo_frutas'] == 1 and datos['consumo_verduras'] == 1: factores_protectores.append("Dieta equilibrada")
            if datos['fumador_historico'] == 0: factores_protectores.append("No fumador")
            if datos['salud_general'] <= 2: factores_protectores.append("Excelente salud")
            
            html += f"""
            <p style='background-color: #e6ffe6; padding: 12px; border-radius: 5px; border-left: 4px solid #27ae60;'>
            <b style='color: #229954; font-size: 14px;'>Resultado Favorable</b><br>
            <span style='font-size: 12px; color: #555;'>
            El modelo indica un riesgo bajo de diabetes. Se han identificado {len(factores_protectores)} factores protectores en su perfil.
            </span>
            </p>
            """
            
            if factores_protectores:
                html += "<p style='font-size: 12px; margin: 10px 0;'><b>Factores protectores:</b></p><ul style='font-size: 11px; margin: 5px 0 10px 15px; line-height: 1.5;'>"
                for factor in factores_protectores:
                    html += f"<li>{factor}</li>"
                html += "</ul>"
            
            areas_mejora = []
            if imc >= 25: areas_mejora.append("peso")
            if datos['actividad_fisica_reciente'] == 0: areas_mejora.append("actividad física")
            if datos['consumo_frutas'] == 0 or datos['consumo_verduras'] == 0: areas_mejora.append("alimentación")
            
            if areas_mejora:
                html += f"""
                <div style='background-color: #e7f3ff; padding: 10px; border-radius: 5px; font-size: 11px; margin-top: 10px;'>
                <b>Oportunidades de Mejora:</b><br>
                Podría optimizar su salud trabajando en: {', '.join(areas_mejora)}.
                </div>
                """
            else:
                html += """
                <div style='background-color: #e7f3ff; padding: 10px; border-radius: 5px; font-size: 11px; margin-top: 10px;'>
                <b>Excelente perfil de salud.</b> Continúe con sus buenos hábitos y realice chequeos preventivos anuales.
                </div>
                """
        
        html += "</div>"
        return html
    
    def _generar_conclusion_compacta(self, prediccion, probabilidad, datos, imc):
        """Genera una conclusión compacta"""
        umbral = self.modelo_info['umbral_optimo']
        
        if prediccion == 1:
            color = "#e74c3c"
            mensaje = "RIESGO ELEVADO"
        else:
            color = "#27ae60"
            mensaje = "RIESGO BAJO"
        
        conclusion_html = f"""
        <div style='background-color: #fff9e6; padding: 10px; border-left: 4px solid {color}; border-radius: 3px; font-size: 11px;'>
        <p style='margin: 5px 0; line-height: 1.4;'>
        <b style='color: {color};'>{mensaje}</b><br>
        Probabilidad: <b>{probabilidad*100:.2f}%</b> (Umbral: {umbral*100:.2f}%)
        </p>
        """
        
        if prediccion == 1:
            factores_riesgo = []
            if imc >= 30: factores_riesgo.append("obesidad")
            elif imc >= 25: factores_riesgo.append("sobrepeso")
            if datos['rango_edad'] >= 9: factores_riesgo.append("edad")
            if datos['actividad_fisica_reciente'] == 0: factores_riesgo.append("sedentarismo")
            if datos['fumador_historico'] == 1: factores_riesgo.append("tabaco")
            if datos['salud_general'] >= 4: factores_riesgo.append("salud deteriorada")
            
            conclusion_html += f"""
            <p style='margin: 5px 0; line-height: 1.4; font-size: 10px;'>
            <b>Factores detectados:</b> {', '.join(factores_riesgo) if factores_riesgo else 'Varios'}<br>
            <b style='color: #c0392b;'>Acción requerida:</b> Consulta médica urgente para exámenes de glucosa.
            </p>
            """
        else:
            factores_protectores = 0
            if 18.5 <= imc < 25: factores_protectores += 1
            if datos['actividad_fisica_reciente'] == 1: factores_protectores += 1
            if datos['consumo_frutas'] == 1 and datos['consumo_verduras'] == 1: factores_protectores += 1
            if datos['fumador_historico'] == 0: factores_protectores += 1
            
            conclusion_html += f"""
            <p style='margin: 5px 0; line-height: 1.4; font-size: 10px;'>
            <b>Factores protectores:</b> {factores_protectores} de 4<br>
            <b style='color: #229954;'>Continúe</b> con sus hábitos saludables y realice chequeos preventivos anuales.
            </p>
            """
        
        conclusion_html += "</div>"
        return conclusion_html
    
    def _generar_tabla_datos_mejorada(self, datos, imc):
        """Tabla de datos compacta y clara"""
        html = "<div style='padding: 8px; font-size: 11px;'>"
        html += "<table style='width: 100%; border-collapse: collapse;'>"
        
        items = [
            ("IMC", f"{imc:.1f}"),
            ("Edad (rango)", f"{datos['rango_edad']}"),
            ("Sexo", f"{datos['sexo']}"),
            ("Actividad Física", f"{datos['actividad_fisica_reciente']}"),
            ("Frutas diarias", f"{datos['consumo_frutas']}"),
            ("Verduras diarias", f"{datos['consumo_verduras']}"),
            ("Fumador", f"{datos['fumador_historico']}"),
            ("Alcohol elevado", f"{datos['consumo_alcohol_elevado']}"),
            ("Salud general", f"{datos['salud_general']}"),
            ("Días salud física", f"{datos['dias_mala_salud_fisica']}"),
            ("Días salud mental", f"{datos['dias_mala_salud_mental']}"),
            ("Dificultad caminar", f"{datos['dificultad_caminar']}")
        ]
        
        for i, (nombre, valor) in enumerate(items):
            bg = "#f8f9fa" if i % 2 == 0 else "white"
            html += f"""
            <tr style='background-color: {bg};'>
                <td style='padding: 4px 8px; font-weight: bold; width: 60%;'>{nombre}</td>
                <td style='padding: 4px 8px; text-align: center; width: 40%;'>{valor}</td>
            </tr>
            """
        
        html += "</table>"
        html += "<p style='font-size: 9px; color: #6c757d; margin-top: 8px; font-style: italic;'>Binarios: 0=No, 1=Sí | Salud: 1=Excelente, 5=Mala</p>"
        html += "</div>"
        return html
    
    def _generar_analisis_visual(self, datos, imc, prediccion):
        """Análisis visual mejorado con barras de progreso"""
        html = "<div style='padding: 8px; font-size: 12px;'>"
        
        parametros = [
            ('IMC', imc, lambda v: (min(100, int((v/40)*100)), 'ALTO' if v >= 30 else 'MEDIO' if v >= 25 else 'BAJO', f"{v:.1f}")),
            ('Edad', datos['rango_edad'], lambda v: (min(100, int((v/13)*100)), 'ALTO' if v >= 9 else 'MEDIO' if v >= 5 else 'BAJO', f"Nivel {v}")),
            ('Actividad Física', datos['actividad_fisica_reciente'], lambda v: (100 if v == 1 else 0, 'BAJO' if v == 1 else 'ALTO', "Sí" if v == 1 else "No")),
            ('Alimentación', (datos['consumo_frutas'] + datos['consumo_verduras'])/2, lambda v: (int(v*100), 'BAJO' if v == 1 else 'MEDIO' if v == 0.5 else 'ALTO', "Buena" if v == 1 else "Regular" if v == 0.5 else "Mejorable")),
            ('Tabaquismo', datos['fumador_historico'], lambda v: (100 if v == 1 else 0, 'ALTO' if v == 1 else 'BAJO', "Sí" if v == 1 else "No")),
            ('Alcohol', datos['consumo_alcohol_elevado'], lambda v: (100 if v == 1 else 0, 'ALTO' if v == 1 else 'BAJO', "Elevado" if v == 1 else "Normal")),
            ('Salud General', datos['salud_general'], lambda v: (int((v/5)*100), 'ALTO' if v >= 4 else 'MEDIO' if v == 3 else 'BAJO', f"Nivel {v}/5")),
            ('Salud Física', datos['dias_mala_salud_fisica'], lambda v: (min(100, int((v/30)*100)), 'ALTO' if v >= 15 else 'MEDIO' if v >= 8 else 'BAJO', f"{v} días")),
            ('Salud Mental', datos['dias_mala_salud_mental'], lambda v: (min(100, int((v/30)*100)), 'ALTO' if v >= 15 else 'MEDIO' if v >= 8 else 'BAJO', f"{v} días")),
            ('Movilidad', datos['dificultad_caminar'], lambda v: (100 if v == 1 else 0, 'ALTO' if v == 1 else 'BAJO', "Con dificultad" if v == 1 else "Normal"))
        ]
        
        for nombre, valor, evaluador in parametros:
            progreso, nivel_riesgo, texto = evaluador(valor)
            color_barra = "#dc3545" if nivel_riesgo == 'ALTO' else "#ffc107" if nivel_riesgo == 'MEDIO' else "#28a745"
            
            html += f"""
            <div style='margin-bottom: 12px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 3px;'>
                    <span style='font-weight: bold; font-size: 11px;'>{nombre}</span>
                    <span style='font-size: 10px; color: #6c757d;'>{texto}</span>
                </div>
                <div style='background-color: #e9ecef; height: 8px; border-radius: 4px; overflow: hidden;'>
                    <div style='background-color: {color_barra}; height: 100%; width: {progreso}%; transition: width 0.3s;'></div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generar_recomendaciones_mejoradas(self, datos, imc, prediccion):
        """Recomendaciones mejoradas con prioridades"""
        html = "<div style='padding: 8px; font-size: 12px;'>"
        
        if prediccion == 1:
            html += "<div style='background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; margin-bottom: 12px; border-radius: 3px;'>"
            html += "<b style='color: #721c24;'>PRIORIDAD ALTA</b>"
            html += "</div>"
            
            recomendaciones = [
                ("Consulta Médica", "Programe cita con médico para exámenes de glucosa (ayunas y HbA1c) lo antes posible.", "alta"),
            ]
            
            if imc >= 25:
                recomendaciones.append(("Control de Peso", "Consulte nutricionista. Objetivo: reducir 5-10% del peso actual en 6 meses.", "alta"))
            
            if datos['actividad_fisica_reciente'] == 0:
                recomendaciones.append(("Ejercicio", "Inicie con 30 min de caminata diaria. Meta: 150 min/semana de actividad moderada.", "alta"))
            
            if datos['consumo_frutas'] == 0 or datos['consumo_verduras'] == 0:
                recomendaciones.append(("Alimentación", "Incluya 5 porciones/día de frutas y verduras. Reduzca azúcares y carbohidratos refinados.", "media"))
            
            if datos['fumador_historico'] == 1:
                recomendaciones.append(("Cesación Tabáquica", "Busque programa de apoyo para dejar de fumar. Fundamental para reducir riesgo.", "alta"))
            
            if datos['consumo_alcohol_elevado'] == 1:
                recomendaciones.append(("Alcohol", "Reduzca consumo a niveles moderados o elimine completamente.", "media"))
            
            if datos['dias_mala_salud_mental'] >= 15:
                recomendaciones.append(("Salud Mental", "Considere apoyo psicológico. El estrés crónico afecta el metabolismo de glucosa.", "media"))
            
            recomendaciones.append(("Monitoreo", "Chequeos cada 3-6 meses: glucosa, presión arterial, perfil lipídico.", "alta"))
            
        else:
            html += "<div style='background-color: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin-bottom: 12px; border-radius: 3px;'>"
            html += "<b style='color: #155724;'>MANTENIMIENTO PREVENTIVO</b>"
            html += "</div>"
            
            recomendaciones = [
                ("Chequeos Preventivos", "Realice controles médicos anuales para mantener su salud óptima.", "baja"),
            ]
            
            if imc >= 25:
                recomendaciones.append(("Peso Saludable", "Aunque su riesgo es bajo, mantener IMC 18.5-24.9 es óptimo.", "media"))
            
            if datos['actividad_fisica_reciente'] == 0:
                recomendaciones.append(("Actividad Física", "Incorpore 150 min/semana de ejercicio para prevención a largo plazo.", "media"))
            
            if datos['consumo_frutas'] == 0 or datos['consumo_verduras'] == 0:
                recomendaciones.append(("Nutrición Óptima", "Aumente consumo de frutas y verduras para maximizar beneficios.", "baja"))
            
            recomendaciones.append(("Estilo de Vida", "Continúe con sus buenos hábitos. Son clave para prevención.", "baja"))
        
        for titulo, desc, prioridad in recomendaciones:
            color = "#dc3545" if prioridad == "alta" else "#ffc107" if prioridad == "media" else "#28a745"
            
            html += f"""
            <div style='margin-bottom: 12px; padding: 10px; background-color: #f8f9fa; border-left: 3px solid {color}; border-radius: 3px;'>
                <div style='font-weight: bold; font-size: 12px; margin-bottom: 4px;'>{titulo}</div>
                <div style='font-size: 11px; color: #495057; line-height: 1.4;'>{desc}</div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generar_analisis_compacto(self, datos, imc, prediccion):
        """Genera un análisis compacto de parámetros"""
        analisis = "<div style='background-color: white; padding: 5px; font-size: 10px;'>"
        
        parametros = [
            ('IMC', imc, lambda v: ('🔴' if v >= 30 else '🟡' if v >= 25 else '🟢', f"{v:.1f}")),
            ('Edad', datos['rango_edad'], lambda v: ('🔴' if v >= 9 else '🟡' if v >= 5 else '🟢', f"Rango {v}")),
            ('Act.Física', datos['actividad_fisica_reciente'], lambda v: ('🟢' if v == 1 else '🔴', "Sí" if v == 1 else "No")),
            ('Frutas', datos['consumo_frutas'], lambda v: ('🟢' if v == 1 else '🟡', "Sí" if v == 1 else "No")),
            ('Verduras', datos['consumo_verduras'], lambda v: ('🟢' if v == 1 else '🟡', "Sí" if v == 1 else "No")),
            ('Fumador', datos['fumador_historico'], lambda v: ('🔴' if v == 1 else '🟢', "Sí" if v == 1 else "No")),
            ('Alcohol', datos['consumo_alcohol_elevado'], lambda v: ('🔴' if v == 1 else '🟢', "Sí" if v == 1 else "No")),
            ('Salud Gral', datos['salud_general'], lambda v: ('🔴' if v >= 4 else '🟡' if v == 3 else '🟢', f"Nivel {v}")),
            ('Salud Física', datos['dias_mala_salud_fisica'], lambda v: ('🔴' if v >= 15 else '🟡' if v >= 8 else '🟢', f"{v} días")),
            ('Salud Mental', datos['dias_mala_salud_mental'], lambda v: ('🔴' if v >= 15 else '🟡' if v >= 8 else '🟢', f"{v} días")),
            ('Movilidad', datos['dificultad_caminar'], lambda v: ('🔴' if v == 1 else '🟢', "Difícil" if v == 1 else "Normal"))
        ]
        
        for nombre, valor, evaluador in parametros:
            icono, texto = evaluador(valor)
            analisis += f"<div style='padding: 2px; border-bottom: 1px solid #f0f0f0;'><b>{nombre}:</b> {icono} {texto}</div>"
        
        analisis += "<p style='font-size: 9px; color: #666; margin: 5px 0 0 0;'>🟢 Saludable | 🟡 Atención | 🔴 Riesgo</p>"
        analisis += "</div>"
        return analisis
    
    def _generar_analisis_parametros(self, datos, imc, prediccion):
        analisis = "<div style='background-color: white; padding: 10px; border-radius: 5px;'>"
        analisis += "<table style='width: 100%; border-collapse: collapse;'>"
        
        # IMC
        if imc >= 30:
            color = "#e74c3c"
            estado = "ALTO RIESGO"
            icono = "⚠"
            mensaje = "Su IMC está en rango de obesidad, un factor de riesgo importante para diabetes."
        elif imc >= 25:
            color = "#f39c12"
            estado = "RIESGO MODERADO"
            icono = "⚠"
            mensaje = "Su IMC indica sobrepeso, lo cual aumenta el riesgo de diabetes."
        elif imc >= 18.5:
            color = "#27ae60"
            estado = "SALUDABLE"
            icono = "✓"
            mensaje = "Su IMC está en rango saludable."
        else:
            color = "#f39c12"
            estado = "ATENCIÓN"
            icono = "⚠"
            mensaje = "Su IMC está por debajo del rango normal."
        
        analisis += f"""
        <tr style='border-bottom: 1px solid #e0e0e0;'>
            <td style='padding: 8px; font-weight: bold; width: 30%;'>IMC ({imc:.1f})</td>
            <td style='padding: 8px; color: {color}; font-weight: bold;'>{icono} {estado}</td>
        </tr>
        <tr style='border-bottom: 2px solid #e0e0e0;'>
            <td colspan='2' style='padding: 8px; font-size: 12px; color: #555;'>{mensaje}</td>
        </tr>
        """
        
        # Resto de parámetros con el mismo formato
        parametros = [
            ('rango_edad', datos['rango_edad'], 'Edad', 
             lambda v: ('ALTO RIESGO', '#e74c3c', '⚠', 'A mayor edad, mayor riesgo de diabetes tipo 2.') if v >= 9
             else ('RIESGO MODERADO', '#f39c12', '⚠', 'Su edad tiene riesgo moderado.') if v >= 5
             else ('BAJO RIESGO', '#27ae60', '✓', 'Su grupo de edad tiene menor riesgo.')),
            
            ('actividad_fisica_reciente', datos['actividad_fisica_reciente'], 'Actividad Física',
             lambda v: ('SALUDABLE', '#27ae60', '✓', 'El ejercicio regular ayuda a prevenir la diabetes.') if v == 1
             else ('ALTO RIESGO', '#e74c3c', '⚠', 'La falta de ejercicio aumenta significativamente el riesgo.')),
            
            ('consumo_frutas', datos['consumo_frutas'], 'Consumo de Frutas',
             lambda v: ('SALUDABLE', '#27ae60', '✓', 'El consumo diario de frutas es beneficioso.') if v == 1
             else ('ATENCIÓN', '#f39c12', '⚠', 'Una dieta sin frutas diarias puede aumentar el riesgo.')),
            
            ('consumo_verduras', datos['consumo_verduras'], 'Consumo de Verduras',
             lambda v: ('SALUDABLE', '#27ae60', '✓', 'Buen hábito que protege contra la diabetes.') if v == 1
             else ('ATENCIÓN', '#f39c12', '⚠', 'Consumir verduras diariamente ayuda a reducir el riesgo.')),
            
            ('fumador_historico', datos['fumador_historico'], 'Tabaquismo',
             lambda v: ('ALTO RIESGO', '#e74c3c', '⚠', 'Fumar aumenta el riesgo de diabetes y complicaciones.') if v == 1
             else ('SALUDABLE', '#27ae60', '✓', 'No fumar reduce significativamente los riesgos.')),
            
            ('consumo_alcohol_elevado', datos['consumo_alcohol_elevado'], 'Consumo de Alcohol',
             lambda v: ('ALTO RIESGO', '#e74c3c', '⚠', 'El consumo frecuente está asociado con mayor riesgo.') if v == 1
             else ('SALUDABLE', '#27ae60', '✓', 'Su consumo no representa un factor de riesgo elevado.')),
            
            ('salud_general', datos['salud_general'], 'Salud General',
             lambda v: ('ALTO RIESGO', '#e74c3c', '⚠', 'Salud percibida como regular/mala indica mayor riesgo.') if v >= 4
             else ('RIESGO MODERADO', '#f39c12', '⚠', 'Salud buena, pero mejorable.') if v == 3
             else ('SALUDABLE', '#27ae60', '✓', 'Excelente percepción de salud.')),
            
            ('dias_mala_salud_fisica', datos['dias_mala_salud_fisica'], 'Días Mala Salud Física',
             lambda v: ('ALTO RIESGO', '#e74c3c', '⚠', 'Muchos días de mala salud física indican problemas importantes.') if v >= 15
             else ('ATENCIÓN', '#f39c12', '⚠', 'Varios días de mala salud física al mes es preocupante.') if v >= 8
             else ('SALUDABLE', '#27ae60', '✓', 'Pocos días de mala salud física.')),
            
            ('dias_mala_salud_mental', datos['dias_mala_salud_mental'], 'Salud Mental',
             lambda v: ('ALTO RIESGO', '#e74c3c', '⚠', 'El estrés crónico puede afectar la salud metabólica.') if v >= 15
             else ('ATENCIÓN', '#f39c12', '⚠', 'El estrés frecuente puede influir en el riesgo.') if v >= 8
             else ('SALUDABLE', '#27ae60', '✓', 'Buen estado de salud mental y emocional.')),
            
            ('dificultad_caminar', datos['dificultad_caminar'], 'Movilidad',
             lambda v: ('ALTO RIESGO', '#e74c3c', '⚠', 'La dificultad para caminar puede indicar problemas subyacentes.') if v == 1
             else ('SALUDABLE', '#27ae60', '✓', 'Buena capacidad de movilidad.'))
        ]
        
        for key, valor, nombre, evaluador in parametros:
            estado, color, icono, mensaje = evaluador(valor)
            analisis += f"""
            <tr style='border-bottom: 1px solid #e0e0e0;'>
                <td style='padding: 8px; font-weight: bold;'>{nombre}</td>
                <td style='padding: 8px; color: {color}; font-weight: bold;'>{icono} {estado}</td>
            </tr>
            <tr style='border-bottom: 2px solid #e0e0e0;'>
                <td colspan='2' style='padding: 8px; font-size: 12px; color: #555;'>{mensaje}</td>
            </tr>
            """
        
        analisis += "</table></div>"
        return analisis
    
    def _generar_recomendaciones(self, datos, imc, prediccion):
        recomendaciones = "<div style='background-color: white; padding: 12px; border-radius: 5px;'>"
        recomendaciones += "<ul style='line-height: 1.8; margin: 5px 0; padding-left: 20px;'>"
        
        if prediccion == 1:
            recomendaciones += "<li style='color: #c0392b; font-weight: bold; margin-bottom: 10px;'>RECOMENDACIONES URGENTES:</li>"
            recomendaciones += "<li><b>Consulte con un médico</b> lo antes posible para realizar pruebas de glucosa y hemoglobina glicosilada (HbA1c).</li>"
            
            if imc >= 25:
                recomendaciones += "<li><b>Control de peso:</b> Su IMC indica sobrepeso/obesidad. Trabaje con un nutricionista para desarrollar un plan alimenticio personalizado.</li>"
            
            if datos['actividad_fisica_reciente'] == 0:
                recomendaciones += "<li><b>Inicie actividad física:</b> Comience con caminatas de 30 minutos diarios y aumente gradualmente hasta 150 minutos semanales.</li>"
            
            if datos['consumo_frutas'] == 0 or datos['consumo_verduras'] == 0:
                recomendaciones += "<li><b>Mejore su dieta:</b> Incluya al menos 5 porciones de frutas y verduras diarias. Reduzca carbohidratos refinados y azúcares.</li>"
            
            if datos['fumador_historico'] == 1:
                recomendaciones += "<li><b>Deje de fumar:</b> Si aún fuma, busque apoyo médico o programas de cesación tabáquica inmediatamente.</li>"
            
            if datos['consumo_alcohol_elevado'] == 1:
                recomendaciones += "<li><b>Reduzca el alcohol:</b> Limite el consumo a cantidades moderadas o elimínelo completamente.</li>"
            
            if datos['dias_mala_salud_mental'] >= 15:
                recomendaciones += "<li><b>Salud mental:</b> Considere apoyo psicológico. El estrés crónico afecta el metabolismo de la glucosa.</li>"
            
            recomendaciones += "<li><b>Monitoreo regular:</b> Realice chequeos de glucosa, presión arterial y perfil lipídico cada 3-6 meses.</li>"
        else:
            recomendaciones += "<li style='color: #229954; font-weight: bold; margin-bottom: 10px;'>RECOMENDACIONES PREVENTIVAS:</li>"
            recomendaciones += "<li><b>Mantenga sus buenos hábitos</b> y realice chequeos médicos preventivos anuales.</li>"
            
            if imc >= 25:
                recomendaciones += "<li><b>Control de peso:</b> Aunque su riesgo es bajo, mantener un IMC entre 18.5-24.9 es óptimo para la salud a largo plazo.</li>"
            
            if datos['actividad_fisica_reciente'] == 0:
                recomendaciones += "<li><b>Considere hacer ejercicio:</b> La actividad física regular (150 min/semana) previene diabetes y otras enfermedades crónicas.</li>"
            
            if datos['consumo_frutas'] == 0 or datos['consumo_verduras'] == 0:
                recomendaciones += "<li><b>Optimice su nutrición:</b> Intente incluir más frutas y verduras variadas para obtener todos los nutrientes esenciales.</li>"
            
            if datos['fumador_historico'] == 1:
                recomendaciones += "<li><b>No fume:</b> Si dejó de fumar, ¡excelente! Si aún fuma, considere dejarlo para reducir riesgos futuros.</li>"
            
            recomendaciones += "<li><b>Prevención continua:</b> Continúe con un estilo de vida saludable para mantener su riesgo bajo a largo plazo.</li>"
        
        recomendaciones += "</ul></div>"
        return recomendaciones

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ventana = EvaluadorRiesgoDiabetes()
    ventana.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

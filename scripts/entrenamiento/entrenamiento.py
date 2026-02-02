#!/usr/bin/env python3

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass, asdict
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, roc_curve

@dataclass
class ConfiguracionModelo:
    iteraciones: int = 150
    tasa_aprendizaje: float = 0.07
    profundidad: int = 6
    reg_l2_hoja: int = 3
    semilla_aleatoria: int = 42
    pesos_clases: Dict[int, int] = None
    
    def __post_init__(self):
        if self.pesos_clases is None:
            self.pesos_clases = {0: 1, 1: 7}

class DetectorRiesgoDiabetes:
    def __init__(self, ruta_base: Path, config: ConfiguracionModelo = None):
        self.ruta_base = ruta_base
        self.config = config if config else ConfiguracionModelo()
        self.modelo = None
        self.n_trabajos = max(1, cpu_count() - 1)
        self.dir_salida = self.ruta_base / "resultados"
        self.dir_salida.mkdir(parents=True, exist_ok=True)
        np.random.seed(self.config.semilla_aleatoria)

    def cargar_datos(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        df_entrenamiento = pd.read_csv(self.ruta_base / "dataset" / "train.csv")
        df_prueba = pd.read_csv(self.ruta_base / "dataset" / "test.csv")
        
        return (
            df_entrenamiento.drop('estado_diabetes', axis=1),
            df_entrenamiento['estado_diabetes'],
            df_prueba.drop('estado_diabetes', axis=1),
            df_prueba['estado_diabetes']
        )

    def _calcular_metricas(self, y_verdadero: np.ndarray, y_predicho: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        vn, fp, fn, vp = confusion_matrix(y_verdadero, y_predicho).ravel()
        return {
            'sensibilidad': recall_score(y_verdadero, y_predicho),
            'roc_auc': roc_auc_score(y_verdadero, y_proba),
            'puntaje_balance': (recall_score(y_verdadero, y_predicho) + roc_auc_score(y_verdadero, y_proba)) / 2,
            'vp': int(vp), 'vn': int(vn), 'fp': int(fp), 'fn': int(fn)
        }

    def optimizar_umbral(self, y_verdadero: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
        umbrales = np.linspace(0.001, 0.5, 1000)
        
        def calcular_youden(umbral):
            y_predicho = (y_proba >= umbral).astype(int)
            vn, fp, fn, vp = confusion_matrix(y_verdadero, y_predicho).ravel()
            sensibilidad = vp / (vp + fn) if (vp + fn) > 0 else 0
            especificidad = vn / (vn + fp) if (vn + fp) > 0 else 0
            return 0.9 * sensibilidad + 0.1 * especificidad

        mejor_umbral = max(umbrales, key=calcular_youden)
        return mejor_umbral, roc_auc_score(y_verdadero, y_proba)

    def entrenar(self, X_entrenamiento, y_entrenamiento, X_prueba, y_prueba):
        self.modelo = CatBoostClassifier(
            iterations=self.config.iteraciones,
            learning_rate=self.config.tasa_aprendizaje,
            depth=self.config.profundidad,
            l2_leaf_reg=self.config.reg_l2_hoja,
            class_weights=self.config.pesos_clases,
            random_seed=self.config.semilla_aleatoria,
            verbose=False,
            thread_count=self.n_trabajos,
            task_type='CPU',
            bootstrap_type='Bernoulli',
            subsample=0.85,
            colsample_bylevel=0.9,
            min_data_in_leaf=20,
            eval_metric='AUC',
            custom_metric=['Recall'],
            allow_writing_files=False
        )
        
        self.modelo.fit(
            X_entrenamiento, y_entrenamiento,
            eval_set=[(X_entrenamiento, y_entrenamiento), (X_prueba, y_prueba)],
            verbose=10
        )

    def generar_graficos(self, historial_entrenamiento: pd.DataFrame, historial_prueba: pd.DataFrame, mejor_iteracion: int, metricas_prueba: Dict, y_prueba, y_proba, umbral_optimo):
        config_graficos = [
            (historial_entrenamiento, 'roc_auc', 'Entrenamiento', '#2E86AB', 'ROC AUC Evaluado en Conjunto de Entrenamiento por Iteración', 'ROC AUC (%)', 'evolucion_entrenamiento.png'),
            (historial_prueba, 'roc_auc', 'Prueba', '#A23B72', 'ROC AUC Evaluado en Conjunto de Prueba por Iteración', 'ROC AUC (%)', 'evolucion_prueba.png')
        ]

        for df, col, etiqueta, color, titulo, etiqueta_y, nombre_archivo in config_graficos:
            plt.figure(figsize=(12, 7))
            plt.plot(df['iteracion'], df[col], linewidth=2.5, color=color, alpha=0.9, label=etiqueta, marker='o', markevery=20, markersize=6)
            plt.scatter([0], [0.0], s=100, color='red', zorder=5, edgecolors='darkred', linewidths=2)
            plt.annotate('Sin entrenamiento\n(0%)', xy=(0, 0), xytext=(15, 5), fontsize=10, color='darkred', fontweight='bold', arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
            
            for i in range(0, len(df), 20):
                if i == 0: continue
                val = df[col].iloc[i]
                plt.annotate(f'{val:.1f}%', xy=(i, val), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
            
            if (len(df)-1) % 20 != 0:
                ultima_iter = len(df) - 1
                ultimo_val = df[col].iloc[-1]
                plt.annotate(f'{ultimo_val:.1f}%', xy=(ultima_iter, ultimo_val), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

            if mejor_iteracion < len(df) - 1:
                plt.axvline(x=mejor_iteracion, color='green', linestyle='--', alpha=0.7, label=f'Mejor iteración ({mejor_iteracion})')
            
            plt.xlabel('Iteración del Entrenamiento', fontsize=13, fontweight='bold')
            plt.ylabel(etiqueta_y, fontsize=13, fontweight='bold')
            plt.title(titulo, fontsize=15, fontweight='bold', pad=15)
            plt.ylim([0, 100])
            plt.xlim([0, df['iteracion'].max()])
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(self.dir_salida / nombre_archivo, dpi=300, bbox_inches='tight')
            plt.close()

        self._graficar_matriz_confusion(metricas_prueba)
        self._graficar_curva_roc(y_prueba, y_proba, umbral_optimo, metricas_prueba)

    def _graficar_matriz_confusion(self, metricas: Dict):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Matriz organizada: [[VP, FN], [FP, VN]]
        cm = np.array([[metricas['vp'], metricas['fn']], 
                       [metricas['fp'], metricas['vn']]])
        
        colores = [['#1E3A8A', '#F97316'], ['#F97316', '#1E3A8A']]
        etiquetas_texto = [['Verdadero Positivo', 'Falso Negativo'], 
                       ['Falso Positivo', 'Verdadero Negativo']]
        
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colores[i][j], alpha=0.9, edgecolor='white', linewidth=2))
                
                val = cm[i, j]
                porcentaje = (val / cm.sum()) * 100
                
                ax.text(j + 0.5, i + 0.25, etiquetas_texto[i][j], ha='center', va='center', color='white', fontsize=12, fontweight='bold')
                ax.text(j + 0.5, i + 0.5, f"{val:,}", ha='center', va='center', color='white', fontsize=28, fontweight='bold')
                ax.text(j + 0.5, i + 0.75, f"{porcentaje:.1f}%", ha='center', va='center', color='white', fontsize=12)

        ax.set_xlim(0, 2)
        ax.set_ylim(2, 0)
        
        nombres_clases = ['Diabético', 'Sano']
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(nombres_clases, fontsize=12, fontweight='bold')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(nombres_clases, fontsize=12, fontweight='bold', rotation=90, va='center')
        
        ax.set_xlabel('Predicción del Modelo', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Estado Real', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='both', which='both', length=0)
        
        plt.tight_layout()
        plt.savefig(self.dir_salida / "matriz_confusion.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _graficar_curva_roc(self, y_verdadero, y_proba, umbral, metricas):
        plt.figure(figsize=(10, 8))
        fpr, tpr, umbrales = roc_curve(y_verdadero, y_proba)
        idx = np.argmin(np.abs(umbrales - umbral))
        
        plt.plot(fpr, tpr, linewidth=3, color='#2E86AB', label=f'Modelo CatBoost (AUC = {metricas["roc_auc"]:.3f})', alpha=0.9)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Clasificador aleatorio (AUC = 0.500)', alpha=0.5)
        plt.scatter([fpr[idx]], [tpr[idx]], s=300, color='red', zorder=5, edgecolors='darkred', linewidths=2, label=f'Umbral óptimo ({umbral:.4f})')
        
        plt.xlabel('Tasa de Falsos Positivos', fontsize=13, fontweight='bold')
        plt.ylabel('Sensibilidad', fontsize=13, fontweight='bold')
        plt.title(f'Curva ROC - Conjunto de Prueba\nSensibilidad: {metricas["sensibilidad"]*100:.2f}%', fontsize=15, fontweight='bold', pad=15)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(self.dir_salida / "curva_roc_prueba.png", dpi=300, bbox_inches='tight')
        plt.close()

    def ejecutar(self):
        print(f"\n{'='*80}")
        print(f"{'MODELO DE DETECCIÓN DE RIESGO DE DIABETES':^80}")
        print(f"{'='*80}\n")
        
        print(f"[*] Cargando conjuntos de datos...")
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba = self.cargar_datos()
        print(f"    > Entrenamiento: {X_entrenamiento.shape[0]:,} muestras | {X_entrenamiento.shape[1]} características")
        print(f"    > Prueba:        {X_prueba.shape[0]:,} muestras | {X_prueba.shape[1]} características")
        print(f"    > Distribución:  {y_entrenamiento.mean()*100:.1f}% positivos en entrenamiento")

        print(f"\n[*] Iniciando entrenamiento del modelo CatBoost...")
        self.entrenar(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)
        
        print(f"\n[*] Optimizando umbral de decisión...")
        y_proba_prueba = self.modelo.predict_proba(X_prueba)[:, 1]
        umbral_optimo, _ = self.optimizar_umbral(y_prueba, y_proba_prueba)
        print(f"    > Umbral óptimo encontrado: {umbral_optimo:.4f}")
        
        y_pred_prueba = (y_proba_prueba >= umbral_optimo).astype(int)
        metricas_prueba = self._calcular_metricas(y_prueba, y_pred_prueba, y_proba_prueba)
        
        y_proba_entrenamiento = self.modelo.predict_proba(X_entrenamiento)[:, 1]
        y_pred_entrenamiento = (y_proba_entrenamiento >= umbral_optimo).astype(int)
        metricas_entrenamiento = self._calcular_metricas(y_entrenamiento, y_pred_entrenamiento, y_proba_entrenamiento)
        
        print(f"\n{'='*80}")
        print(f"{'RESULTADOS FINALES':^80}")
        print(f"{'='*80}")
        print(f"{'Métrica':<35} | {'Entrenamiento':<15} | {'Prueba (Validación)':<15}")
        print(f"{'-'*35}-+-{'-'*15}-+-{'-'*15}")
        print(f"{'Sensibilidad (Recall)':<35} | {metricas_entrenamiento['sensibilidad']*100:6.2f}%         | {metricas_prueba['sensibilidad']*100:6.2f}%")
        print(f"{'ROC AUC (ROC AUC)':<35} | {metricas_entrenamiento['roc_auc']*100:6.2f}%         | {metricas_prueba['roc_auc']*100:6.2f}%")
        print(f"{'Puntaje Balance (Balance Score)':<35} | {metricas_entrenamiento['puntaje_balance']*100:6.2f}%         | {metricas_prueba['puntaje_balance']*100:6.2f}%")

        print(f"\n[*] Guardando artefactos del modelo...")
        joblib.dump({
            'modelo': self.modelo,
            'nombres_caracteristicas': list(X_entrenamiento.columns),
            'umbral_optimo': umbral_optimo,
            'metricas': metricas_prueba
        }, self.dir_salida / "modelo.pkl")
            
        evals = self.modelo.get_evals_result()
        recall_key = next((k for k in evals['validation_0'].keys() if 'Recall' in k), None)
        
        auc_entrenamiento = [x * 100 for x in evals['validation_0']['AUC']]
        auc_prueba = [x * 100 for x in evals['validation_1']['AUC']]
        recall_entrenamiento = [x * 100 for x in evals['validation_0'][recall_key]] if recall_key else [0.0] * len(auc_entrenamiento)
        recall_prueba = [x * 100 for x in evals['validation_1'][recall_key]] if recall_key else [0.0] * len(auc_prueba)

        iteraciones = range(len(auc_entrenamiento) + 1)
        historial_entrenamiento = pd.DataFrame({
            'iteracion': iteraciones,
            'roc_auc': [0.0] + auc_entrenamiento,
            'sensibilidad': [0.0] + recall_entrenamiento
        })
        
        historial_prueba = pd.DataFrame({
            'iteracion': iteraciones,
            'roc_auc': [0.0] + auc_prueba,
            'sensibilidad': [0.0] + recall_prueba
        })
        
        historial_entrenamiento.to_csv(self.dir_salida / "historial_entrenamiento.csv", index=False)
        historial_prueba.to_csv(self.dir_salida / "historial_prueba.csv", index=False)
        
        print(f"[*] Generando visualizaciones...")
        self.generar_graficos(historial_entrenamiento, historial_prueba, self.modelo.get_best_iteration(), metricas_prueba, y_prueba, y_proba_prueba, umbral_optimo)
        
        print(f"\n[OK] Proceso completado exitosamente.")
        print(f"     Resultados guardados en: {self.dir_salida}\n")
        
        return metricas_prueba

if __name__ == "__main__":
    DetectorRiesgoDiabetes(Path(__file__).parent.parent.parent).ejecutar()
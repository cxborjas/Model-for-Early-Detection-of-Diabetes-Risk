import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Configuración de rutas (Ajusta si tu carpeta 'resultados' está en otro lado)
BASE_DIR = Path(__file__).parent.parent.parent # Misma lógica que tu script original
RUTA_MODELO = BASE_DIR / "resultados" / "modelo.pkl"
RUTA_SALIDA = BASE_DIR / "resultados" / "Figura6_Feature_Importance.png"

def generar_grafico_importancia():
    print(f"[*] Buscando modelo en: {RUTA_MODELO}")
    
    if not RUTA_MODELO.exists():
        print("[!] Error: No encuentro el archivo modelo.pkl. Verifica la ruta.")
        return

    # 2. Cargar el diccionario guardado
    datos_guardados = joblib.load(RUTA_MODELO)
    modelo = datos_guardados['modelo']
    nombres_cols = datos_guardados['nombres_caracteristicas'] # Tu script original guardó esto, ¡genial!

    # 3. Calcular importancia
    importancia = modelo.get_feature_importance()
    indices = np.argsort(importancia)

    # 4. Graficar con estilo profesional
    plt.figure(figsize=(12, 8))
    
    # Barra horizontal
    plt.barh(range(len(indices)), importancia[indices], align='center', color='#4c72b0')
    
    # Etiquetas
    plt.yticks(range(len(indices)), np.array(nombres_cols)[indices], fontsize=11)
    plt.xlabel('Puntuación de Importancia (PredictionValuesChange)', fontsize=13, fontweight='bold')
    plt.title('Jerarquía de Importancia de Variables - CatBoost', fontsize=15, fontweight='bold', pad=15)
    
    # Grid suave
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 5. Guardar
    plt.savefig(RUTA_SALIDA, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] ¡Figura 6 generada exitosamente!")
    print(f"     Guardada en: {RUTA_SALIDA}")

if __name__ == "__main__":
    generar_grafico_importancia()
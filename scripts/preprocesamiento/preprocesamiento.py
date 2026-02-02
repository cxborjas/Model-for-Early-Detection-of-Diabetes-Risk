import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Dict

@dataclass
class ConfiguracionPreprocesamiento:
    archivo_entrada: str = "dataset/dataset.csv"
    archivo_salida_completo: str = "dataset/modificado.csv"
    archivo_train: str = "dataset/train.csv"
    archivo_test: str = "dataset/test.csv"
    carpeta_informe: str = "dataset/informe"
    nombre_informe_html: str = "informe_preprocesamiento.html"
    test_size: float = 0.2
    random_state: int = 42

class PreprocesadorDatos:
    def __init__(self, config: ConfiguracionPreprocesamiento):
        self.config = config
        self.mapeo_columnas = {
            "Diabetes_012": "estado_diabetes",
            "HighBP": "hipertension",
            "HighChol": "colesterol_alto",
            "CholCheck": "chequeo_colesterol_reciente",
            "BMI": "imc",
            "Smoker": "fumador_historico",
            "Stroke": "derrame_cerebral_previo",
            "HeartDiseaseorAttack": "enfermedad_cardiaca_o_infarto",
            "PhysActivity": "actividad_fisica_reciente",
            "Fruits": "consumo_frutas",
            "Veggies": "consumo_verduras",
            "HvyAlcoholConsump": "consumo_alcohol_elevado",
            "AnyHealthcare": "tiene_atencion_medica",
            "NoDocbcCost": "no_fue_medico_por_costo",
            "GenHlth": "salud_general",
            "MentHlth": "dias_mala_salud_mental",
            "PhysHlth": "dias_mala_salud_fisica",
            "DiffWalk": "dificultad_caminar",
            "Sex": "sexo",
            "Age": "rango_edad",
            "Education": "nivel_educativo",
            "Income": "rango_ingresos",
        }
        self.columnas_modelo = [
            "estado_diabetes",
            "imc",
            "rango_edad",
            "sexo",
            "actividad_fisica_reciente",
            "consumo_frutas",
            "consumo_verduras",
            "fumador_historico",
            "consumo_alcohol_elevado",
            "salud_general",
            "dias_mala_salud_fisica",
            "dias_mala_salud_mental",
            "dificultad_caminar",
        ]

    def generar_graficos(self, dataset: pd.DataFrame) -> Dict[str, str]:
        rutas_imagenes = {}
        
        if "estado_diabetes" in dataset.columns:
            frec = dataset["estado_diabetes"].value_counts().sort_index()
            total = frec.sum()
            plt.figure()
            frec.plot(kind="bar")
            plt.title("Distribución de estado_diabetes (0=Sin riesgo, 1=Riesgo)")
            plt.xlabel("estado_diabetes")
            plt.ylabel("Cantidad de registros")
            
            for i, (idx, val) in enumerate(frec.items()):
                porcentaje = val / total * 100
                plt.text(i, val, f"{val}\n({porcentaje:.1f}%)", ha="center", va="bottom", fontsize=8)
            
            nombre_img = "dist_estado_diabetes.png"
            ruta_img = os.path.join(self.config.carpeta_informe, nombre_img)
            plt.tight_layout()
            plt.savefig(ruta_img, dpi=300)
            plt.close()
            rutas_imagenes["dist_estado_diabetes"] = nombre_img

        if "imc" in dataset.columns:
            plt.figure()
            dataset["imc"].hist(bins=40)
            plt.title("Histograma de IMC")
            plt.xlabel("IMC")
            plt.ylabel("Cantidad de registros")
            nombre_img = "hist_imc.png"
            ruta_img = os.path.join(self.config.carpeta_informe, nombre_img)
            plt.tight_layout()
            plt.savefig(ruta_img, dpi=300)
            plt.close()
            rutas_imagenes["hist_imc"] = nombre_img
            
        return rutas_imagenes

    def generar_informe_html(self, dataset: pd.DataFrame, rutas_imagenes: Dict[str, str]):
        n_filas, n_columnas = dataset.shape
        
        df_info_vars = pd.DataFrame({
            "variable": dataset.columns,
            "tipo_dato": [str(dataset[col].dtype) for col in dataset.columns],
            "valores_nulos": [dataset[col].isna().sum() for col in dataset.columns],
        })
        
        desc = dataset.describe(include="all").transpose()
        
        tabla_vars_html = df_info_vars.to_html(index=False, classes="table table-striped", border=0)
        tabla_desc_html = desc.to_html(classes="table table-striped", border=0)
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Informe de preprocesamiento - Riesgo de diabetes</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333333; }}
        .resumen {{ background-color: #f5f5f5; padding: 10px 15px; border-radius: 5px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; margin-bottom: 20px; width: 100%; font-size: 13px; }}
        th, td {{ border: 1px solid #dddddd; padding: 6px 8px; text-align: left; }}
        th {{ background-color: #f0f0f0; }}
        img {{ max-width: 600px; display: block; margin-bottom: 20px; }}
        .nota {{ font-size: 12px; color: #666666; }}
    </style>
</head>
<body>
<h1>Informe de preprocesamiento del dataset de riesgo de diabetes</h1>
<div class="resumen">
    <p><strong>Cantidad de registros:</strong> {n_filas}</p>
    <p><strong>Cantidad de variables:</strong> {n_columnas}</p>
</div>
<h2>Variables incluidas en el modelo</h2>
{tabla_vars_html}
<h2>Estadísticas descriptivas</h2>
{tabla_desc_html}
<h2>Gráficas descriptivas</h2>
"""
        if "dist_estado_diabetes" in rutas_imagenes:
            html += f"""<h3>Distribución de estado_diabetes</h3><img src="{rutas_imagenes['dist_estado_diabetes']}" alt="Distribución estado_diabetes">"""
        
        if "hist_imc" in rutas_imagenes:
            html += f"""<h3>Histograma de IMC</h3><img src="{rutas_imagenes['hist_imc']}" alt="Histograma IMC">"""
            
        html += """
<p class="nota">
    Nota: este informe corresponde al dataset ya preprocesado, con la variable
    <code>estado_diabetes</code> binarizada (0 = sin riesgo, 1 = riesgo de prediabetes/diabetes)
    y utilizando únicamente factores individuales seleccionados para el modelo preventivo.
</p>
</body>
</html>
"""
        ruta_html = os.path.join(self.config.carpeta_informe, self.config.nombre_informe_html)
        with open(ruta_html, "w", encoding="utf-8") as f:
            f.write(html)
        return ruta_html

    def ejecutar(self):
        os.makedirs(self.config.carpeta_informe, exist_ok=True)
        
        print(f"Cargando dataset desde: {self.config.archivo_entrada}")
        df = pd.read_csv(self.config.archivo_entrada)
        
        df_es = df.rename(columns=self.mapeo_columnas)
        dataset_modelo = df_es[self.columnas_modelo].copy()
        
        print("Binarizando 'estado_diabetes' (0 vs 1/2 → 0/1)...")
        dataset_modelo["estado_diabetes"] = pd.to_numeric(dataset_modelo["estado_diabetes"], errors="coerce")
        dataset_modelo["estado_diabetes"] = (dataset_modelo["estado_diabetes"] > 0).astype(int)
        
        print("Distribución global de 'estado_diabetes' binarizado:")
        print(dataset_modelo["estado_diabetes"].value_counts())
        
        dataset_modelo.to_csv(self.config.archivo_salida_completo, index=False, encoding="utf-8")
        print(f"✅ Dataset completo preprocesado guardado en: {self.config.archivo_salida_completo}")
        
        print("\nRealizando partición train/test...")
        X = dataset_modelo.drop(columns=["estado_diabetes"])
        y = dataset_modelo["estado_diabetes"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y
        )
        
        train_df = X_train.copy()
        train_df["estado_diabetes"] = y_train
        test_df = X_test.copy()
        test_df["estado_diabetes"] = y_test
        
        train_df.to_csv(self.config.archivo_train, index=False, encoding="utf-8")
        test_df.to_csv(self.config.archivo_test, index=False, encoding="utf-8")
        
        print(f"Conjunto de entrenamiento guardado en: {self.config.archivo_train}")
        print(f"Conjunto de prueba guardado en: {self.config.archivo_test}")
        
        print("\nDistribución en TRAIN:")
        print(train_df["estado_diabetes"].value_counts(normalize=True))
        print("\nDistribución en TEST:")
        print(test_df["estado_diabetes"].value_counts(normalize=True))
        
        print("\nGenerando gráficas e informe HTML...")
        rutas_imagenes = self.generar_graficos(dataset_modelo)
        ruta_html = self.generar_informe_html(dataset_modelo, rutas_imagenes)
        print(f"Informe HTML generado en: {ruta_html}")
        
        print("\n Preprocesamiento + partición train/test + informe HTML completados.")
        print(f"   - {self.config.archivo_salida_completo}")
        print(f"   - {self.config.archivo_train}")
        print(f"   - {self.config.archivo_test}")
        print(f"   - {ruta_html}")

if __name__ == "__main__":
    config = ConfiguracionPreprocesamiento()
    preprocesador = PreprocesadorDatos(config)
    preprocesador.ejecutar()
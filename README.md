## Pipeline de Extracción de Características de Conectividad Funcional en rs-fMRI

Este pipeline, implementado en Python, transforma datos de resonancia magnética funcional en estado de reposo (rs-fMRI) en descriptores de conectividad estilo **tensor multi-canal**, listos para alimentar arquitecturas de aprendizaje profundo (por ejemplo, Autoencoders Variacionales). Está diseñado para maximizar la reproducibilidad, la escalabilidad y la rigurosidad metodológica.

---

### 1. Visión General

1. **Control de Calidad (QC)**

   * Eliminación de sujetos y volúmenes con artefactos graves.
   * Métricas univariantes (Z-score robusto) y multivariantes (Distancia de Mahalanobis con covarianza regularizada).
2. **Ingeniería de Conectividad**

   * Generación de un tensor `(canales, ROIs, ROIs)` por sujeto.
   * Extracción de biomarcadores de segundo orden (topología de grafos, dinámica de estados latentes).

---

### 2. Características Principales

* **Modularidad**: Paquetes independientes (`qc_bold`, `fmri_features`) para facilidad de mantenimiento y pruebas unitarias.
* **Configuración Declarativa**: Parámetros gestionados mediante archivos YAML (`config.yaml`, `config_connectivity.yaml`).
* **Procesamiento Paralelo**: Uso de `concurrent.futures.ProcessPoolExecutor` para distribuir el análisis por sujeto.
* **Canales de Conectividad**:

  1. `pearson_full`
  2. `pearson_omst` (grafo de expansión mínima ortogonal)
  3. `graphical_lasso`
  4. `tangent_space`
  5. `wavelet_coherence`
* **Reportes Automatizados**: CSV y HTML interactivo para auditoría de QC y resumen de resultados.

---

### 3. Estructura del Repositorio

```
├── scripts/
│   ├── run_qc_pipeline.py
│   └── run_connectivity_pipeline.py
│
├── qc_bold/
│   ├── io.py                # Gestión de I/O y configuración
│   ├── univariate.py        # Z-score robusto por ROI
│   ├── multivariate.py      # Distancia de Mahalanobis global
│   └── report.py            # Generación de CSV/HTML
│
├── fmri_features/
│   ├── data_loader.py       # Preprocesamiento y limpieza de series temporales
│   ├── connectome_generator.py  # Cálculo de tensores multi-canal
│   └── feature_extractor.py # Extracción de topología y HMM
│
├── config.yaml
├── config_connectivity.yaml
├── qc_outputs_example/      # Salidas de ejemplo de QC
└── connectivity_features/   # Salidas de conectividad
```

---

### 4. Flujo de Trabajo

#### Etapa 1: Control de Calidad (QC)

```bash
python3 scripts/run_qc_pipeline.py --config config.yaml
```

1. **Carga de datos**: Lectura de `.mat` y CSV.
2. **Cálculo de métricas básicas**: Longitud, ROIs válidos, % NaNs.
3. **Detección univariante**: Puntos atípicos por ROI.
4. **Detección multivariante**: Volúmenes atípicos por sujeto.
5. **Reporte**: `report_qc_final.csv` y `summary_report.html`.

#### Etapa 2: Extracción de Conectividad

```bash
python3 scripts/run_connectivity_pipeline.py --config config_connectivity.yaml
```

1. **Preprocesamiento avanzado**: Filtrado banda \[0.01–0.1 Hz], normalización, exclusión de ROIs erróneos.
2. **Generación de tensores**: Cálculo de cada canal de conectividad.
3. **Extracción de biomarcadores**:

   * **Dinámica de estados**: HMM.
   * **Topología de grafos**: Modularidad, clustering, eficiencia.
4. **Consolidación**: `processing_summary_log.csv` con rutas y métricas.

---

### 5. Configuración

* **`config.yaml`** (QC): Rutas, atlas, umbrales de outliers, criterios de descarte.
* **`config_connectivity.yaml`**: Canales activos, parámetros de filtrado, HMM.

---

### 6. Instalación Rápida

```bash
git clone <URL>
cd <REPO>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dependencias clave**: `numpy`, `pandas`, `scikit-learn`, `nilearn`, `pyyaml`, `dyconnmap`, `hmmlearn`, `bctpy`.

---

### 7. Salidas Generadas

* **QC**: `qc_outputs_<fecha>/report_qc_final.csv`, `summary_report.html`.
* **Conectividad**: `/connectivity_features/<run_id>/tensor_<SUBJECT>.npy`, `processing_summary_log.csv`.

---

### 8. Licencia y Contacto

* **Licencia**: MIT
* **Contacto**: [santiblaas@gmail.com](mailto:santiblaas@gmail.com)


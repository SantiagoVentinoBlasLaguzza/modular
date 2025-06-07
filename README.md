# Pipeline de Extracción de Características de Conectividad Funcional en rs-fMRI

Este repositorio ofrece un **pipeline modular** en Python para procesar series temporales de fMRI en estado de reposo (rs-fMRI) y extraer un conjunto multimodal de características de conectividad, listo para alimentar modelos de Machine Learning (p. ej. Autoencoders Variacionales).

---

## 1. Visión General

El flujo de trabajo consta de dos etapas secuenciales:

1. **Control de Calidad (QC)**  
   Detección de sujetos y *timepoints* con artefactos o ruido excesivo, mediante métodos univariantes y multivariantes.

2. **Extracción de Conectividad y Características**  
   Para los sujetos que superan el QC, generación de un tensor de conectividad 5-canal y cómputo de biomarcadores topológicos y dinámicos.

---

## 2. Características Destacadas

- **Arquitectura Modular**: Paquetes `qc_bold` y `fmri_features` independientes, con lógica claramente separada.  
- **Detección de Outliers**:  
  - *Univariante*: Z-score robusto (mediana + MAD).  
  - *Multivariante*: Distancia de Mahalanobis con covarianza robusta (Ledoit-Wolf, MCD) sobre 162 ROIs.  
- **Tensor Multi-Canal** (5 vistas):  
  1. Pearson completo  
  2. Pearson OMST (árbol de expansión mínima)  
  3. Correlación parcial (Graphical Lasso)  
  4. Espacio tangente (covarianza en `nilearn`)  
  5. Coherencia por wavelets  
- **Computación Paralela**: Uso de `ProcessPoolExecutor` para acelerar el procesamiento por sujeto.  
- **Configuración Centralizada**: Parámetros agrupados en `config.yaml` y `config_connectivity.yaml`.  
- **Reportes Interactivos**: CSV + HTML para revisar resultados de QC; logs detallados de extracción.

---

## 3. Estructura del Repositorio

.
├── scripts/
│   ├── run\_qc\_pipeline.py             # Ejecuta Etapa 1 (QC)
│   └── run\_connectivity\_pipeline.py   # Ejecuta Etapa 2 (Conectividad)
│
├── qc\_bold/                           # Paquete QC
│   ├── io.py                          # I/O y carga de config
│   ├── univariate.py                  # Outliers univariantes
│   ├── multivariate.py                # Outliers multivariantes
│   └── report.py                      # Generación de reportes CSV/HTML
│
├── fmri\_features/                     # Paquete de conectividad
│   ├── data\_loader.py                 # Carga, limpieza y preprocesamiento
│   ├── connectome\_generator.py        # Cálculo de tensor de conectividad
│   └── feature\_extractor.py           # HMM y métricas de teoría de grafos
│
├── config.yaml                        # Configuración QC
├── config\_connectivity.yaml           # Configuración Conectividad
├── requirements.txt                   # Dependencias mínimas
│
├── qc\_outputs/                        # Ejemplo de salida QC
│   ├── report\_qc\_final.csv
│   └── summary\_report.html
│
└── connectivity\_features/             # Salidas de extracción
└── connectivity\_YYYYMMDD\_HHMMSS/
├── tensor\_\<SUBJECT\_ID>.npy
└── processing\_summary\_log.csv

---

## 4. Flujo de Procesamiento

### 4.1 Etapa 1: Control de Calidad (QC)

```bash
python3 scripts/run_qc_pipeline.py
````

1. **Carga de datos** (`qc_bold/io.py`):
   Lee un CSV maestro, localiza cada archivo `.mat` y carga la configuración de `config.yaml`.

2. **QC inicial**:
   Verifica número de timepoints, porcentaje de NaNs y ROIs disponibles.

3. **Outliers univariantes** (`qc_bold/univariate.py`):
   Calcula Z-score robusto en cada ROI para detectar picos aislados.

4. **Outliers multivariantes** (`qc_bold/multivariate.py`):

   * Filtra 4 canales vacíos de AAL3, dejando 162 ROIs.
   * Calcula distancia de Mahalanobis de cada timepoint al centroide global.
   * Usa estimadores robustos de covarianza (Ledoit-Wolf/MCD).

5. **Reporte** (`qc_bold/report.py`):
   Genera `report_qc_final.csv` con columna `to_discard_overall` y `summary_report.html` interactivo.

### 4.2 Etapa 2: Extracción de Conectividad

```bash
python3 scripts/run_connectivity_pipeline.py
```

Solo se procesan sujetos con `to_discard_overall = False`.

1. **Preprocesamiento** (`fmri_features/data_loader.py`):

   * Elimina 39 ROIs (4 vacíos + 35 muy pequeños).
   * Descarta ROIs con varianza cero.
   * Aplica filtro banda 0.01–0.1 Hz y estandariza (μ=0, σ=1).

2. **Generación de tensor** (`fmri_features/connectome_generator.py`):
   Para cada sujeto, crea un array `(5, 162, 162)` con:

   * `pearson_full`
   * `pearson_omst`
   * `graphical_lasso`
   * `tangent_space`
   * `wavelet_coherence`

3. **Biomarcadores adicionales** (`fmri_features/feature_extractor.py`):

   * **HMM**: Ajuste de un Modelo Oculto de Markov.
   * **Grafos**: Eficiencia global, modularidad, coeficiente de clustering.

4. **Log final**:
   Escribe `processing_summary_log.csv` con rutas a tensores y todas las métricas extraídas.

---

## 5. Configuración

### config.yaml (QC)

```yaml
paths:
  input_csv: "/ruta/a/subjects.csv"
  output_dir: "/ruta/a/qc_outputs/"
atlas:
  remove_rois: [0,1,2,3]          # Índices de AAL3 vacíos
thresholds:
  z_score: 3.5
  max_nan_pct: 0.05
  max_mv_outliers_pct: 0.10
```

### config\_connectivity.yaml

```yaml
paths:
  qc_report: "/ruta/a/qc_outputs/report_qc_final.csv"
  output_base: "/ruta/a/connectivity_features/"
preprocessing:
  tr: 2.0             # Tiempo de repetición en segundos
  bandpass: [0.01,0.1]
channels: ["pearson_full","pearson_omst","graphical_lasso","tangent_space","wavelet_coherence"]
parameters:
  hmm:
    n_states: 5
    covariance_type: "full"
  graph:
    lasso_alpha: 0.01
```

---

## 6. Instalación

1. **Clonar y entrar al repositorio**

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```

2. **Entorno virtual (recomendado)**

   ```bash
   python3 -m venv venv_fmri
   source venv_fmri/bin/activate
   ```

3. **Dependencias mínimas**

   ```bash
   pip install -r requirements.txt
   ```

4. **Dependencias opcionales**

   ```bash
   pip install dyconnmap hmmlearn mne bctpy plotly
   ```

---

## 7. Uso

1. Actualiza rutas y parámetros en `config.yaml` y `config_connectivity.yaml`.
2. Ejecuta QC:

   ```bash
   python3 scripts/run_qc_pipeline.py
   ```
3. Ejecuta extracción de conectividad:

   ```bash
   python3 scripts/run_connectivity_pipeline.py
   ```

---

## 8. Salidas

* **QC**

  * `report_qc_final.csv`
  * `summary_report.html`

* **Conectividad**

  * Carpeta única por ejecución (`connectivity_YYYYMMDD_HHMMSS/`)
  * Tensores: `tensor_<SUBJECT_ID>.npy`
  * `processing_summary_log.csv`

---

> **Nota**: Ajusta parámetros de umbral y número de estados HMM según tus datos y objetivos de análisis.

# Pipeline de Extracción de Características de Conectividad Funcional en fMRI

## 1. Visión General

Este repositorio contiene un pipeline de neurociencia computacional en Python, diseñado para transformar series temporales de fMRI en estado de reposo (rs-fMRI) en un conjunto rico de características de conectividad. El objetivo final es generar datos robustos y de alta calidad, listos para ser utilizados en modelos de Machine Learning, como Autoencoders Variacionales (VAEs), para estudiar la dinámica cerebral.

El pipeline está dividido en dos etapas principales y modulares:

1.  **Control de Calidad (QC):** Una fase rigurosa para identificar y marcar sujetos cuyos datos son ruidosos o anómalos, asegurando la fiabilidad del análisis posterior.
2.  **Extracción de Conectividad y Características:** Un pipeline paralelo que procesa los sujetos que pasaron el QC, generando un "tensor de conectividad" multi-canal para cada uno y extrayendo características adicionales de topología de grafos y dinámica cerebral.

---

## 2. Características Principales

* **Modularidad:** El código está organizado en paquetes lógicos (`qc_bold`, `fmri_features`), facilitando el mantenimiento, la depuración y la extensión de la funcionalidad.
* **Control de Calidad Robusto:** Implementa métodos univariantes (Z-score robusto) y multivariantes (Distancia de Mahalanobis con estimadores de covarianza adaptativos) para una detección de outliers de alta fiabilidad.
* **Análisis Multi-Canal:** No se limita a una sola métrica. Genera un tensor de conectividad que apila múltiples "vistas" de la conectividad cerebral, incluyendo:
    1.  Correlación de Pearson
    2.  Correlación Parcial (Graphical Lasso)
    3.  Grafo de Expansión Mínima (OMST)
    4.  Espacio Tangente (Covarianza)
    5.  Coherencia por Wavelets
* **Paralelización Eficiente:** Utiliza `concurrent.futures.ProcessPoolExecutor` para procesar múltiples sujetos en paralelo, acelerando significativamente el tiempo de ejecución en máquinas multi-core.
* **Configuración Centralizada:** Todos los parámetros del pipeline (rutas, umbrales, configuraciones de modelos) se gestionan a través de archivos `YAML` (`config.yaml`, `config_connectivity.yaml`), permitiendo una fácil experimentación sin modificar el código.
* **Reportes Detallados:** Genera reportes exhaustivos en formato CSV y HTML interactivo para el QC, permitiendo una inspección visual y cuantitativa de la calidad de los datos.

---

## 3. Estructura del Repositorio


.
├── 📂 scripts/
│   ├── run_qc_pipeline.py             # 📜 Script ejecutor de la Etapa 1 (Control de Calidad)
│   └── run_connectivity_pipeline.py   # 📜 Script ejecutor de la Etapa 2 (Extracción de Características)
│
├── 📦 qc_bold/                           # 🐍 Paquete de Python para el Control de Calidad
│   ├── io.py                          #    - Manejo de entrada/salida de datos y configuración
│   ├── univariate.py                  #    - Detección de outliers en series temporales individuales
│   ├── multivariate.py                #    - Detección de outliers en patrones de actividad global
│   └── report.py                      #    - Generación de reportes CSV y HTML
│
├── 📦 fmri_features/                     # 🐍 Paquete de Python para la Extracción de Conectividad
│   ├── data_loader.py                 #    - Carga, limpieza y preprocesamiento de series temporales
│   ├── connectome_generator.py        #    - Cálculo del tensor de conectividad multi-canal
│   └── feature_extractor.py           #    - Extracción de biomarcadores (topología, HMM, etc.)
│
├── 📄 config.yaml                        # ⚙️ Archivo de configuración para la Etapa 1 (QC)
├── 📄 config_connectivity.yaml           # ⚙️ Archivo de configuración para la Etapa 2 (Conectividad)
│
├── 📊 qc_outputs_refactored_v1.0/        # 📁 Directorio de salida del QC (Ejemplo)
│   ├── report_qc_final.csv            #    - Reporte tabular con métricas y decisiones de descarte
│   └── summary_report.html            #    - Reporte visual e interactivo de la calidad de datos
│
└── 🧠 connectivity_features/             # 📁 Directorio base para las características extraídas
└── 📂 connectivity_5ch_20250607_161844/ # 📁 Carpeta de una ejecución específica (Ejemplo)
├── tensor_002_S_0295.npy      #    - Tensor de conectividad para un sujeto
├── ...                        #    - (Más archivos .npy para otros sujetos)
└── processing_summary_log.csv #    - Log y características escalares de todos los sujetos


---

## 4. El Pipeline de Procesamiento (Paso a Paso)

### **Etapa 1: Control de Calidad (QC)**

Este paso se ejecuta con `scripts/run_qc_pipeline.py` y su objetivo es limpiar la cohorte de sujetos.

1.  **Carga de Datos:** El script utiliza `qc_bold/io.py` para encontrar los archivos `.mat` de cada sujeto basándose en un CSV maestro.
2.  **QC Inicial:** Se realiza una verificación básica: número de puntos temporales (timepoints), ROIs y porcentaje de valores `NaN`.
3.  **Detección de Outliers Univariantes (`qc_bold/univariate.py`):**
    * Para cada ROI, se analiza su serie temporal de forma independiente.
    * Se utiliza un **Z-score robusto** (basado en la mediana y la Desviación Absoluta Mediana) para identificar puntos temporales cuyo valor de señal es extremadamente anómalo comparado con la actividad típica de ese mismo ROI. Esto es ideal para detectar picos de ruido aislados.
4.  **Detección de Outliers Multivariantes (`qc_bold/multivariate.py`):**
    * Este método evalúa cada *timepoint* (un "fotograma" de la actividad cerebral) en su conjunto.
    * Primero, se filtran los 4 canales del atlas AAL3 que no contienen datos.
    * Se calcula la **Distancia de Mahalanobis** de cada *timepoint* al centroide de todos los *timepoints*. Esta distancia tiene en cuenta la covarianza entre los 166 ROIs.
    * Es capaz de detectar *timepoints* anómalos que el método univariante no vería (ej. un patrón de actividad global atípico, a menudo causado por movimiento).
    * Utiliza estimadores de covarianza robustos como **Ledoit-Wolf** o **Minimum Covariance Determinant (MCD)** para evitar problemas matemáticos (singularidad de la matriz) cuando el número de ROIs es alto en relación con los *timepoints*.
5.  **Reporte Final (`qc_bold/report.py`):**
    * Se genera el archivo `report_qc_final.csv`, que contiene todas las métricas de QC para cada sujeto.
    * Una columna clave, `to_discard_overall`, marca con `True` a los sujetos que no cumplen con los criterios de calidad definidos en `config.yaml`.
    * Se crea un `summary_report.html` interactivo para visualizar las distribuciones y los resultados.

### **Etapa 2: Extracción de Conectividad**

Este paso se ejecuta con `scripts/run_connectivity_pipeline.py` y solo procesa los sujetos marcados con `False` en la columna `to_discard_overall` del reporte de QC.

1.  **Carga y Preprocesamiento de Datos (`fmri_features/data_loader.py`):**
    * Para cada sujeto válido, se carga su serie temporal.
    * **Limpieza de ROIs:** Se eliminan los 39 ROIs identificados en la configuración (los 4 de AAL3 y 35 que son demasiado pequeños).
    * **Filtrado de Varianza Cero:** Se realiza una verificación crucial para eliminar cualquier ROI cuya señal sea una línea plana (varianza cero), lo cual previene errores matemáticos posteriores.
    * **Procesamiento de Señal:** Se aplica un filtro de paso de banda (ej. 0.01-0.1 Hz) y se estandariza la señal (media cero, varianza uno).
2.  **Generación del Tensor de Conectividad (`fmri_features/connectome_generator.py`):**
    * Para cada sujeto, se genera un tensor de dimensiones `(canales, ROIs, ROIs)`. Cada "canal" es una matriz de conectividad calculada con un método diferente:
        * `pearson_full`: Matriz de correlación estándar.
        * `pearson_omst`: Una versión "filtrada" de la correlación de Pearson, que mantiene solo las conexiones más importantes para formar un grafo conexo (un "árbol de expansión"). Es útil para reducir el ruido.
        * `graphical_lasso`: Estimación de la correlación parcial. Mide la conexión directa entre dos ROIs eliminando la influencia de todos los demás.
        * `tangent_space`: Basado en la matriz de covarianza, es un método común en análisis de `nilearn`.
        * `wavelet_coherence`: Mide la coherencia (similitud de fase) entre pares de ROIs en diferentes bandas de frecuencia.
    * El tensor resultante se guarda como un archivo `.npy` individual para cada sujeto.
3.  **Extracción de Características Adicionales (`fmri_features/feature_extractor.py`):**
    * **Dinámica HMM:** Si se activa, se ajusta un Modelo Oculto de Markov (HMM) a la serie temporal para descubrir "estados cerebrales" recurrentes y se calculan métricas como la ocupación fraccional de cada estado.
    * **Topología de Grafos:** Usando la matriz de Pearson, se calculan métricas de teoría de grafos como la eficiencia global, la modularidad y el coeficiente de clustering, que describen la organización de la red cerebral.
4.  **Log Final:** Se genera un `processing_summary_log.csv` que contiene las rutas a los tensores y todas las características extraídas para cada sujeto, listo para ser analizado.

---

## 5. Configuración

La flexibilidad del pipeline reside en sus archivos de configuración:

* **`config.yaml`:** Controla la **Etapa 1 (QC)**. Aquí se definen:
    * Rutas a los datos de entrada y salida del QC.
    * Parámetros del atlas (índices de ROIs a eliminar).
    * Umbrales para la detección de outliers (ej. `z_threshold`).
    * Criterios para descartar sujetos (ej. `max_nan_pct`, `max_mv_outliers_pct`).

* **`config_connectivity.yaml`:** Controla la **Etapa 2 (Conectividad)**. Aquí se definen:
    * Rutas a los datos de entrada (incluyendo la salida del QC) y el directorio base de salida.
    * Qué canales de conectividad (`channels`) se deben calcular.
    * Parámetros de preprocesamiento de la señal (`preprocessing`), como la frecuencia de muestreo (TR) y la banda de frecuencia del filtro.
    * Parámetros específicos de cada modelo (ej. número de estados para HMM en `parameters.hmm`).

---

## 6. Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python3 -m venv venv_fmri
    source venv_fmri/bin/activate
    ```

3.  **Instalar las dependencias:**
    Se recomienda instalar las librerías principales primero. Un archivo `requirements.txt` sería ideal.
    ```bash
    pip install numpy pandas scipy scikit-learn nilearn matplotlib seaborn tqdm pyyaml
    # Librerías opcionales pero importantes para la funcionalidad completa
    pip install dyconnmap hmmlearn mne bctpy plotly
    ```

---

## 7. Uso

1.  **Configurar Rutas:** Edita `config.yaml` y `config_connectivity.yaml` para que las rutas (`paths`) apunten a las ubicaciones correctas de tus datos en tu sistema.
2.  **Ajustar Parámetros:** Revisa los umbrales y parámetros en ambos archivos de configuración para que se ajusten a las necesidades de tu análisis.
3.  **Ejecutar el Control de Calidad (Etapa 1):**
    ```bash
    python3 scripts/run_qc_pipeline.py
    ```
    Al finalizar, revisa los archivos en `qc_outputs.../` para asegurar que la calidad de los datos es la esperada.
4.  **Ejecutar la Extracción de Conectividad (Etapa 2):**
    ```bash
    python3 scripts/run_connectivity_pipeline.py
    ```
    Este paso puede tardar un tiempo considerable, dependiendo del número de sujetos y los canales de conectividad activados.

---

## 8. Salidas (Outputs)

* **Salidas del QC:**
    * `qc_outputs_.../report_qc_final.csv`: El reporte maestro que indica qué sujetos deben ser descartados.
    * `qc_outputs_.../summary_report.html`: Un reporte visual e interactivo para una fácil inspección.
* **Salidas de Conectividad:**
    * `connectivity_features/connectivity_.../`: Un directorio único para cada ejecución.
    * `tensor_SUBJECT_ID.npy`: Un tensor `(canales, ROIs, ROIs)` para cada sujeto.
    * `processing_summary_log.csv`: Un archivo CSV que resume la ejecución y contiene las características escalares extraídas (HMM, topología) para cada sujeto.


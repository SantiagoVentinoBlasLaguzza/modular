# Pipeline de Extracción de Características de Conectividad Funcional en fMRI

## 1. Visión General

Este repositorio alberga un pipeline avanzado de neurociencia computacional, implementado en Python, cuyo propósito es la transformación de datos de resonancia magnética funcional en estado de reposo (rs-fMRI) en un espacio de características de alta dimensionalidad. El objetivo primordial es generar descriptores de conectividad funcional robustos y fiables, optimizados para su ingesta en arquitecturas de aprendizaje profundo, tales como Autoencoders Variacionales (VAEs), con el fin de modelar la dinámica cerebral y extraer biomarcadores neurofuncionales.

El pipeline se articula en dos etapas fundamentales y modulares:

1.  **Protocolo de Control de Calidad (QC):** Una fase de cribado riguroso para la identificación y exclusión de adquisiciones de baja calidad o con artefactos significativos, garantizando la integridad de los análisis subsecuentes.
2.  **Ingeniería de Características de Conectividad:** Un proceso de alto rendimiento que opera sobre la cohorte depurada para generar un tensor de conectividad multi-modal y extraer descriptores de segundo orden (e.g., topología de grafos, dinámica de estados latentes).

---

## 2. Características Principales

* **Arquitectura Modular:** El código fuente está encapsulado en paquetes de Python (`qc_bold`, `fmri_features`), promoviendo la mantenibilidad, escalabilidad y la reutilización de componentes.
* **Control de Calidad Riguroso:** Implementa un protocolo de QC de dos niveles, combinando métricas univariantes (Z-score robusto) y multivariantes (Distancia de Mahalanobis) para una detección exhaustiva de artefactos.
* **Evaluación de Conectividad Multi-Modal:** Construye un conectoma funcional enriquecido apilando múltiples estimadores de conectividad, lo que proporciona una visión poliédrica de las interacciones cerebrales. Los canales incluyen:
    1.  Correlación de Pearson
    2.  Correlación Parcial (Graphical Lasso)
    3.  Grafo de Expansión Mínima Ortogonal (OMST)
    4.  Espacio Tangente (basado en Covarianza)
    5.  Coherencia espectral por Wavelets
* **Procesamiento Paralelo de Alto Rendimiento:** Aprovecha `concurrent.futures.ProcessPoolExecutor` para la paralelización a nivel de sujeto, optimizando el uso de recursos computacionales en sistemas multi-core.
* **Configuración Declarativa y Reproducible:** Emplea archivos de configuración `YAML` para gestionar todos los parámetros, desde rutas de sistema hasta hiperparámetros de modelos, garantizando la reproducibilidad y facilitando la experimentación.
* **Generación de Reportes Exhaustivos:** Produce reportes cuantitativos (CSV) y visuales (HTML interactivo) que resumen los resultados del QC, posibilitando una auditoría transparente de la calidad de los datos.

---

## 3. Arquitectura del Repositorio


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
│   ├── connectome_generator.py        #    - Cálculo del tensor de conectividad multi-modal
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

## 4. El Pipeline de Procesamiento: Detalle Técnico

### **Etapa 1: Protocolo de Control de Calidad (QC)**

Ejecutado a través de `scripts/run_qc_pipeline.py`, su función es la de depurar la cohorte inicial de sujetos.

1.  **Carga de Datos (`qc_bold/io.py`):** Se localizan y cargan las series temporales de cada sujeto, típicamente desde archivos `.mat`.
2.  **Métricas de Calidad Primarias:** Se computan descriptores básicos como la longitud de la serie temporal, el número de ROIs y el porcentaje de artefactos `NaN`.
3.  **Detección de Outliers Univariantes (`qc_bold/univariate.py`):** Se implementa un análisis a nivel de cada ROI individual. Mediante el **Z-score robusto**, basado en la mediana y la Desviación Absoluta Mediana (MAD), se identifican puntos temporales con fluctuaciones de señal atípicas, una técnica eficaz para la detección de picos de ruido esporádicos.
4.  **Detección de Outliers Multivariantes (`qc_bold/multivariate.py`):** Se evalúa la integridad de cada volumen funcional como un todo.
    * Tras el filtrado programático de ROIs inválidos del atlas, se calcula la **Distancia de Mahalanobis** para cada volumen, cuantificando la atipicidad de cada patrón de activación cerebral instantáneo con respecto a la distribución global del sujeto.
    * Este método es sensible a artefactos que afectan a múltiples regiones simultáneamente (e.g., movimiento cefálico).
    * Se emplean estimadores de covarianza regularizados (**Ledoit-Wolf**) y robustos (**Minimum Covariance Determinant**) para garantizar la estabilidad numérica en regímenes de alta dimensionalidad (N_ROIs > N_Timepoints).
5.  **Generación de Reportes (`qc_bold/report.py`):** Se sintetizan los resultados en:
    * `report_qc_final.csv`: Un reporte tabular que, para cada sujeto, contiene todas las métricas de QC y una columna booleana (`to_discard_overall`) que dictamina su inclusión o exclusión del análisis posterior.
    * `summary_report.html`: Un dashboard interactivo para la inspección visual de las métricas de calidad a nivel de cohorte.

### **Etapa 2: Extracción de Características de Conectividad**

Ejecutado con `scripts/run_connectivity_pipeline.py`, este pipeline procesa exclusivamente a los sujetos que han superado el QC.

1.  **Carga y Preprocesamiento Avanzado (`fmri_features/data_loader.py`):**
    * **Limpieza de ROIs:** Se eliminan sistemáticamente las columnas correspondientes a ROIs inválidos o con un volumen de vóxeles insuficiente, según lo especificado en la configuración.
    * **Filtrado de Varianza Nula:** Se implementa una verificación para detectar y eliminar ROIs con series temporales de varianza cero (señales planas), previniendo así singularidades en los cálculos de correlación.
    * **Procesamiento de Señal:** Las series temporales se someten a un filtro de paso de banda (e.g., 0.01-0.1 Hz) y se normalizan mediante estandarización (media 0, varianza 1).
2.  **Generación del Tensor de Conectividad (`fmri_features/connectome_generator.py`):**
    * El núcleo del pipeline. Para cada sujeto, se construye un tensor `(canales, ROIs, ROIs)`. Cada canal representa una matriz de adyacencia del conectoma funcional:
        * **`pearson_full`:** Correlación estándar, una medida de asociación lineal.
        * **`pearson_omst`:** Un subgrafo del conectoma de Pearson, filtrado para retener solo las conexiones más eficientes que garantizan la conectividad total de la red.
        * **`graphical_lasso`:** Estima la matriz de precisión inversa (correlación parcial), revelando las dependencias condicionales directas entre regiones tras eliminar las influencias de terceros.
        * **`tangent_space`:** Una representación de la conectividad basada en la matriz de covarianza, comúnmente utilizada en el marco de `nilearn`.
        * **`wavelet_coherence`:** Cuantifica la similitud de fase y frecuencia entre pares de ROIs a lo largo del tiempo.
    * Cada tensor se serializa a un archivo `.npy` individual.
3.  **Extracción de Biomarcadores (`fmri_features/feature_extractor.py`):**
    * **Dinámica de Estados Latentes:** Opcionalmente, se ajusta un Modelo Oculto de Markov (HMM) para inferir estados cerebrales discretos y se extraen métricas como la ocupación fraccional y la frecuencia de transición entre estados.
    * **Topología de Grafos:** Se calculan descriptores de la organización de la red cerebral (eficiencia global, modularidad, coeficiente de clustering) a partir de la matriz de conectividad base.
4.  **Consolidación de Resultados:** Se genera un archivo maestro `processing_summary_log.csv` que contiene las rutas a los tensores y todas las características escalares extraídas para cada sujeto de la cohorte final.

---

## 5. Configuración del Pipeline

La reproducibilidad y flexibilidad se gestionan a través de archivos `YAML` centralizados:

* **`config.yaml`:** Gobierna la **Etapa 1 (QC)**.
    * **`paths`:** Rutas de entrada y salida para el proceso de QC.
    * **`atlas`:** Especificaciones del atlas, incluyendo índices de ROIs a excluir.
    * **`outliers`:** Hiperparámetros para los algoritmos de detección de outliers.
    * **`exclusion_criteria`:** Umbrales cuantitativos para el descarte de sujetos.
* **`config_connectivity.yaml`:** Gobierna la **Etapa 2 (Conectividad)**.
    * **`paths`:** Rutas a los datos de entrada (incluyendo la salida del QC) y el directorio de salida final.
    * **`channels`:** Selector booleano para activar o desactivar el cálculo de cada canal de conectividad.
    * **`preprocessing`:** Parámetros para el procesamiento de la señal (TR, frecuencias de corte).
    * **`parameters`:** Hiperparámetros para los modelos de extracción de características (e.g., número de estados HMM).

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
    Se recomienda instalar las librerías a partir de un archivo `requirements.txt`.
    ```bash
    pip install numpy pandas scipy scikit-learn nilearn matplotlib seaborn tqdm pyyaml
    # Librerías opcionales pero importantes para la funcionalidad completa
    pip install dyconnmap hmmlearn mne mne-connectivity bctpy plotly
    ```

---

## 7. Modo de Empleo

1.  **Configurar Rutas:** Adapte las secciones `paths` en `config.yaml` y `config_connectivity.yaml` para que coincidan con la estructura de directorios de su sistema local.
2.  **Ajustar Parámetros:** Revise los umbrales y parámetros en ambos archivos de configuración para adecuarlos a los objetivos específicos de su estudio.
3.  **Ejecutar Etapa 1: Control de Calidad:**
    ```bash
    python3 scripts/run_qc_pipeline.py
    ```
    Tras la ejecución, inspeccione los artefactos generados en el directorio de salida del QC para verificar los resultados.
4.  **Ejecutar Etapa 2: Extracción de Conectividad:**
    ```bash
    python3 scripts/run_connectivity_pipeline.py
    ```
    Este proceso es computacionalmente intensivo. El tiempo de ejecución dependerá del tamaño de la cohorte y del número de canales de conectividad habilitados.

---

## 8. Artefactos Generados (Outputs)

* **Salidas del QC:**
    * `qc_outputs_.../report_qc_final.csv`: El reporte tabular maestro que dictamina la exclusión de sujetos.
    * `qc_outputs_.../summary_report.html`: El dashboard interactivo para la auditoría visual de la calidad de los datos.
* **Salidas de Conectividad:**
    * `connectivity_features/connectivity_.../`: Directorio único que contiene los resultados de una ejecución.
    * `tensor_SUBJECT_ID.npy`: Un tensor `(canales, ROIs, ROIs)` para cada sujeto procesado.
    * `processing_summary_log.csv`: El archivo CSV final que agrega las rutas a los tensores y las características escalares extraídas (HMM, topología, etc.) para toda la cohorte.

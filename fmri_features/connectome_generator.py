# fmri_features/connectome_generator.py
"""
Calcula diversas métricas de conectividad funcional (canales).
"""
from __future__ import annotations
import numpy as np
import logging
from sklearn.covariance import GraphicalLassoCV
from nilearn.connectome import ConnectivityMeasure
from typing import Dict
from numpy.linalg import LinAlgError

# Carga opcional de librerías avanzadas
try:
    from dyconnmap.graphs import threshold_omst_global_cost_efficiency
    OMST_AVAILABLE = True
except ImportError:
    OMST_AVAILABLE = False
try:
    from mne_connectivity.spectral import spectral_connectivity_time
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

log = logging.getLogger(__name__)

def _fisher_r_to_z(matrix: np.ndarray) -> np.ndarray:
    """Transformada de Fisher r-a-z para estabilizar la varianza."""
    return np.arctanh(np.clip(matrix, -0.9999, 0.9999))

# --- Funciones de Cálculo de Canales ---

def pearson_full(ts: np.ndarray, **kwargs) -> np.ndarray:
    """Canal 1: Correlación de Pearson con transformada de Fisher."""
    corr_matrix = np.corrcoef(ts, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix)
    return _fisher_r_to_z(corr_matrix)

def pearson_omst(ts: np.ndarray, **kwargs) -> np.ndarray | None:
    """Canal 2: Grafo de expansión mínima ortogonal (OMST) sobre Pearson."""
    if not OMST_AVAILABLE:
        log.warning("Librería 'dyconnmap' no encontrada. Omitiendo canal OMST.")
        return None
    corr_matrix = np.corrcoef(ts, rowvar=False)
    z_corr = _fisher_r_to_z(corr_matrix)
    
    z_corr_cleaned = np.nan_to_num(z_corr)
    
    if not np.any(z_corr_cleaned):
        log.warning(f"Sujeto {kwargs.get('subject_id', 'N/A')}: Matriz de correlación es cero. Devolviendo matriz de ceros para OMST.")
        return np.zeros_like(z_corr_cleaned)

    try:
        # --- CAMBIO PRINCIPAL: Añadir un epsilon para estabilidad numérica ---
        # Esto evita la división por cero en la librería dyconnmap si existen correlaciones de valor cero.
        epsilon = 1e-9
        input_matrix = np.abs(z_corr_cleaned) + epsilon
        
        omst_mask = threshold_omst_global_cost_efficiency(input_matrix)[1] > 0
        return z_corr_cleaned * omst_mask 
    except ValueError as e:
        # Capturar el error "argmax of empty sequence" si ocurre
        log.error(f"Fallo en dyconnmap para sujeto {kwargs.get('subject_id', 'N/A')}: {e}. Devolviendo None.")
        return None


def graphical_lasso(ts: np.ndarray, cfg: dict, **kwargs) -> np.ndarray | None:
    """Canal 3: Correlación parcial estimada con GraphicalLassoCV."""
    try:
        ts_cleaned = np.nan_to_num(ts)
        if np.any(np.std(ts_cleaned, axis=0) < 1e-8):
            log.warning(f"Sujeto {kwargs.get('subject_id', 'N/A')}: Datos con varianza cero detectados. GraphicalLasso puede fallar.")
        
        params = cfg['parameters']['graphical_lasso']
        estimator = GraphicalLassoCV(cv=params['cv_folds'], n_jobs=1).fit(ts_cleaned) # n_jobs=1 para mejor depuración
        log.info(f"GraphicalLassoCV finalizado para sujeto {kwargs.get('subject_id', 'N/A')}. Alpha: {estimator.alpha_:.4f}")
        return estimator.precision_.astype(np.float32)
    except LinAlgError:
        log.error(f"Fallo GraphicalLassoCV para sujeto {kwargs.get('subject_id', 'N/A')}: Matriz singular. Se devolverá None para este canal.")
        return None
    except Exception as e:
        log.error(f"Fallo inesperado en GraphicalLassoCV para sujeto {kwargs.get('subject_id', 'N/A')}: {e}")
        return None

def tangent_space(ts: np.ndarray, **kwargs) -> np.ndarray | None:
    """Canal 4: Conectividad basada en Covarianza (compatible con procesamiento individual)."""
    try:
        conn_measure = ConnectivityMeasure(kind='covariance')
        covariance_matrix = conn_measure.fit_transform([ts])[0]
        return covariance_matrix.astype(np.float32)
    except Exception as e:
        log.error(f"Fallo cálculo de Covariance: {e}")
        return None

def wavelet_coherence(ts: np.ndarray, cfg: dict, **kwargs) -> np.ndarray | None:
    """Canal 5: Coherencia media por Wavelets con mne_connectivity."""
    if not MNE_AVAILABLE:
        log.warning("Librería 'mne_connectivity' no encontrada. Omitiendo canal Wavelet.")
        return None
    try:
        pp_cfg = cfg['preprocessing']
        ts_mne = ts.T[np.newaxis, :, :]
        freqs = np.linspace(pp_cfg['low_cut_hz'], pp_cfg['high_cut_hz'], num=20)
        
        con = spectral_connectivity_time(
            ts_mne, freqs=freqs, method='coh', mode='cwt_morlet',
            sfreq=1.0/pp_cfg['tr_seconds'], n_jobs=1, verbose=False # n_jobs=1 para mejor depuración
        )
        mean_coh = con.get_data(output='dense').mean(axis=(2, 3))
        return mean_coh.astype(np.float32)
    except Exception as e:
        log.error(f"Fallo cálculo de Wavelet Coherence: {e}")
        return None

# Mapeo de nombres de canal a funciones
CONNECTIVITY_METHODS = {
    'pearson_full': pearson_full,
    'pearson_omst': pearson_omst,
    'graphical_lasso': graphical_lasso,
    'tangent_space': tangent_space,
    'wavelet_coherence': wavelet_coherence,
}

def generate_connectivity_tensor(ts_data: np.ndarray, cfg: Dict, subject_id: str) -> np.ndarray | None:
    """Genera un tensor 3D (canales x ROIs x ROIs) para un sujeto."""
    log.info(f"Sujeto {subject_id}: Generando tensor de conectividad.")
    matrices = []
    
    for channel_name, enabled in cfg['channels'].items():
        if not enabled:
            continue
        
        if channel_name in CONNECTIVITY_METHODS:
            log.debug(f"Sujeto {subject_id}: Calculando canal '{channel_name}'...")
            func = CONNECTIVITY_METHODS[channel_name]
            matrix = func(ts=ts_data, cfg=cfg, subject_id=subject_id)
            
            if matrix is not None:
                matrices.append(np.nan_to_num(matrix))
            else:
                log.error(f"Sujeto {subject_id}: El canal '{channel_name}' no se pudo calcular. Abortando tensor para este sujeto.")
                return None
        else:
            log.warning(f"No se encontró una función para el canal '{channel_name}' en la configuración.")

    if not matrices:
        log.error(f"Sujeto {subject_id}: No se generó ninguna matriz. No se puede crear el tensor.")
        return None
        
    return np.stack(matrices, axis=0).astype(np.float32)
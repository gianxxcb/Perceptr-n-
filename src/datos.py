"""Carga y adaptación explícita del dataset de Kaggle al enunciado del PDF."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTAS = ["grade_1", "grade_2", "final_grade"]


def cargar_estudiantes():
    datos = pd.read_csv(ROOT / "data/student_math_clean.csv")
    notas = datos[NOTAS].to_numpy(dtype=float)
    if len(datos) == 0 or not np.isfinite(notas).all() or ((notas < 0) | (notas > 20)).any():
        raise ValueError("El dataset debe contener notas completas entre 0 y 20.")
    if datos["student_id"].duplicated().any():
        raise ValueError("Los identificadores de estudiantes deben ser únicos.")
    return datos[["student_id", *NOTAS]].copy()


def preparar_datos(umbral=14.0):
    """80/20 reproducible por estudiante, sin usar el conjunto de prueba al entrenar."""
    if not np.isfinite(umbral) or not 0 < umbral <= 20:
        raise ValueError("El umbral debe estar entre 0 (excluido) y 20.")
    datos = cargar_estudiantes()
    x = (datos[NOTAS].to_numpy() >= umbral).astype(int)
    # Esta etiqueta es la regla adaptada del PDF, no la aprobación original de UCI.
    yd = (x.sum(axis=1) >= 2).astype(int)
    indices = np.random.default_rng(42).permutation(len(datos))
    corte = int(len(datos) * 0.8)
    entrenamiento, prueba = np.sort(indices[:corte]), np.sort(indices[corte:])
    return datos, x, yd, entrenamiento, prueba

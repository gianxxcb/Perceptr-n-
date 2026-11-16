"""Perceptrón binario con actualización secuencial según las diapositivas 9–11."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Resultado:
    pesos: np.ndarray
    bias: float
    historial: list[dict]
    pasos: list[dict]
    convergio: bool

    def net(self, entradas):
        """Suma ponderada; el bias tiene una entrada constante igual a uno."""
        return np.asarray(entradas, dtype=float) @ self.pesos + self.bias

    def predecir(self, entradas):
        return activar(self.net(entradas))


def activar(net):
    """El cero pertenece a la clase 1, tal como indica el PDF."""
    return (np.asarray(net) >= 0).astype(int)


def codificar_notas(notas, umbral=14.0):
    """Adapta las tres notas a unidades aprobadas (1) o desaprobadas (0)."""
    notas = np.asarray(notas, dtype=float)
    if notas.shape != (3,) or not np.all(np.isfinite(notas)):
        raise ValueError("Ingresa exactamente tres notas numéricas finitas.")
    if not np.isfinite(umbral) or not 0 < umbral <= 20:
        raise ValueError("La nota mínima debe ser mayor que 0 y hasta 20.")
    if np.any((notas < 0) | (notas > 20)):
        raise ValueError("Cada nota debe estar entre 0 y 20.")
    return (notas >= umbral).astype(int)


def entrenar(x, yd, fa=0.5, max_epocas=100, pesos=(0.5, 0.4, 0.3), bias=1.5):
    """Aprende sin mezclar patrones; conserva cada cálculo para poder explicarlo."""
    x, yd = np.asarray(x, dtype=float), np.asarray(yd, dtype=float)
    w = np.asarray(pesos, dtype=float).copy()
    if x.ndim != 2 or len(x) == 0 or x.shape[1] != 3:
        raise ValueError("Se requieren patrones con tres entradas.")
    if yd.shape != (len(x),) or not np.isin(yd, [0, 1]).all():
        raise ValueError("Cada patrón debe tener una salida binaria.")
    if not np.isin(x, [0, 1]).all():
        raise ValueError("Las entradas del entrenamiento deben ser binarias.")
    if w.shape != (3,) or not np.isfinite(w).all() or not np.isfinite(bias):
        raise ValueError("Los tres pesos y el bias deben ser finitos.")
    if not np.isfinite(fa) or not 0 < fa <= 1:
        raise ValueError("El factor de aprendizaje debe ser mayor que 0 y hasta 1.")
    if isinstance(max_epocas, bool) or not isinstance(max_epocas, int) or max_epocas < 1:
        raise ValueError("El máximo de épocas debe ser un entero positivo.")
    historial, pasos = [], []
    b = float(bias)
    convergio = False
    for epoca in range(1, max_epocas + 1):
        errores_online = 0
        for patron, (fila, deseado) in enumerate(zip(x, yd), start=1):
            anteriores, b_anterior = w.copy(), b
            net = float(fila @ w + b)
            salida = int(activar(net))
            error = int(deseado) - salida
            # Se usan inmediatamente los pesos corregidos para el siguiente patrón.
            w += fa * error * fila
            b += fa * error  # x_bias = 1
            errores_online += int(error != 0)
            pasos.append({
                "Época": epoca, "Patrón": patron,
                "x1": int(fila[0]), "x2": int(fila[1]), "x3": int(fila[2]),
                "w1 antes": anteriores[0], "w2 antes": anteriores[1],
                "w3 antes": anteriores[2], "b antes": b_anterior,
                "net": net, "yd": int(deseado), "y": salida, "error": error,
                "w1 después": w[0], "w2 después": w[1], "w3 después": w[2],
                "b después": b,
            })
        # Evaluar de nuevo todos los patrones con un único conjunto de pesos final.
        fallos = int(np.count_nonzero(activar(x @ w + b) != yd))
        historial.append({"Época": epoca, "Errores durante recorrido": errores_online,
                          "Errores al cierre": fallos, "w1": w[0], "w2": w[1],
                          "w3": w[2], "bias": b})
        if fallos == 0:
            convergio = True
            break
    return Resultado(w, b, historial, pasos, convergio)

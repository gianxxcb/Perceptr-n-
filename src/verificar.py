"""Comprobaciones reproducibles: python -m src.verificar."""

import unittest
from itertools import product
from pathlib import Path
import numpy as np
from .perceptron import activar, codificar_notas, entrenar
from .datos import preparar_datos


class PruebasPerceptron(unittest.TestCase):
    def test_dataset_kaggle(self):
        datos, x, yd, tr, te = preparar_datos()
        self.assertEqual(len(datos), 395)
        self.assertEqual((len(tr), len(te)), (316, 79))
        self.assertFalse(set(tr) & set(te))
        self.assertEqual(len(set(tr) | set(te)), len(datos))
        np.testing.assert_array_equal(preparar_datos()[3], tr)
        np.testing.assert_array_equal(yd, x.sum(axis=1) >= 2)
        r = entrenar(x[tr], yd[tr])
        self.assertTrue(r.convergio)
        np.testing.assert_array_equal(r.predecir(x[te]), yd[te])

    def test_solucion_completa(self):
        datos = np.loadtxt(Path(__file__).resolve().parents[1] / "data/patrones.csv",
                           delimiter=",", skiprows=1)
        x, yd = datos[:, :3], datos[:, 3]
        np.testing.assert_array_equal(x, list(product([0, 1], repeat=3)))
        np.testing.assert_array_equal(yd, x.sum(axis=1) >= 2)
        for fa in (0.01, 0.5, 1.0):
            r = entrenar(x, yd, fa=fa, max_epocas=1000)
            self.assertTrue(r.convergio)
            np.testing.assert_array_equal(r.predecir(x), yd)

    def test_actualizacion_y_limite(self):
        r = entrenar([[0, 0, 0]], [0], max_epocas=1)
        np.testing.assert_array_equal(r.pesos, [0.5, 0.4, 0.3])
        self.assertEqual(r.bias, 1.0)
        self.assertFalse(r.convergio)
        self.assertEqual(int(activar(0)), 1)
        np.testing.assert_array_equal(codificar_notas([13.99, 14, 20]), [0, 1, 1])
        np.testing.assert_array_equal(codificar_notas([0, 14, 20], 14), [0, 1, 1])

    def test_notas_invalidas(self):
        for notas in ([21, 0, 0], [-1, 11, 12], [float("nan"), 0, 0], [1, 2]):
            with self.assertRaises(ValueError):
                codificar_notas(notas)


if __name__ == "__main__":
    unittest.main()

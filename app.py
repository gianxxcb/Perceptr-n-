"""Interfaz educativa; los cálculos y la validación viven en src/."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


# Al ejecutar `python app.py`, inicia Streamlit con este mismo archivo.
# Dentro de Streamlit ya existe un contexto y no se vuelve a lanzar el servidor.
if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    import json
    import subprocess
    import tomllib

    tema = tomllib.loads((ROOT / "streamlit/theme.toml").read_text(encoding="utf-8"))
    opciones = []
    for seccion, valores in tema.items():
        for clave, valor in valores.items():
            if isinstance(valor, list):
                valor = json.dumps(valor)
            elif isinstance(valor, bool):
                valor = str(valor).lower()
            opciones.append(f"--{seccion}.{clave}={valor}")
    try:
        codigo_salida = subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
             "--server.headless=false", *opciones, *sys.argv[1:]], cwd=ROOT,
        )
    except KeyboardInterrupt:
        # subprocess.call cierra y espera al proceso hijo antes de propagar Ctrl+C.
        print("\nAplicación detenida.")
        codigo_salida = 0
    raise SystemExit(codigo_salida)

import numpy as np
import pandas as pd
from src.perceptron import codificar_notas, entrenar
from src.datos import preparar_datos


st.set_page_config(page_title="Perceptrón · Tres unidades", layout="wide", initial_sidebar_state="collapsed")
st.html("<style>" + (ROOT / "streamlit/styles.css").read_text(encoding="utf-8") + "</style>")
with st.sidebar:
    st.header("Perceptrón")
    st.caption("Clasificación de aprobación académica")
    pagina = st.radio("Navegación", ["Calcular Nota", "Cómo se hizo", "Resultados"],
                      key="navegacion", label_visibility="collapsed")
    st.divider()
    st.write("**Criterio de aprobación**")
    st.write("Al menos 2 de 3 unidades con nota mayor o igual a 14.")

st.title("Tres unidades. Una decisión.")
st.write("Explora cómo aprende un perceptrón a reconocer si un estudiante aprueba: "
         "necesita al menos dos unidades aprobadas.")
st.caption("Datos reales de Kaggle · Matemáticas · 395 estudiantes · 316 para entrenar y 79 para prueba.")


@st.cache_data
def cargar_datos():
    return pd.read_csv(ROOT / "data" / "patrones.csv")


patrones = cargar_datos()
umbral = 14.0
st.write("**Nota mínima para aprobar cada unidad: 14/20.**")
estudiantes, x, yd, idx_train, idx_test = preparar_datos(umbral)

if "resultado" not in st.session_state:
    st.session_state.resultado = entrenar(x[idx_train], yd[idx_train])
    st.session_state.parametros = (0.5, 100, (0.5, 0.4, 0.3), 1.5)

st.html('<div class="palette" aria-hidden="true"><span></span><span></span><span></span><span></span></div>')

if pagina == "Calcular Nota":
    st.subheader("Prueba tus notas")
    st.caption("Una unidad se aprueba con una nota mayor o igual a 14.")
    columnas = st.columns(3)
    notas = [col.number_input(f"Unidad {i + 1}", 0.0, 20.0, valor, 0.5)
             for i, (col, valor) in enumerate(zip(columnas, st.session_state.get("notas", [16.0, 14.0, 8.0])))]
    st.session_state.notas = notas
    entradas = codificar_notas(notas, umbral)
    r = st.session_state.resultado
    net = float(r.net(entradas))
    prediccion = int(r.predecir(entradas))
    esperado = int(entradas.sum() >= 2)
    st.html(f'<div class="verdict"><strong>{"Aprueba" if prediccion else "No aprueba"}</strong>'
            f'<p>Predicción del perceptrón · {int(entradas.sum())} de 3 unidades aprobadas</p></div>')
    st.write(f"Entradas binarias: **{entradas.tolist()}** · net: **{net:.4f}**")
    st.write(f"Criterio de aprobación: **{'Aprueba' if esperado else 'No aprueba'}**. "
             f"{'Coincide con el modelo.' if esperado == prediccion else 'El modelo todavía no coincide con la regla.'}")
    if not r.convergio:
        st.write("**Entrenamiento incompleto.** Ve a Cómo se hizo → Entrenar modelo y aumenta el máximo de épocas antes de usar la predicción.")
    st.caption("Las notas no se promedian ni se redondean: cada una se compara directamente con el umbral.")

elif pagina == "Cómo se hizo":
    st.subheader("Conoce y entrena el perceptrón")
    st.caption("Selecciona una de las dos secciones para consultar la explicación o entrenar el modelo.")
    explicacion, entrenamiento = st.tabs(["Explicación del modelo", "Entrenar modelo"], key="detalle_modelo")
    with explicacion:
        st.subheader("De las notas a una salida binaria")
        st.write("**1. Definir el criterio de aprobación.** Hay tres unidades, notas entre 0 y 20 "
                 "y una única condición: aprobar al menos dos unidades. No se calcula una nota final.")
        st.latex(r"x_i = \begin{cases}1 & \text{si nota}_i \geq 14 \\ 0 & \text{si nota}_i < 14\end{cases}")
        st.latex(r"y_d = \mathbb{1}[x_1+x_2+x_3\geq 2]")
        st.write("La codificación binaria representa cada unidad como aprobada o desaprobada, "
                 "utilizando un umbral de 14. Los ocho patrones agotan "
                 "todas las combinaciones de aprobación posibles y sirven de control teórico. "
                 "El entrenamiento utiliza los registros descargados de Kaggle.")
        st.write("**Adaptación del dataset.** Se usan grade_1, grade_2 y final_grade. En la fuente son "
                 "dos notas de período y una nota final; tratarlas como tres unidades es una adaptación "
                 "didáctica explícita. La etiqueta se calcula con el criterio de al menos dos unidades aprobadas, no se toma de "
                 "la aprobación original. No se trata de predecir una nota futura.")
        st.markdown("[Dataset en Kaggle](https://www.kaggle.com/datasets/dillonmyrick/high-school-student-performance-and-demographics) · "
                    "[Fuente original UCI, Paulo Cortez](https://archive.ics.uci.edu/dataset/320/student-performance)")
        st.write("**2. Calcular la suma ponderada.** Se combinan las tres entradas con sus pesos. "
                 "El bias tiene una entrada constante de valor 1.")
        st.latex(r"net = x_1w_1+x_2w_2+x_3w_3+b")
        st.write("**3. Aplicar la activación.** La salida es 1 cuando net es mayor o igual a cero y 0 cuando es negativo.")
        st.latex(r"y = \begin{cases}1 & net\geq 0\\0 & net<0\end{cases}")
        st.write("**Dos umbrales con funciones diferentes.** El 14 se aplica a cada nota antes "
                 "de entrar al modelo. El 0 se aplica a la suma ponderada después de combinar las entradas.")
        st.table(pd.DataFrame({
            "Umbral": ["Nota: 14", "Activación: 0"],
            "Qué compara": ["Cada nota entre 0 y 20", "net = suma ponderada + bias"],
            "Resultado": ["Nota ≥ 14 → entrada 1; nota < 14 → entrada 0",
                          "net ≥ 0 → predicción 1; net < 0 → predicción 0"],
        }))
        st.write("Una nota de 14 aprueba la unidad y un net de 0 activa la clase 1. "
                 "El bias desplaza la frontera de decisión; no es una nota ni se compara con 14.")
        st.write("**4. Corregir por patrón.** Se compara la salida con la deseada. "
                 "Si el error es cero, los parámetros no cambian.")
        st.latex(r"\begin{aligned}e&=y_d-y\\ w_i&\leftarrow w_i+fa\,e\,x_i\\ b&\leftarrow b+fa\,e\cdot1\end{aligned}")
        st.write("**5. Repetir y comprobar.** Se separan estudiantes con semilla 42: 80 % para entrenamiento "
                 "y 20 % para prueba. Cada época recorre únicamente los registros de entrenamiento, "
                 "conservando su orden original. Se detiene cuando todos ellos coinciden al cierre o "
                 "al alcanzar el límite. La prueba no interviene en las actualizaciones ni en la parada.")
        st.write("El modelo comienza con w1=0.5, w2=0.4, w3=0.3, bias=1.5 y fa=0.5. "
                 "Estos parámetros pueden ajustarse en la sección Entrenar modelo. "
                 "El aprendizaje corrige los pesos de un perceptrón de una sola neurona.")
        st.subheader("Ejemplo numérico paso a paso")
        st.write("Ejemplo didáctico independiente: partimos de los parámetros iniciales "
                 "predeterminados, no de los pesos actuales de la sesión ni del orden del dataset.")
        st.write("**Entradas y salida deseada.** Para las notas (16, 12, 8), la comparación con 14 "
                 "produce x=(1, 0, 0). Solo hay una unidad aprobada, por lo que yd=0.")
        st.latex(r"net = 1(0.5)+0(0.4)+0(0.3)+1.5=2")
        st.write("**Predicción y error.** Como net ≥ 0, el modelo predice y=1, aunque se espera 0.")
        st.latex(r"e=y_d-y=0-1=-1")
        st.write("**Corrección con fa=0.5.** Cada peso se ajusta según su propia entrada; "
                 "el bias utiliza una entrada constante de 1.")
        st.latex(r"w'_1=0.5+0.5(-1)(1)=0")
        st.latex(r"w'_2=0.4+0.5(-1)(0)=0.4")
        st.latex(r"w'_3=0.3+0.5(-1)(0)=0.3")
        st.latex(r"b'=1.5+0.5(-1)=1")
        st.write("Después de esta corrección, net=1 y la predicción todavía es 1: "
                 "una actualización no garantiza resolver un caso ni todo el conjunto. "
                 "El entrenamiento continúa corrigiendo y comprobando los patrones. "
                 "Cuando el error es 0, pesos y bias permanecen iguales.")
        st.subheader("Errores y criterio de parada")
        st.write("Una **época** es un recorrido completo de los registros de entrenamiento. "
                 "En este proyecto hay 316 registros por época.")
        st.write("**Errores durante el recorrido:** se cuenta cada predicción equivocada justo "
                 "antes de corregir sus pesos. Como estos cambian de un registro al siguiente, "
                 "las predicciones del recorrido se calculan con diferentes versiones del modelo.")
        st.write("**Errores al cierre:** después del recorrido se evalúan de nuevo los 316 registros "
                 "con un único conjunto de pesos finales, sin hacer correcciones. "
                 "La curva de convergencia usa esta evaluación; el entrenamiento termina cuando "
                 "el conteo llega a 0 o se alcanza el límite de épocas.")
        st.write("Puede haber errores durante una época y ninguno al cierre: las últimas "
                 "correcciones pueden haber resuelto los errores anteriores. Por eso no se exige "
                 "otra época completa sin correcciones para declarar la convergencia.")
        st.write("**Ejemplo con OR.** Con entradas (0,0), (0,1), (1,0), (1,1), pesos iniciales "
                 "(0,0), bias=-1 y fa=1, los errores durante los tres primeros recorridos son "
                 "1, 2 y 1; los errores al cierre son 1, 1 y 0. Este proyecto se detendría en "
                 "la época 3 con pesos (1,1) y bias=-1. Un procedimiento que espere un recorrido "
                 "entero sin errores registraría una cuarta época con 0 errores. Ambos obtienen "
                 "el mismo modelo; cambia la forma de comprobar la parada.")
        st.subheader("Alcance y limitación del perceptrón")
        st.write("**Por qué se puede resolver:** en las entradas binarias, w=(1,1,1), b=-1.5 "
                 "separa exactamente los casos con dos o tres unidades aprobadas. Es un ejemplo "
                 "analítico de separabilidad; los pesos que muestra la aplicación se aprenden.")
        st.write("Con tres entradas binarias, la frontera es un plano. La separabilidad lineal "
                 "se refiere a esas entradas transformadas, no a las notas originales sin codificar.")
        st.write("**Por qué no resuelve XOR.** XOR exige una salida 1 cuando sus dos entradas "
                 "son diferentes y 0 cuando son iguales:")
        st.table(pd.DataFrame({"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1],
                               "Salida XOR": [0, 1, 1, 0]}))
        st.write("Los dos casos de la clase 1 están en esquinas opuestas del cuadrado y "
                 "los de la clase 0 ocupan las otras dos. Ninguna recta puede separar "
                 "las clases. Cambiar la tasa de aprendizaje o aumentar las épocas no "
                 "elimina esa limitación: el perceptrón simple no puede acertar los cuatro casos. "
                 "Para resolver XOR se necesitan, por ejemplo, capas ocultas con activaciones "
                 "no lineales o una transformación de características que lo haga separable.")
        st.caption("XOR y OR son ejemplos explicativos independientes; no se incorporan al entrenamiento con estudiantes.")
    with entrenamiento:
        st.subheader("Entrenar modelo")
        st.caption("La sesión empieza con un modelo entrenado. Aplica nuevos parámetros para comparar.")
        fa_actual, epocas_actuales, pesos_actuales, bias_actual = st.session_state.parametros
        with st.form("entrenamiento"):
            fa = st.slider("Factor de aprendizaje (fa)", 0.01, 1.0, fa_actual, 0.01)
            max_epocas = st.number_input("Máximo de épocas", 1, 1000, epocas_actuales)
            cs = st.columns(3)
            pesos = tuple(c.number_input(f"Peso inicial w{i + 1}", -10.0, 10.0, v, 0.1)
                          for i, (c, v) in enumerate(zip(cs, pesos_actuales)))
            bias = st.number_input("Bias inicial", -10.0, 10.0, bias_actual, 0.1)
            if st.form_submit_button("Entrenar perceptrón", width="stretch"):
                st.session_state.resultado = entrenar(x[idx_train], yd[idx_train], fa, int(max_epocas), pesos, bias)
                st.session_state.parametros = (fa, int(max_epocas), pesos, bias)
                st.rerun()
        st.caption("Partición 80/20 con semilla 42. Solo se entrena con 316 registros. Umbral fijo: 14.")

elif pagina == "Resultados":
    r = st.session_state.resultado
    pred = r.predecir(x)
    fallos = int(np.count_nonzero(pred[idx_test] != yd[idx_test]))
    st.subheader("Lo que aprendió el perceptrón")
    st.write("**Convergencia alcanzada**" if r.convergio else "**Límite de épocas alcanzado: quedan errores**")
    cols = st.columns(3)
    cols[0].metric("Aciertos en prueba (79 registros)", f"{(len(idx_test) - fallos) / len(idx_test):.1%}")
    cols[1].metric("Épocas ejecutadas", len(r.historial))
    cols[2].metric("Errores en prueba", fallos)
    st.caption("Se evalúa una etiqueta derivada del criterio de aprobación sobre registros reales reservados. "
               "Esto mide reproducción de esa regla; no valida un pronóstico académico. "
               "Los patrones binarios pueden repetirse entre ambos conjuntos.")
    st.table(pd.DataFrame({"Parámetro": ["w1", "w2", "w3", "bias"],
                           "Valor final": [*r.pesos, r.bias]}))
    fa_usado, limite, pesos_usados, bias_usado = st.session_state.parametros
    st.caption(f"Entrenamiento aplicado: fa={fa_usado}; máximo={limite}; "
               f"pesos iniciales={pesos_usados}; bias inicial={bias_usado}.")
    st.subheader("Predicciones sobre estudiantes")
    tabla = estudiantes.copy()
    tabla[["x1", "x2", "x3"]] = x
    tabla["Resultado esperado"] = yd
    tabla["Conjunto"] = "Entrenamiento"
    tabla.loc[idx_test, "Conjunto"] = "Prueba"
    tabla["net"] = r.net(x)
    tabla["Predicción"] = pred
    tabla["Coincide"] = np.where(pred == yd, "Sí", "No")
    st.dataframe(tabla, hide_index=True)
    st.download_button("Descargar predicciones CSV", tabla.to_csv(index=False).encode("utf-8-sig"),
                       "predicciones.csv", "text/csv")
    st.subheader("Control de los ocho casos teóricos")
    control = patrones.copy()
    control["Predicción"] = r.predecir(control[["x1", "x2", "x3"]].to_numpy())
    control["Coincide"] = np.where(control["Predicción"] == control["yd"], "Sí", "No")
    st.table(control)
    st.write(f"Aciertos teóricos: {int((control['Predicción'] == control['yd']).sum())}/8. "
             "Estos casos no se añaden al entrenamiento.")
    st.subheader("Convergencia por época")
    errores_iniciales = int(np.count_nonzero(
        (x[idx_train] @ np.array(pesos_usados) + bias_usado >= 0).astype(int) != yd[idx_train]))
    puntos = [{"epoca": 0, "errores": errores_iniciales}] + [
        {"epoca": p["Época"], "errores": p["Errores al cierre"]} for p in r.historial]
    st.vega_lite_chart(spec={
        "data": {"values": puntos},
        "mark": {"type": "line", "point": {"filled": True, "fill": "#115166", "stroke": "#115166"}},
        "encoding": {
            "x": {"field": "epoca", "title": "Época", "type": "quantitative", "axis": {"tickMinStep": 1, "format": "d"}},
            "y": {"field": "errores", "title": "Errores de entrenamiento", "type": "quantitative",
                  "axis": {"format": "d", "tickMinStep": 1}, "scale": {"zero": True}},
            "color": {"value": "#115166"},
            "tooltip": [{"field": "epoca", "title": "Época", "type": "quantitative"},
                        {"field": "errores", "title": "Errores", "type": "quantitative"}],
        },
        "config": {"background": "#FFFFFF", "view": {"stroke": "#BDF1FF"},
                   "axis": {"labelColor": "#115166", "titleColor": "#115166",
                            "gridColor": "#BDF1FF", "domainColor": "#115166", "tickColor": "#115166"}},
    }, theme=None)
    st.caption("Época 0: errores con los pesos iniciales. Desde la época 1: errores sobre los 316 registros "
               "de entrenamiento después de actualizar los pesos de cada época.")
    st.write("**Cómo leer los errores:** durante el recorrido se cuentan fallos con pesos que "
             "van cambiando. Al cierre se comprueban todos los registros con los mismos pesos "
             "finales. Puede haber fallos durante la época y 0 al cierre; este último valor "
             "determina la parada. Son errores de entrenamiento, distintos de los errores "
             "en los 79 registros de prueba.")
    st.dataframe(pd.DataFrame(r.historial)[
        ["Época", "Errores durante recorrido", "Errores al cierre"]], hide_index=True)
    with st.expander("Ver cálculos de cada patrón"):
        epoca = st.number_input("Época que deseas inspeccionar", 1, len(r.historial), 1)
        pasos = pd.DataFrame(r.pasos)
        st.dataframe(pasos[pasos["Época"] == epoca], hide_index=True)
        st.download_button("Descargar todos los pasos", pasos.to_csv(index=False).encode("utf-8-sig"),
                           "entrenamiento.csv", "text/csv")

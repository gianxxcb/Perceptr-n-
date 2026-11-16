# Perceptrón: tres unidades, una decisión

Solución educativa del ejercicio de la última diapositiva de **EJEMPLO PERCEPTRON.pdf**,
con Python, Streamlit y un dataset real descargado de Kaggle. Se aprueba al tener
al menos dos unidades aprobadas entre tres, sin calcular un promedio final.

## Estructura detallada

```text
Perceptron/
├── app.py                            # Interfaz y arranque con python app.py
├── data/
│   ├── student_math_clean.csv        # 395 registros reales de Matemáticas
│   ├── fuente.json                   # Procedencia, fecha, adaptación y SHA-256
│   └── patrones.csv                  # Ocho casos teóricos para comprobación
├── src/
│   ├── __init__.py
│   ├── datos.py                      # Validación, codificación y partición 80/20
│   ├── perceptron.py                 # Activación y entrenamiento puro
│   └── verificar.py                  # Pruebas matemáticas y del dataset
├── streamlit/
│   ├── styles.css                    # Estilos de la interfaz
│   └── theme.toml                    # Colores nativos de Streamlit
├── requirements.txt                  # Dependencias directas con versiones exactas
└── README.md
```

`.venv/` es el entorno local con las dependencias instaladas. La carpeta
`streamlit/` solo contiene estilos y no lleva `__init__.py` para no ocultar el
paquete externo del mismo nombre. Las rutas de datos y estilos son absolutas,
calculadas a partir de la ubicación de `app.py`.

## Instalación y ejecución

Requiere Python 3.11 o posterior. Desde `Perceptron`:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Abre la URL que muestre la terminal, normalmente `http://localhost:8501`.
`app.py` inicia automáticamente el servidor de Streamlit y abre el navegador.
No hace falta ejecutar otro archivo ni escribir `streamlit run`.
Detén la aplicación con `Ctrl+C` en la terminal.

Si usas el entorno `.venv` ya preparado en este proyecto, puedes ejecutar:

```powershell
.\.venv\Scripts\python.exe app.py
```

El arranque carga automáticamente `streamlit/theme.toml`. Admite opciones como
`python app.py --server.headless=true --server.port=8502`.
En Linux/macOS, usa `python3` si tu instalación lo requiere.

## Dataset descargado y adaptación

- **Kaggle:** [High School Student Performance & Demographics, dillonmyrick](https://www.kaggle.com/datasets/dillonmyrick/high-school-student-performance-and-demographics).
- **Origen:** [Student Performance, UCI, Paulo Cortez (2008)](https://archive.ics.uci.edu/dataset/320/student-performance).
- **Cita:** Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. [DOI 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).
- **Licencia de la fuente original:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

La descarga contiene archivos de Matemáticas y Portugués. Esta aplicación usa solo
**Matemáticas: 395 filas**, sin mezclar ambas materias ni duplicar estudiantes entre
ellas. Se conserva únicamente el CSV que utiliza la aplicación, sin el ZIP duplicado.
`fuente.json` registra la URL de descarga, la fecha UTC y los hashes de la descarga original y del CSV.

Se usan `student_id`, `grade_1`, `grade_2` y `final_grade`. Las otras variables
demográficas del CSV se conservan como parte de la fuente y no se usan para entrenar.
Las tres notas están en escala de 0 a 20.

**Diferencia semántica importante:** la fuente contiene dos notas de período y una
nota final; no contiene tres unidades independientes. Aquí se reutilizan como tres
entradas en una adaptación didáctica explícita. No se presenta esta adaptación como
el criterio académico original de las escuelas portuguesas.

El PDF no define una nota mínima por unidad: **14 es el umbral indicado por el usuario**,
fijado en la interfaz y utilizado también como valor predeterminado de los cálculos.
Para cada una de las tres notas, `xi=1 si nota_i >= umbral; 0 en otro caso`.
No se redondean notas. La salida deseada se genera según el ejercicio:
`yd=1 si x1+x2+x3 >= 2; 0 en otro caso`.
La etiqueta es derivada, no una etiqueta real de aprobación proporcionada por Kaggle.
`final_grade` se utiliza como tercera entrada; **no se intenta predecirla**.

Los registros se dividen por estudiante con `numpy.random.default_rng(42)`:
**316 para entrenamiento y 79 para prueba**. Se conserva el orden original dentro
de cada subconjunto. Los índices no se solapan; las combinaciones binarias sí pueden
repetirse. La prueba no participa en actualizaciones ni decide la parada.
`patrones.csv` contiene los ocho casos posibles solo para verificar el modelo final;
no se añade a los datos de entrenamiento.

El dataset ya está incluido: la aplicación no necesita Internet ni descargas al arrancar.

## Navegación desde el menú hamburguesa

El botón hamburguesa **Menú**, situado arriba a la izquierda, abre la navegación
lateral. El menú empieza cerrado y permite cambiar de sección sin perder el modelo entrenado ni las
notas ingresadas. La opción seleccionada y los botones de acción utilizan fondo
oscuro con texto blanco; las superficies claras utilizan texto oscuro.

1. **Calcular Nota:** contiene únicamente **Prueba tus notas**, para introducir las
   tres notas y comprobar el resultado. El modelo inicial ya está entrenado.
2. **Cómo se hizo:** incluye dos subpestañas visibles: **Explicación del modelo**,
   con el desarrollo matemático y la preparación de datos, y **Entrenar modelo**,
   para ajustar factor de aprendizaje, pesos, bias y épocas. Los ajustes se aplican
   al pulsar **Entrenar perceptrón**.
   La explicación distingue el umbral de nota (14) del de activación (0), desarrolla
   un ejemplo numérico de corrección, compara errores durante el recorrido y al cierre,
   y explica el conteo de épocas con OR y la limitación ante XOR.
3. **Resultados:** pesos y bias aprendidos, aciertos y errores en los 79 registros
   de prueba, predicciones para los 395 estudiantes con su conjunto identificado,
   control de los ocho casos teóricos, curva de errores y trazas por patrón.
   Las predicciones y trazas pueden descargarse en CSV.

El umbral de la interfaz es fijo en 14. Los parámetros de entrenamiento realmente
aplicados aparecen en Resultados. Si se alcanza el límite antes de converger,
la interfaz lo indica.

## Matemática según el PDF

La diapositiva 12 establece tres unidades, al menos dos aprobadas y notas entre 0 y 20.
La codificación binaria adapta estas condiciones al perceptrón de las diapositivas 9–11:

```text
net = x1*w1 + x2*w2 + x3*w3 + b
y = 1 si net >= 0; en otro caso y = 0
error = yd - y
wi_nuevo = wi_anterior + fa*error*xi
b_nuevo = b_anterior + fa*error*1
```

Se extiende el ejemplo de dos a tres entradas. Se usan inicialmente `w1=0.5`,
`w2=0.4`, `b=1.5`, `fa=0.5` del ejemplo OR del PDF, con `w3=0.3` añadido
explícitamente. El ejemplo OR y el ejercicio final tienen distintas salidas deseadas.

Cada época recorre únicamente los registros de entrenamiento y corrige los pesos
inmediatamente después de cada patrón. Después, evalúa todo ese subconjunto con
los mismos pesos finales. Termina cuando no quedan errores al cierre o cuando
alcanza el límite de épocas. No usa backpropagation, sigmoide ni scikit-learn.

Ejemplo: el patrón `000 -> 0`, con los parámetros iniciales, produce `net=1.5`,
`y=1`, `error=-1`. Los tres pesos no cambian porque sus entradas son cero; el bias
pasa de `1.5` a `1.0`. Con error cero, ningún parámetro cambia.

Existe la solución analítica `w=(1,1,1), b=-1.5`, que demuestra separabilidad.
El programa no fija esos pesos: los aprende. Los errores del recorrido usan pesos
variables; la curva y la parada se basan en la evaluación conjunta al cierre.
La curva incluye una época 0 con los errores antes de entrenar para visualizar el cambio.

Con el CSV incluido, umbral 14 y parámetros predeterminados se obtuvo:

| Resultado | Valor |
|---|---|
| Pesos aprendidos | w1=0.5, w2=0.9, w3=0.8 |
| Bias aprendido | -1.0 |
| Épocas | 1 |
| Aciertos en entrenamiento | 316/316 |
| Aciertos en prueba | 79/79 |
| Control de los casos teóricos | 8/8 |

Los valores pueden cambiar con otros parámetros o una nueva versión del dataset.
**El 100 % mide reproducción de una regla determinista con etiquetas derivadas**,
no capacidad para anticipar el rendimiento de estudiantes ni validación del criterio
original de aprobación de UCI. Entradas y etiquetas se construyen a partir de las
mismas notas conocidas; no es una tarea de predicción de notas futuras.

## Diseño y paleta

| Código extraído de la imagen | Uso |
|---|---|
| `#FFFFFF` | Fondo principal y texto en botones oscuros |
| `#E9FBFF` | Superficies y controles |
| `#BDF1FF` | Bordes y cabeceras |
| `#3CC7E8` | Acento visual |
| `#115166` | Textos, botones, foco y gráfica |

Tema claro, controles nativos y estados comunicados por texto, sin alertas verdes
o rojas ni fuentes o imágenes externas. El tema usa las [opciones oficiales de
Streamlit](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-colors-and-borders).
Las dependencias directas están fijadas en `requirements.txt`; pip resuelve las transitivas.

## Verificación

```powershell
.\.venv\Scripts\python.exe -m src.verificar
.\.venv\Scripts\python.exe -m pip check
```

Las pruebas comprueban el dataset, separación sin solapamientos, reproducibilidad,
las ocho combinaciones, aprendizaje con tres factores distintos, actualización del
bias, límite de épocas, activación en cero, notas en el umbral y entradas inválidas.

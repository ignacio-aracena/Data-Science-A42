# TP1 — Clasificación del Dataset Iris
### Ignacio Aracena · Tomás Arizu | Ciencia de Datos — UdeSA

---

## Resumen ejecutivo

Este trabajo práctico analiza el dataset Iris con el objetivo de clasificar tres especies de flores (*Iris setosa*, *Iris versicolor* e *Iris virginica*) a partir de cuatro variables morfológicas. El trabajo sigue una estructura progresiva e intencionada: cada decisión —qué features crear, qué modelos probar, qué hiperparámetros buscar— se justifica a partir de lo que el análisis previo reveló. No se elige un modelo al azar y se reporta su accuracy: se construye entendimiento del problema desde los datos, y ese entendimiento guía cada paso del modelado.

Los hallazgos principales son:
- *Setosa* es **perfectamente separable** del resto con cualquier modelo — sus pétalos son tan pequeños que no hay superposición con las otras dos especies.
- El par difícil es **versicolor vs virginica**, cuyo solapamiento tiene una explicación histórica (orígenes de recolección distintos) y no solo biológica.
- El **feature engineering basado en el pétalo** es el factor que más mejora el rendimiento: pasar de 4 variables originales a 4 variables derivadas del pétalo sube la accuracy de 90% a 96.67%.
- En un dataset tan pequeño y limpio, todos los modelos bien configurados convergen al mismo techo. Las diferencias reales están en interpretabilidad, velocidad y cantidad de features necesarias.

---

## Índice

1. [El dataset Iris](#1-el-dataset-iris)
2. [Exploración de Datos (EDA)](#2-exploración-de-datos-eda)
3. [Análisis de Componentes Principales (PCA)](#3-análisis-de-componentes-principales-pca)
4. [Feature Engineering](#4-feature-engineering)
5. [Metodología de Modelado](#5-metodología-de-modelado)
6. [Experimentos y Resultados](#6-experimentos-y-resultados)
7. [Evaluación Completa: Errores Tipo I/II y Curvas ROC](#7-evaluación-completa-errores-tipo-iii-y-curvas-roc)
8. [Tabla Comparativa Final](#8-tabla-comparativa-final)
9. [Conclusiones](#9-conclusiones)

---

## 1. El dataset Iris

### Origen histórico y relevancia del contexto

El dataset Iris fue recolectado por el botánico Edgar Anderson y publicado por R. A. Fisher en 1936 para ilustrar el análisis discriminante lineal. Hoy es el "Hello World" del machine learning — el dataset más consultado en el repositorio UCI con casi 4 millones de visitas desde 2007.

Lo que raramente se menciona, pero es **fundamental para interpretar nuestros resultados**, es el contexto de recolección:

- *Iris setosa* e *Iris versicolor* fueron recolectadas **juntas** en la Península de Gaspé, Quebec, en 1935. Mismo día, misma persona, mismo instrumento.
- *Iris virginica* proviene de Camden, Tennessee, en **1926** — diferente lugar, diferente momento, posiblemente diferente instrumento.

El propio Fisher advirtió esta limitación en su paper original. Esto significa que el solapamiento que vamos a observar entre *versicolor* y *virginica* no es solo un fenómeno biológico: **tiene un componente metodológico** derivado de las condiciones de recolección. Este contexto explica por qué *setosa* es perfectamente separable (recolectada junto a versicolor, en condiciones controladas) mientras que el par versicolor/virginica siempre presenta confusión.

### Estructura del dataset

- **150 muestras** en total, **50 por especie** → dataset perfectamente balanceado
- **4 variables numéricas** (en centímetros): longitud y ancho del sépalo, longitud y ancho del pétalo
- **3 clases**: *Iris setosa* (0), *Iris versicolor* (1), *Iris virginica* (2)

El balance perfecto es una característica clave: permite usar *accuracy* como métrica confiable sin técnicas de balanceo (SMOTE, undersampling, etc.), y garantiza que el conjunto de test tendrá exactamente 10 muestras por especie.

---

## 2. Exploración de Datos (EDA)

El EDA no es un paso decorativo: es donde se construye la intuición que guía **todas** las decisiones posteriores de feature engineering y selección de modelos.

### 2.1 Estadísticas descriptivas

| Variable | Media | Std | Min | Max |
|---|---|---|---|---|
| longitud_sepalo | 5.84 cm | 0.83 | 4.3 | 7.9 |
| ancho_sepalo | 3.06 cm | 0.44 | 2.0 | 4.4 |
| longitud_petalo | 3.76 cm | **1.77** | 1.0 | 6.9 |
| ancho_petalo | 1.20 cm | **0.76** | 0.1 | 2.5 |

La primera señal aparece acá: las variables del pétalo tienen una desviación estándar mucho mayor (1.77 y 0.76) que las del sépalo (0.83 y 0.44). Mayor varianza entre muestras implica más información disponible para discriminar entre clases. Esto anticipa que el pétalo será más útil que el sépalo para clasificar, hipótesis que se confirma en cada análisis subsiguiente.

### 2.2 Balance de clases

50 muestras exactas por especie. El balance perfecto valida el uso de *accuracy* como métrica principal y elimina la necesidad de técnicas de corrección. Con clases desbalanceadas, un modelo podría lograr 90% de accuracy prediciendo siempre la clase mayoritaria — acá eso no es posible.

### 2.3 Histogramas con curva de densidad (KDE)

Al graficar la distribución de cada variable separada por especie, emerge el patrón central del dataset:

- **`longitud_petalo`**: distribución marcadamente bimodal. *Setosa* se concentra entre 1 y 2 cm. *Versicolor* y *virginica* se distribuyen entre 3 y 7 cm con solapamiento en la zona 4.5–5.5 cm.
- **`ancho_petalo`**: mismo patrón. *Setosa* en 0.1–0.6 cm; las otras dos en valores más altos con solapamiento.
- **`longitud_sepalo`** y **`ancho_sepalo`**: las tres especies se superponen significativamente. Estas variables solas no permiten distinguir bien las clases.

**Conclusión del EDA hasta acá**: las variables del pétalo son mucho más informativas que las del sépalo. Esta observación va a motivar directamente los experimentos 3 (solo pétalo) y 4 (solo binarias del pétalo) en la sección de modelado.

### 2.4 Scatter plots

Al graficar pares de variables coloreados por especie, la separabilidad se vuelve visual y concreta:

| Par de variables | Observación |
|---|---|
| longitud_petalo vs ancho_petalo | **Combinación más separable.** *Setosa* perfectamente aislada en la esquina inferior izquierda. *Versicolor* y *virginica* separadas pero con solapamiento en la región 4–5 cm. |
| longitud_sepalo vs ancho_sepalo | Máximo solapamiento entre las tres especies. La combinación menos útil para clasificación. |
| longitud_sepalo vs longitud_petalo | *Setosa* aislada, pero versicolor y virginica se confunden. |
| ancho_sepalo vs ancho_petalo | Ídem anterior. |

**Conclusión**: el par longitud/ancho del pétalo es la combinación más discriminativa. La separación de *setosa* es tan marcada que cualquier umbral simple en el pétalo la identifica. El desafío real es la frontera entre versicolor y virginica.

### 2.5 Boxplots

Los boxplots confirman los scatter plots y agregan información sobre variabilidad interna y outliers:

- Para las variables del pétalo, los rangos inter-cuartílicos de *setosa* no se solapan en absoluto con los de las otras dos especies.
- Se observan algunos outliers en las variables del sépalo de *virginica*, consistente con su origen geográfico diferente señalado en la sección 1.

### 2.6 Matrices de correlación

**Correlación global:**
- `longitud_petalo` ↔ `ancho_petalo`: correlación ~0.96 — altamente redundantes.
- `longitud_petalo` ↔ `longitud_sepalo`: correlación ~0.87.
- Variables del sépalo entre sí: correlación baja (~-0.11).

**Correlación por especie** — este análisis es el que más directamente motiva el feature engineering:

Las correlaciones cambian significativamente entre especies. Por ejemplo, la correlación `longitud_sepalo` ↔ `ancho_sepalo` es 0.74 en *setosa*, 0.53 en *versicolor* y 0.46 en *virginica*. Las relaciones entre variables no son uniformes en el dataset. Esto sugiere que existen interacciones entre variables que las variables originales capturan de forma incompleta, y que features derivadas (ratios, áreas, diferencias) podrían capturar mejor.

### Síntesis del EDA — ¿Qué aprendimos y cómo lo usamos?

| Hallazgo del EDA | Decisión que motiva |
|---|---|
| Variables del pétalo tienen mayor varianza y separabilidad | Diseñar features centradas en el pétalo |
| *Setosa* tiene pétalo muy pequeño (< 2 cm de longitud) | Crear variables binarias con ese umbral |
| Alta correlación entre longitud y ancho del pétalo (0.96) | Crear features que combinen ambas (área, suma, diferencia) |
| Correlaciones diferentes por especie | Crear ratios que capturen proporciones relativas |
| Solapamiento versicolor/virginica siempre presente | Esperar que ningún modelo los separe perfectamente |

---

## 3. Análisis de Componentes Principales (PCA)

### ¿Por qué aplicar PCA antes de modelar?

El PCA cumple dos roles distintos en este trabajo:

1. **Validación visual de la separabilidad**: proyectar 4 dimensiones en 2 permite ver con un único gráfico si el dataset es clasificable. Es una herramienta de comunicación poderosa.
2. **Cuantificación de la dimensionalidad intrínseca**: si el dataset puede resumirse en 2 componentes sin perder información relevante, eso tiene implicancias para la complejidad de los modelos que necesitamos.

Importante: escalamos los datos con `StandardScaler` **antes** de aplicar PCA. PCA maximiza la varianza en la dirección de los componentes principales, por lo que si las variables tienen escalas muy diferentes, las de mayor escala dominarían artificialmente las componentes. El escalado previo garantiza que todas las variables compiten en igualdad de condiciones.

### Varianza explicada

| Componente | Varianza individual | Acumulada |
|---|---|---|
| PC1 | ~73% | ~73% |
| PC2 | ~23% | **~96%** |
| PC3 | ~3.7% | ~99.5% |
| PC4 | ~0.5% | 100% |

Con solo **2 componentes se captura el 95.8% de la varianza total**. Esto significa que casi toda la información del dataset vive en un espacio de 2 dimensiones, aunque las variables originales sean 4.

**¿Qué representa cada componente?**
- **PC1** (73% de varianza): dominada por las variables del pétalo. Esencialmente mide el "tamaño del pétalo" — valores negativos = pétalo pequeño (*setosa*), valores positivos = pétalo grande (*versicolor* y *virginica*).
- **PC2** (23% de varianza): captura variación adicional, principalmente en las variables del sépalo. Permite distinguir parcialmente versicolor de virginica.

### Visualización 2D

La proyección en 2D confirma y sintetiza todo lo del EDA en un solo gráfico:
- *Setosa*: cluster perfectamente aislado en el extremo izquierdo (pétalo pequeño → PC1 muy negativa).
- *Versicolor* y *virginica*: separadas en la dirección de PC2, pero con solapamiento en la zona de transición.

Este gráfico es la evidencia más directa de que el dataset es **parcialmente separable**: dos clases perfectamente diferenciables, un par con solapamiento estructural. Cualquier modelo que alcance 96–97% en este dataset está en el límite de lo que los datos permiten.

---

## 4. Feature Engineering

### ¿Por qué hacer feature engineering?

El EDA mostró que:
- Las variables originales del pétalo son discriminativas pero tienen alta correlación entre sí (0.96).
- Las correlaciones entre variables varían por especie, sugiriendo que las combinaciones entre variables pueden capturar información que las variables individuales no capturan.
- Existen umbrales simples en el pétalo que permiten identificar directamente a *setosa*.

El feature engineering es la respuesta directa a estos hallazgos: creamos nuevas variables que capturan explícitamente las relaciones que el EDA identificó.

### Variables creadas

Partiendo de las 4 variables originales, creamos 11 nuevas features, llegando a **15 en total**:

| Tipo | Variable | Fórmula | Motivación |
|---|---|---|---|
| Ratio | `ratio_petalo` | longitud / ancho (pétalo) | Captura la proporción, independiente del tamaño absoluto |
| Ratio | `ratio_sepalo` | longitud / ancho (sépalo) | Ídem para el sépalo |
| Binaria | `es_petalo_pequeno` | 1 si longitud_petalo < 2.0 cm | Umbral observado en EDA que separa *setosa* del resto |
| Binaria | `es_ancho_petalo_pequeno` | 1 si ancho_petalo < 0.6 cm | Ídem, segunda condición que aísla a *setosa* |
| Área | `area_petalo` | longitud × ancho (pétalo) | Tamaño total del pétalo — captura la interacción entre dimensiones |
| Área | `area_sepalo` | longitud × ancho (sépalo) | Ídem para el sépalo |
| Polinómica | `longitud_petalo_2` | longitud_petalo² | Captura relaciones no lineales en la longitud |
| Polinómica | `ancho_petalo_2` | ancho_petalo² | Ídem para el ancho |
| Combinación lineal | `diff_petalo` | longitud − ancho (pétalo) | Mide si el pétalo es alargado o cuadrado |
| Combinación lineal | `diff_sepalo` | longitud − ancho (sépalo) | Ídem para el sépalo |
| Combinación lineal | `suma_petalo` | longitud + ancho (pétalo) | Tamaño total aproximado (alternativa al área) |

**Por qué cada tipo de feature tiene sentido:**

- **Ratios**: dos flores pueden tener el mismo largo de pétalo pero distinto ancho. El ratio captura la "forma" del pétalo independientemente de su tamaño. Un *setosa* con pétalo corto y ancho tiene un ratio diferente a una *versicolor* con pétalo largo y delgado.

- **Binarias**: si en el EDA observamos que *setosa* siempre tiene `longitud_petalo < 2.0`, crear `es_petalo_pequeno = 1` codifica esa regla directamente como feature. El modelo ya no tiene que "descubrir" ese umbral — se lo damos explícito.

- **Áreas**: la correlación de 0.96 entre longitud y ancho del pétalo sugiere que ambas variables miden cosas similares. El área (`longitud × ancho`) las combina en una sola variable con significado geométrico claro: es el tamaño total del pétalo. Esta combinación puede ser más informativa que cada dimensión por separado porque captura la interacción entre ambas.

- **Polinómicas**: si la frontera de decisión entre versicolor y virginica no es lineal en el espacio original, elevar al cuadrado puede ayudar a capturar esa curvatura. Es una forma de permitir que modelos lineales se comporten como no lineales.

- **Combinaciones lineales**: la diferencia entre longitud y ancho captura si el pétalo es alargado o cuadrado, una característica que las dimensiones individuales no expresan directamente.

### ¿Qué tan útiles resultaron estas features?

La importancia de features del Random Forest entrenado con las 15 variables lo confirma:

| Posición | Feature | Importancia | Tipo |
|---|---|---|---|
| 1 | `area_petalo` | 16.8% | Área |
| 2 | `ancho_petalo_2` | 13.3% | Polinómica |
| 3 | `suma_petalo` | 12.1% | Combinación lineal |
| 4 | `ancho_petalo` | 12.0% | Original |
| 5 | `longitud_petalo` | 11.4% | Original |
| 6 | `longitud_petalo_2` | 10.6% | Polinómica |
| 7 | `diff_petalo` | 6.0% | Combinación lineal |
| 8 | `es_petalo_pequeno` | 6.0% | Binaria |
| 9 | `diff_sepalo` | 5.4% | Combinación lineal |

Las 9 features con importancia > 5% son **todas derivadas del pétalo o combinaciones que lo incluyen**. Las variables del sépalo puras (`longitud_sepalo`, `ancho_sepalo`) quedan por debajo del 5% de importancia. Esto valida retrospectivamente la hipótesis del EDA: el sépalo no agrega información discriminativa relevante una vez que el pétalo está bien representado.

Las features creadas (área, polinómicas, sumas) ocupan los primeros puestos por encima de las variables originales. El feature engineering funcionó.

---

## 5. Metodología de Modelado

### División de datos

Dividimos el dataset en **80% entrenamiento (120 muestras) y 20% prueba (30 muestras)**. Dos decisiones técnicas que garantizan la validez de las comparaciones:

- **`stratify=y`**: garantiza que la proporción de clases sea idéntica en train y test. Sin esto, el split podría quedar desbalanceado por azar. Con stratify: train tiene 40/40/40 y test tiene 10/10/10.
- **`random_state=42` igual en todos los experimentos**: garantiza que todos los modelos se evalúan sobre **exactamente el mismo conjunto de test**. Si usáramos semillas distintas, los modelos verían conjuntos de test distintos y no sería una comparación justa.

### Elección de métricas y justificación de la prioridad

Para este problema usamos **accuracy** como métrica principal y **F1-macro** como complemento:

- **Accuracy** es válida porque el dataset es balanceado. Con 10 muestras por clase en test, un error en cualquier especie pesa igual (3.33%).
- **F1-macro** promedia el F1 de cada clase por igual. Es más robusta que accuracy cuando algún modelo falla completamente con una clase (como el experimento de solo binarias, donde el modelo no puede predecir *versicolor*). En esos casos, accuracy puede ser engañosa pero F1-macro lo detecta.
- **No usamos recall** como métrica principal porque no hay un costo asimétrico entre errores: clasificar una *versicolor* como *virginica* no es más grave que el error inverso. En un diagnóstico médico, priorizaríamos recall para minimizar Falsos Negativos. En un problema botánico-académico, el rendimiento balanceado es suficiente.

### ¿Qué métrica priorizar y por qué?

Esta es una decisión que depende del **contexto del problema**, no del algoritmo. En este trabajo priorizamos **accuracy** por las siguientes razones:

**1. El dataset está perfectamente balanceado (50 muestras por especie).**
Con clases balanceadas, accuracy refleja fielmente el rendimiento global. Si hubiera desbalance —por ejemplo, 90 *setosa* y 5 *versicolor*— un modelo que predice siempre *setosa* tendría 90% de accuracy sin aprender nada. Acá eso no es posible: las 3 clases tienen el mismo peso.

**2. No hay un costo asimétrico entre los tipos de error.**
En este problema, clasificar una *virginica* como *versicolor* tiene el mismo "costo" que el error inverso. No hay una especie más importante que otra ni consecuencias graves por confundirlas. Si fuera un problema médico (ej: clasificar un tumor como benigno cuando es maligno), priorizaríamos **recall** para minimizar los Falsos Negativos — un error de ese tipo puede costar una vida. En botánica académica, ese costo asimétrico no existe.

**3. F1-macro como métrica de seguridad.**
Aunque accuracy es la métrica principal, F1-macro actúa como verificación. El experimento de solo binarias lo ilustra claramente: ese modelo tiene 66.67% de accuracy pero F1-macro de 55.56% — la diferencia revela que *versicolor* no está siendo clasificada en absoluto. F1-macro detecta cuando un modelo sacrifica una clase para optimizar el promedio global.

**Resumen de criterios para elegir métricas en clasificación:**

| Situación | Métrica recomendada |
|---|---|
| Clases balanceadas, errores simétricos | Accuracy |
| Clases desbalanceadas | F1-macro o F1-weighted |
| Costo alto de Falsos Negativos (ej: diagnóstico) | Recall |
| Costo alto de Falsos Positivos (ej: spam) | Precision |
| Evaluación de capacidad discriminativa por clase | AUC-ROC |

En este TP, las condiciones del primer caso se cumplen. Accuracy es la elección correcta, respaldada por F1-macro para detectar comportamientos anómalos por clase.

### Estrategia de experimentos

Los 11 experimentos no son arbitrarios. Siguen una lógica progresiva basada en los hallazgos del EDA y el feature engineering:

| Grupo | Experimentos | Pregunta que responden |
|---|---|---|
| **Referencia** | RF Baseline | ¿Qué accuracy se logra sin ningún feature engineering? |
| **Impacto del FE** | RF Todas, RF Solo pétalo, RF Solo binarias, RF Orig+Ratios | ¿Cuánto aporta cada tipo de feature? ¿Cuáles son prescindibles? |
| **Algoritmo distinto** | KNN Pipeline | ¿Un algoritmo basado en distancias hace mejor o peor que uno basado en árboles? |
| **Boosting** | Gradient Boosting, AdaBoost, XGBoost | ¿Los métodos de boosting superan al Random Forest? |
| **Pipeline complejo** | RF→GB tuneado, PCA→GB | ¿Combinar selección/reducción + tuneo mejora los resultados? |

### Escalado explícito de datos

Para los experimentos de KNN y PCA→GB, el escalado y la reducción dimensional se realizan en pasos explícitos y separados. La clave es usar siempre `.fit_transform()` sobre el conjunto de entrenamiento y `.transform()` sobre el de test — nunca `.fit_transform()` sobre el test. Si ajustáramos el `StandardScaler` sobre todos los datos (incluyendo el test), el scaler "vería" las muestras de test durante el ajuste de la media y el desvío, lo que se conoce como **data leakage** y produce evaluaciones optimistas que no reflejan la performance real del modelo ante datos nuevos.

---

## 6. Experimentos y Resultados

### Experimento 1 — RF Baseline: el punto de partida

**Setup:** `RandomForestClassifier(n_estimators=100)` con las 4 variables morfológicas originales.

**¿Por qué empezar con Random Forest?** Es un modelo robusto, no requiere escalado de variables, maneja bien correlaciones entre features, y provee importancia de features que usaremos más adelante. Es el candidato natural para baseline.

**Resultados:**
- Accuracy: **90.0%** | F1-macro: **89.97%**
- Tiempo entrenamiento: 51.2ms | Tiempo predicción: 1.98ms
- *Setosa*: FP=0, FN=0 (clasificada perfectamente, como predijo el EDA)
- *Versicolor*: FP=2, FN=1
- *Virginica*: FP=1, FN=2
- **Total errores: 3 sobre 30 muestras**

**Interpretación:** El 90% sin feature engineering ya es un resultado sólido. Los 3 errores ocurren todos entre *versicolor* y *virginica*, exactamente donde el EDA mostraba solapamiento. *Setosa* no genera ningún error, también exactamente como el EDA anticipaba. Esto valida que el EDA describió correctamente la estructura del problema.

El 10% de error que queda es la motivación para el siguiente experimento: ¿puede el feature engineering reducirlo?

---

### Experimento 2 — RF con todas las features: ¿cuánto aporta el feature engineering?

**Setup:** Mismo RF, ahora con las 15 features (originales + engineered).

**Resultados:**
- Accuracy: **96.67%** | F1-macro: **96.66%**
- Tiempo entrenamiento: 52.3ms | Tiempo predicción: 2.63ms
- *Setosa*: FP=0, FN=0
- *Versicolor*: FP=0, FN=1
- *Virginica*: FP=1, FN=0
- **Total errores: 1 sobre 30 muestras**

**Interpretación:** El salto de 90% a 96.67% —de 3 a 1 error— demuestra que el feature engineering aporta información real. El modelo ahora se equivoca solo una vez, y ese error único ocurre en la zona de solapamiento estructural entre versicolor y virginica. Hemos llegado cerca del techo del dataset.

Ahora surge la siguiente pregunta: ¿**todas** las 15 features contribuyen al resultado, o hay un subconjunto más pequeño que logra lo mismo?

---

### Experimento 3 — RF Solo pétalo: ¿necesitamos el sépalo?

**Setup:** RF con solo las 4 features derivadas del pétalo (`longitud_petalo`, `ancho_petalo`, `ratio_petalo`, `area_petalo`).

**Motivación:** El EDA mostró que el sépalo tiene menor poder discriminativo. La importancia de features del experimento 2 confirmó que las variables del sépalo no llegan al 5% de importancia. ¿Podemos lograr el mismo resultado sin ellas?

**Resultados:**
- Accuracy: **96.67%** | F1-macro: **96.66%**
- Tiempo entrenamiento: 49.8ms

**Interpretación:** Exactamente el mismo resultado que con 15 features, usando solo 4. Esta es la conclusión más importante del trabajo sobre el feature engineering: **el sépalo no aporta información discriminativa relevante**. Toda la capacidad predictiva está concentrada en el pétalo.

Este resultado también tiene implicaciones prácticas: si quisiéramos aplicar este modelo en el mundo real, solo necesitaríamos medir el pétalo — la mitad de las mediciones.

---

### Experimento 4 — RF Solo binarias: ¿las reglas simples son suficientes?

**Setup:** RF con solo las 2 variables binarias (`es_petalo_pequeno`, `es_ancho_petalo_pequeno`).

**Motivación:** Las variables binarias fueron diseñadas para identificar directamente a *setosa*. Sirven como experimento límite: ¿qué pasa si solo usamos las reglas más simples que el EDA sugirió?

**Resultados:**
- Accuracy: **66.67%** | F1-macro: **55.56%**
- *Setosa*: FP=0, FN=0 (identificada perfectamente, como se esperaba)
- *Versicolor*: FP=0, FN=10 — **el modelo no predice ninguna *versicolor* correctamente**
- *Virginica*: FP=10, FN=0

**Interpretación:** El resultado expone la limitación de estas features de forma clara. Las variables binarias son perfectas para separar *setosa* del resto (todas las *setosa* tienen pétalo pequeño), pero incapaces de distinguir *versicolor* de *virginica* porque **ambas tienen pétalo grande** — los valores de estas dos variables son idénticos para las dos especies. El modelo termina asignando todas las flores de pétalo grande a *virginica*.

F1-macro de 55% es engañoso en este caso: el modelo tiene 100% de accuracy para *setosa* pero 0% para *versicolor*, lo que promedia a un número intermedio. El experimento enseña que las features binarias son útiles como complemento, no como conjunto único.

---

### Experimento 5 — RF Originales + Ratios: ¿los ratios agregan valor?

**Setup:** RF con las 4 variables originales más los 2 ratios (6 features en total).

**Resultados:**
- Accuracy: **93.33%** | F1-macro: **93.33%**
- **Total errores: 2 sobre 30 muestras**

**Interpretación:** Los ratios mejoran respecto al baseline (90% → 93.33%), pero no llegan al nivel de las features de área o polinómicas (96.67%). Los ratios capturan la proporción entre dimensiones, que es información útil pero menos potente que el área (que captura la interacción multiplicativa) o las polinómicas (que capturan curvatura). Este experimento ayuda a priorizar qué tipos de features generar en problemas futuros similares.

---

### Experimento 6 — KNN: ¿un algoritmo distinto da mejor resultado?

**Setup:** `KNeighborsClassifier` con las 4 features originales, escaladas previamente con `StandardScaler`. La selección de k se hace por validación cruzada de 5 folds sobre el conjunto de entrenamiento.

**¿Por qué escalar antes de KNN?** KNN mide distancias entre puntos. Si una variable va de 1 a 7 y otra de 0.1 a 2.5, la primera dominaría artificialmente el cálculo de distancias. El `StandardScaler` normaliza todas las variables antes de calcular distancias. El escalado se ajusta solo sobre los datos de entrenamiento y se aplica al test sin re-ajustar, evitando data leakage.

**Selección de k:**

| k | Accuracy (CV 5-fold) |
|---|---|
| 1 | 0.9417 |
| 3 | 0.9583 |
| **5** | **0.9667** |
| 7 | 0.9583 |
| 9 | 0.9583 |
| 11 | 0.9583 |

k=1 overfittea (cada punto es su propio vecino más cercano). k=5 es el óptimo: suficientemente robusto para generalizar sin perder sensibilidad.

**Resultados con k=5:**
- Accuracy: **93.33%** | F1-macro: **93.27%**
- Tiempo entrenamiento: **1.6ms** — el más rápido de todos los modelos
- *Versicolor*: FP=2, FN=0
- *Virginica*: FP=0, FN=2
- **Total errores: 2 sobre 30 muestras**

**Interpretación:** KNN logra 93.33% con las features originales, nivel similar al RF con features originales + ratios, pero con un tiempo de entrenamiento 30 veces menor. La velocidad de entrenamiento se explica porque KNN técnicamente no "aprende" nada: simplemente memoriza el conjunto de entrenamiento y clasifica cada nueva muestra por proximidad. El costo se paga en predicción (debe calcular distancias a todos los puntos de train), pero en Iris ese costo es negligible.

El resultado del KNN es comparable al RF baseline pero inferior al RF con feature engineering — tiene sentido: KNN en el espacio original de 4 features ve lo mismo que el RF baseline, y con las mismas limitaciones.

---

### Experimento 7 — Gradient Boosting: ¿el boosting supera al bagging?

**Setup:** `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)` con las 15 features.

**¿Cómo funciona el Gradient Boosting?** Construye árboles secuencialmente: cada árbol nuevo se ajusta para corregir los errores residuales del conjunto anterior. Es conceptualmente diferente al Random Forest (que construye árboles independientes en paralelo y promedia sus predicciones). El boosting tiende a tener menor sesgo pero mayor riesgo de overfitting.

**Resultados:**
- Accuracy: **96.67%** | F1-macro: **96.66%**
- Tiempo entrenamiento: 124.5ms — más del doble que el RF
- *Setosa*: FP=0, FN=0
- *Versicolor*: FP=0, FN=1
- *Virginica*: FP=1, FN=0
- **Total errores: 1 sobre 30 muestras**

**Interpretación:** Mismo rendimiento que el RF con todas las features, pero tardando 2.4 veces más en entrenar. En Iris, el problema es demasiado simple para que la mayor expresividad del boosting marque diferencia respecto al bagging. Ambos alcanzan el mismo techo de 1 error.

---

### Experimento 8 — AdaBoost: ¿otro método de boosting?

**Setup:** `AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME')` con las 15 features.

**¿Cómo difiere del Gradient Boosting?** AdaBoost usa árboles muy simples (stumps de 1 nivel) como estimadores base, y pondera las muestras: las mal clasificadas reciben más peso en la iteración siguiente. El Gradient Boosting usa árboles más profundos y ajusta los residuos directamente.

**Resultados:**
- Accuracy: **93.33%** | F1-macro: **93.33%**
- Tiempo predicción: 3.94ms — el más lento en predicción
- *Versicolor*: FP=1, FN=1 | *Virginica*: FP=1, FN=1
- **Total errores: 2 sobre 30 muestras**

**Interpretación:** AdaBoost queda por debajo del Gradient Boosting (2 errores vs 1). Los stumps de profundidad 1 son menos expresivos que los árboles de profundidad 3 del GB, lo que limita la capacidad del modelo para capturar la frontera sutil entre versicolor y virginica. Es un recordatorio de que la variante concreta de boosting importa.

---

### Experimento 9 — XGBoost: ¿la implementación más eficiente?

**Setup:** `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)` con las 15 features. Solo se ejecuta si xgboost está instalado.

**¿Qué hace diferente XGBoost?** Es una implementación optimizada de Gradient Boosting con regularización adicional (L1 y L2 sobre los pesos de los árboles), paralelización y manejo eficiente de datos dispersos. En datasets grandes es significativamente más rápido que sklearn.

**Resultados:**
- Accuracy: **93.33%** | F1-macro: **93.33%**
- Tiempo predicción: 1.63ms — el más rápido en predicción
- **Total errores: 2 sobre 30 muestras**

**Interpretación:** En Iris, XGBoost no supera al Gradient Boosting de sklearn ni al Random Forest. Sus ventajas (regularización adicional, manejo de datos dispersos, paralelización) son invisibles en un dataset de 150 muestras y 15 features densas. Las ventajas de XGBoost se vuelven evidentes en datasets de millones de filas y miles de columnas.

---

### Experimento 10 — Pipeline RF → Gradient Boosting tuneado: combinando lo aprendido

**Setup:** Tres pasos secuenciales que combinan los hallazgos previos.

**¿Por qué este experimento?** Los experimentos anteriores mostraron que: (a) la importancia de features del RF identifica las variables relevantes, y (b) el Gradient Boosting es el modelo de mayor rendimiento. Este experimento combina ambos: usar el RF como selector de features y el GB con hiperparámetros optimizados como clasificador final.

**Paso 1 — Selección de features por importancia del RF:**

Se filtran las features con importancia > 5% según el experimento 2. Resultado: **9 features** seleccionadas, todas del pétalo o derivadas de él (`area_petalo`, `ancho_petalo_2`, `suma_petalo`, `ancho_petalo`, `longitud_petalo`, `longitud_petalo_2`, `diff_petalo`, `es_petalo_pequeno`, `diff_sepalo`).

**Paso 2 — Búsqueda de hiperparámetros con `StratifiedKFold` (5 folds):**

| n_estimators | learning_rate | max_depth | F1-macro (CV 5-fold) |
|---|---|---|---|
| **50** | **0.10** | **2** | **0.9582** |
| 100 | 0.10 | 3 | 0.9582 |
| 150 | 0.05 | 3 | 0.9582 |
| 200 | 0.05 | 4 | 0.9496 |

Las tres primeras combinaciones empatan en 0.9582. Se elige `n_estimators=50` porque con menos árboles se obtiene el mismo resultado — más simple y más rápido de entrenar. Es el principio de parsimonia aplicado al tuneo de hiperparámetros.

**Paso 3 — Modelo final:**
- Accuracy: **93.33%** | F1-macro: **93.33%**
- Tiempo entrenamiento: 36ms
- **Total errores: 2 sobre 30 muestras**

**Interpretación:** A pesar de usar las mejores features según el RF y haber buscado los mejores hiperparámetros, el resultado es 93.33% — por debajo del RF directo con todas las features (96.67%). En un dataset de solo 120 muestras de entrenamiento, el proceso de selección de features + tuneo puede introducir varianza adicional que no se traduce en mejor generalización. En datasets más grandes, este pipeline metodológico marcaría más diferencia.

---

### Experimento 11 — PCA + Gradient Boosting: ¿la reducción dimensional penaliza?

**Setup:** Tres pasos explícitos: `StandardScaler` → `PCA(n_components=2)` → `GradientBoosting`, cada uno ajustado solo sobre los datos de entrenamiento. Features originales (4).

**Motivación:** El PCA mostró que 2 componentes capturan el 95.8% de la varianza. ¿Es suficiente ese 95.8% para clasificar bien, o el 4.2% restante contiene información crítica?

**Resultados:**
- Accuracy: **86.67%** | F1-macro: **86.67%**
- Tiempo entrenamiento: 156.1ms
- *Versicolor*: FP=2, FN=2 | *Virginica*: FP=2, FN=2
- **Total errores: 4 sobre 30 muestras**

**Interpretación:** El rendimiento cae respecto a todos los modelos con features completas. El 4.2% de varianza descartada es suficiente para duplicar los errores. Esto enseña algo importante: **la varianza no es lo mismo que la información discriminativa**. Una componente puede capturar poca varianza global pero ser muy relevante para separar dos clases específicas. El PCA maximiza varianza total, no separabilidad entre clases — para ese objetivo, técnicas como LDA (Linear Discriminant Analysis) serían más apropiadas.

Además, este experimento tiene la interpretabilidad más baja del trabajo: las features son combinaciones lineales abstractas (PC1, PC2) sin significado directo en el dominio del problema.

---

## 7. Evaluación Completa: Errores Tipo I/II y Curvas ROC

### Errores Tipo I y Tipo II

La matriz de confusión de cada experimento permite desglosar los errores por clase:

- **Error Tipo I (Falso Positivo)**: el modelo predice una especie que en realidad no es esa. Ejemplo: decir "es virginica" cuando en realidad es versicolor.
- **Error Tipo II (Falso Negativo)**: el modelo no detecta una especie cuando debería. Ejemplo: decir "no es versicolor" cuando sí lo es.

**Patrón consistente en todos los experimentos:**

*Setosa* tiene **cero errores Tipo I y Tipo II en todos los modelos**. Esta consistencia es la validación experimental más directa de la hipótesis del EDA: *setosa* es perfectamente separable porque sus pétalos son estructuralmente distintos a los de las otras dos especies.

Todos los errores ocurren exclusivamente entre *versicolor* y *virginica*. El único modelo que rompe este patrón es el RF Solo binarias, donde *versicolor* tiene 10 Falsos Negativos porque el modelo la confunde sistemáticamente con *virginica* (no con *setosa*).

### Curvas ROC (One-vs-Rest)

Las curvas ROC evalúan la capacidad de discriminación de cada modelo **por clase**, usando el enfoque One-vs-Rest: para cada especie, se trata como positiva y las demás como negativas.

Se comparan 7 modelos en 3 paneles (uno por especie):

**Setosa**: AUC = 1.00 en todos los modelos sin excepción. Cualquier umbral de probabilidad separa perfectamente a *setosa* del resto. Este resultado es la confirmación definitiva de su separabilidad perfecta.

**Versicolor y Virginica**: AUC entre 0.97 y 1.00 según el modelo, con leve variación. Los modelos con feature engineering completo tienden a tener AUC más alto. La línea de referencia (clasificador aleatorio) tiene AUC = 0.50.

El AUC como complemento a la accuracy es valioso porque mide la capacidad de discriminación a todos los umbrales posibles, no solo al umbral por defecto de 0.5. Un modelo puede tener alta accuracy pero bajo AUC (si ajustó el umbral a un valor no generalizable). En este caso, ambas métricas cuentan la misma historia: los modelos con features del pétalo son los mejores discriminadores.

---

## 8. Tabla Comparativa Final

| # | Modelo | Features | N° feat. | Accuracy | F1-macro | T. train | T. pred | Interpretabilidad |
|---|---|---|---|---|---|---|---|---|
| 1 | RF Todas features | todas (15) | 15 | **0.9667** | **0.9666** | 47ms | 2.2ms | Media |
| 2 | RF Solo pétalo | petalo (4) | 4 | **0.9667** | **0.9666** | 46ms | 1.8ms | Media |
| 3 | Gradient Boosting | todas (15) | 15 | **0.9667** | **0.9666** | 122ms | 1.0ms | Baja |
| 4 | RF Orig + Ratios | orig + ratios (6) | 6 | 0.9333 | 0.9333 | 46ms | 1.8ms | Media |
| 5 | AdaBoost | todas (15) | 15 | 0.9333 | 0.9333 | 55ms | 4.0ms | Baja |
| 6 | XGBoost | todas (15) | 15 | 0.9333 | 0.9333 | 71ms | 1.3ms | Baja |
| 7 | RF→GB tuneado | top features (9) | 9 | 0.9333 | 0.9333 | 36ms | 0.7ms | Baja |
| 8 | KNN (k=5) | originales (4) | 4 | 0.9333 | 0.9327 | **0.5ms** | 1.6ms | Alta |
| 9 | RF Baseline | originales (4) | 4 | 0.9000 | 0.8997 | 46ms | 1.8ms | Media |
| 10 | PCA + GB | PCA 2 componentes | 2 | 0.8667 | 0.8667 | 90ms | 0.5ms | Muy baja |
| 11 | RF Solo binarias | binarias (2) | 2 | 0.6667 | 0.5556 | 45ms | 1.8ms | Alta |

**Escala de interpretabilidad:**
- **Alta**: el razonamiento del modelo es directamente explicable (KNN: "clasifiqué por los 5 vecinos más cercanos"; binarias: "si longitud_petalo < 2, es setosa").
- **Media**: el modelo es una caja gris — se puede analizar con importancia de features, pero no se puede trazar el camino de decisión para cada muestra (Random Forest).
- **Baja**: los modelos de boosting combinan cientos de árboles secuenciales. La importancia de features existe pero el razonamiento global es opaco.
- **Muy baja**: PCA + boosting. Las features son combinaciones lineales abstractas sin significado en el dominio del problema.

---

## 9. Conclusiones

### El problema y la solución que encontramos

El dataset Iris plantea un problema de clasificación multiclase con una asimetría inherente: una de las tres clases (*setosa*) es trivialmente separable, mientras que el par *versicolor*/*virginica* presenta solapamiento estructural que ningún modelo puede eliminar completamente.

El análisis progresivo nos permitió entender esta estructura antes de modelar, y diseñar los experimentos para responder preguntas específicas en lugar de probar modelos al azar.

### Los tres aprendizajes centrales

**1. El EDA predijo correctamente lo que los modelos encontraron.**

Todo lo que el EDA mostró —que *setosa* es perfectamente separable, que las variables del pétalo dominan, que versicolor/virginica siempre se solapan— se confirmó exactamente en los resultados de modelado. El 100% de los errores en todos los experimentos ocurrió entre versicolor y virginica. Setosa tuvo cero errores en los 11 experimentos. Un EDA bien hecho no solo describe los datos: predice el comportamiento del modelo.

**2. El feature engineering importa, pero hay que elegir bien qué features crear.**

El paso de 4 features originales (90% accuracy) a 4 features del pétalo bien elegidas (96.67%) es el hallazgo más importante del trabajo. No fue agregar más variables lo que mejoró el modelo — fue agregar las variables correctas. El área del pétalo (longitud × ancho) resultó ser la feature más importante, superando a las variables originales. La diferencia entre el experimento 2 (15 features, 96.67%) y el experimento 3 (4 features del pétalo, 96.67%) muestra que el sépalo es irrelevante una vez que el pétalo está bien representado.

**3. En este dataset, la elección del algoritmo importa menos que la elección de las features.**

Cuatro modelos distintos — RF con todas las features, RF con solo pétalo, Gradient Boosting y Pipeline RF→GB tuneado — logran exactamente el mismo 96.67% con exactamente el mismo error (1 muestra mal clasificada). El techo no está en el algoritmo sino en los datos: ese 1 error restante es la muestra en la zona de solapamiento entre versicolor y virginica que ningún modelo, con ningún algoritmo, clasifica correctamente.

### Sobre cada familia de modelos

**Random Forest**: el modelo más equilibrado del trabajo. Alcanza el máximo rendimiento, es rápido (50ms), tiene interpretabilidad media a través de la importancia de features, y no requiere escalado previo. Es el candidato natural para este tipo de problema.

**KNN**: el resultado más sorprendente en términos de velocidad. Con 0.5ms de entrenamiento y 93.33% de accuracy, es el mejor trade-off rendimiento/velocidad del trabajo. La clave es el escalado previo con StandardScaler: sin él, KNN daría resultados mucho peores porque las variables del sépalo dominarían artificialmente el cálculo de distancias.

**Gradient Boosting (sklearn)**: igual rendimiento que el RF pero 2.4 veces más lento. En Iris, la mayor expresividad del boosting no agrega valor porque el problema es demasiado simple. El boosting brilla en datasets grandes y ruidosos donde reducir el sesgo iterativamente sí marca diferencia.

**AdaBoost y XGBoost**: por debajo del GB de sklearn. AdaBoost por usar estimadores base demasiado simples; XGBoost por no poder aprovechar sus ventajas (regularización, paralelización) en un dataset de 150 muestras.

**Pipeline PCA → GB**: el único experimento donde la reducción dimensional penaliza el rendimiento. El 4.2% de varianza descartada es suficiente para duplicar los errores. Enseña que varianza explicada no equivale a información discriminativa.

### La limitación estadística del dataset

Con 30 muestras de test y 3 clases, **cada error representa exactamente un 3.33% de accuracy**. La diferencia entre 90% y 96.67% corresponde a exactamente 2 errores de diferencia (de 3 a 1). Las conclusiones sobre qué modelo es "mejor" deben tomarse con cautela: en otra partición aleatoria, el orden podría cambiar. Para conclusiones estadísticamente robustas sería necesario validación cruzada repetida con múltiples particiones aleatorias, no un único split.

### Recomendación final

Si el objetivo fuera aplicar este modelo en un contexto real de identificación de especies de iris, la recomendación es el **RF con features del pétalo (experimento 3)**:

- Máximo rendimiento alcanzable en el dataset (96.67%).
- Solo 4 variables a medir, todas del pétalo — simplifica la recolección de datos.
- Interpretable a través de importancia de features.
- Rápido de entrenar y predecir.
- No requiere ajuste de hiperparámetros complejos.

El modelo más complejo (Pipeline RF→GB tuneado) logra el mismo resultado con mayor costo computacional y menor interpretabilidad. En ausencia de una razón específica para preferir la complejidad, la parsimonia es la mejor guía.

---

## Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `Iris_tp1.ipynb` | Notebook principal con todo el análisis y código ejecutado |
| `README.md` | Informe completo del trabajo práctico (este archivo) |
| `README_codigo.md` | Explicación técnica bloque por bloque del notebook |

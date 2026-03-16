# README — Explicación bloque por bloque: `Iris_tp1.ipynb`
**Ignacio Aracena · Tomás Arizu | Ciencia de Datos**

Este documento describe qué hace cada celda del notebook `Iris_tp1.ipynb`, qué output produce y qué significa cada resultado.

---

## Bloque 1 — Imports y configuración global

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
```

**¿Qué hace?**
- Importa todas las librerías del proyecto. Cada una tiene un rol específico: `pandas` para manejar el DataFrame, `sklearn` para los modelos y métricas, `seaborn`/`matplotlib` para visualizaciones.
- Intenta importar XGBoost con `try/except` — si no está instalado, `XGBOOST_AVAILABLE = False` y el experimento de XGBoost se omite automáticamente con un `if`.
- Define `CLASS_NAMES = ['setosa', 'versicolor', 'virginica']`, lista usada en todos los gráficos y reportes.
- Inicializa `resultados = []`, lista donde cada experimento agrega sus métricas para armar la tabla comparativa al final.

---

## Bloque 2 — Carga del dataset

```python
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo'])
iris_df['target']  = iris.target
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print('Dimensiones del dataset:', iris_df.shape)
display(iris_df.describe().T.round(2))
display(iris_df.head())
```

**¿Qué hace?**
- Carga el dataset Iris desde scikit-learn y lo convierte a un DataFrame con nombres de columnas en español.
- Agrega dos columnas: `target` (numérica: 0/1/2) y `species` (texto: nombre de la especie). El `.map()` convierte los números a etiquetas legibles.
- Imprime dimensiones (150 × 6) y estadísticas descriptivas transpuestas (`.T`) para leer una fila por variable en lugar de una columna.

**Output relevante:**
```
Dimensiones del dataset: (150, 6)
                 count  mean   std  min  25%   50%  75%  max
longitud_sepalo  150.0  5.84  0.83  4.3  5.1  5.80  6.4  7.9
longitud_petalo  150.0  3.76  1.77  1.0  1.6  4.35  5.1  6.9  ← std más alta
ancho_petalo     150.0  1.20  0.76  0.1  0.3  1.30  1.8  2.5  ← std más alta
```
Las variables del pétalo tienen desviación estándar mucho mayor, lo que anticipa mayor poder discriminativo para distinguir las especies.

---

## Bloque 3 — Balance de clases

```python
print(iris_df['species'].value_counts())
iris_df['species'].value_counts().plot(kind='bar', ...)
```

**¿Qué hace?**
- Cuenta muestras por especie e imprime el resultado.
- Genera un gráfico de barras de la distribución de clases.

**Output:** 50 muestras exactas por especie → dataset perfectamente balanceado. Esto valida usar *accuracy* como métrica principal sin necesidad de técnicas de corrección de desbalance.

---

## Bloque 4 — Histogramas con KDE

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(data=iris_df, x='longitud_sepalo', hue='species', kde=True, ax=axes[0, 0])
sns.histplot(data=iris_df, x='ancho_sepalo',    hue='species', kde=True, ax=axes[0, 1])
sns.histplot(data=iris_df, x='longitud_petalo', hue='species', kde=True, ax=axes[1, 0])
sns.histplot(data=iris_df, x='ancho_petalo',    hue='species', kde=True, ax=axes[1, 1])
```

**¿Qué hace?**
- Genera 4 histogramas (uno por variable) con curva de densidad KDE superpuesta, coloreados por especie.
- `kde=True` agrega la curva de densidad suavizada sobre el histograma, que facilita ver la forma de la distribución más allá de las barras.
- Cada subplot se asigna explícitamente con `ax=axes[fila, columna]`.

**Lectura clave:** `longitud_petalo` y `ancho_petalo` muestran distribución bimodal — *setosa* forma un grupo separado en valores bajos, mientras que *versicolor* y *virginica* se solapan en valores altos. Las variables del sépalo muestran mucho más solapamiento entre las tres especies.

---

## Bloque 5 — Scatter Plots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.scatterplot(data=iris_df, x='longitud_sepalo', y='ancho_sepalo',    hue='species', ax=axes[0, 0], palette='Set1')
sns.scatterplot(data=iris_df, x='longitud_petalo', y='ancho_petalo',    hue='species', ax=axes[0, 1], palette='Set1')
sns.scatterplot(data=iris_df, x='longitud_sepalo', y='longitud_petalo', hue='species', ax=axes[1, 0], palette='Set1')
sns.scatterplot(data=iris_df, x='ancho_sepalo',    y='ancho_petalo',    hue='species', ax=axes[1, 1], palette='Set1')
```

**¿Qué hace?**
- Grafica 4 combinaciones de variables en scatter plots coloreados por especie. Cada subplot corresponde a un par de variables.
- `hue='species'` colorea automáticamente cada punto según la especie.

**Lectura clave:** El par `longitud_petalo` vs `ancho_petalo` es la combinación más separable: *setosa* queda completamente aislada en la esquina inferior izquierda. *Versicolor* y *virginica* tienen solapamiento en la región 4–5 cm.

---

## Bloque 6 — Boxplots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(data=iris_df, x='species', y='longitud_sepalo', ax=axes[0, 0], palette='Set2')
sns.boxplot(data=iris_df, x='species', y='ancho_sepalo',    ax=axes[0, 1], palette='Set2')
sns.boxplot(data=iris_df, x='species', y='longitud_petalo', ax=axes[1, 0], palette='Set2')
sns.boxplot(data=iris_df, x='species', y='ancho_petalo',    ax=axes[1, 1], palette='Set2')
```

**¿Qué hace?**
- Genera 4 boxplots (uno por variable) con la distribución por especie: mediana (línea central), cuartiles Q1 y Q3 (caja), rango (bigotes) y outliers (puntos).

**Lectura clave:** En las variables del pétalo, los rangos de *setosa* no se solapan en absoluto con los de las otras dos especies. Se observan algunos outliers en las variables del sépalo de *virginica*.

---

## Bloque 7 — Matriz de correlación global

```python
sns.heatmap(iris_df[features].corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
```

**¿Qué hace?**
- Calcula la matriz de correlación de Pearson entre las 4 variables y la visualiza como heatmap.
- `annot=True` muestra los valores numéricos en cada celda. `cmap='coolwarm'`: azul = correlación negativa, rojo = positiva.

**Lectura clave:** `longitud_petalo` ↔ `ancho_petalo` tienen correlación ~0.96 — casi perfecta. Esto significa que cuando el pétalo es largo, también es ancho. `ancho_sepalo` tiene correlaciones bajas con el resto, confirmando que es la variable menos informativa.

---

## Bloque 8 — Matrices de correlación por especie

```python
for ax, sp in zip(axes, ['setosa', 'versicolor', 'virginica']):
    datos_especie = iris_df[iris_df['species'] == sp][features]
    sns.heatmap(datos_especie.corr(), annot=True, ...)
```

**¿Qué hace?**
- Filtra el DataFrame por especie y calcula la correlación de forma independiente para cada una.
- Genera 3 heatmaps apilados verticalmente, uno por especie.

**Lectura clave:** Las correlaciones varían entre especies — por ejemplo, la correlación entre longitud y ancho del sépalo es 0.74 en *setosa* y solo 0.46 en *virginica*. Esto justifica crear features que capturen estas diferencias.

---

## Bloque 9 — PCA: varianza explicada

```python
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(iris_df[features])

pca_full = PCA()
pca_full.fit(X_scaled)

varianza = pca_full.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza)
```

**¿Qué hace?**
1. Escala los datos con `StandardScaler` — PCA es sensible a la escala, sin este paso las variables con mayor rango dominarían las componentes.
2. Ajusta PCA con todas las componentes (`PCA()` sin `n_components`).
3. Calcula varianza individual y acumulada con `np.cumsum`.
4. Grafica barras (varianza individual) + línea con umbral del 95% (varianza acumulada).

**Output:** Con 2 componentes se explica ~95.8% de la varianza total. Esto significa que casi toda la información del dataset vive en 2 dimensiones.

---

## Bloque 10 — PCA: visualización 2D

```python
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = iris_df['species'].values
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='species', ...)
```

**¿Qué hace?**
- Aplica PCA reduciendo a 2 dimensiones y grafica el scatter plot en el espacio reducido.
- El título incluye el porcentaje de varianza explicada calculado dinámicamente.

**Lectura clave:** *Setosa* queda perfectamente aislada a la izquierda; *versicolor* y *virginica* se superponen parcialmente. Confirma que el dataset tiene solapamiento estructural entre esas dos especies.

---

## Bloque 11 — Feature Engineering

```python
# Ratios entre dimensiones
iris_df['ratio_petalo'] = iris_df['longitud_petalo'] / iris_df['ancho_petalo']
iris_df['ratio_sepalo'] = iris_df['longitud_sepalo'] / iris_df['ancho_sepalo']

# Variables binarias: reglas basadas en umbrales observados en el EDA
iris_df['es_petalo_pequeno']       = (iris_df['longitud_petalo'] < 2.0).astype(int)
iris_df['es_ancho_petalo_pequeno'] = (iris_df['ancho_petalo']    < 0.6).astype(int)

# Áreas: interacción entre longitud y ancho
iris_df['area_petalo'] = iris_df['longitud_petalo'] * iris_df['ancho_petalo']
iris_df['area_sepalo'] = iris_df['longitud_sepalo'] * iris_df['ancho_sepalo']

# Polinómicas: capturan relaciones no lineales
iris_df['longitud_petalo_2'] = iris_df['longitud_petalo'] ** 2
iris_df['ancho_petalo_2']    = iris_df['ancho_petalo']    ** 2

# Combinaciones lineales
iris_df['diff_petalo'] = iris_df['longitud_petalo'] - iris_df['ancho_petalo']
iris_df['diff_sepalo'] = iris_df['longitud_sepalo'] - iris_df['ancho_sepalo']
iris_df['suma_petalo'] = iris_df['longitud_petalo'] + iris_df['ancho_petalo']
```

**¿Qué hace?**
- Agrega 11 nuevas columnas directamente al DataFrame existente, llegando a 15 features en total.
- Las columnas binarias usan `(condicion).astype(int)` para convertir `True/False` a `1/0`.
- Cada feature tiene una motivación directa en el EDA: los umbrales de las binarias (< 2.0 y < 0.6) son los valores donde *setosa* se separa del resto según los histogramas.

**Output:** `Total de features: 15` + tabla con las primeras filas del DataFrame ampliado.

---

## Bloque 12 — Preparación para el modelado

```python
y = iris_df['target']

features_originales = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo']
features_petalo     = ['longitud_petalo', 'ancho_petalo', 'ratio_petalo', 'area_petalo']
features_binarias   = ['es_petalo_pequeno', 'es_ancho_petalo_pequeno']
features_orig_ratio = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo', 'ratio_petalo', 'ratio_sepalo']
features_todas      = [c for c in iris_df.columns if c not in ['target', 'species']]

X_tr_orig, X_te_orig, y_train, y_test = train_test_split(
    iris_df[features_originales], y, test_size=0.2, random_state=42, stratify=y
)
# ... idem para los otros 4 conjuntos de features
```

**¿Qué hace?**
- Define 5 conjuntos de features distintos, uno para cada grupo de experimentos.
- Realiza 5 splits con los mismos `random_state=42` y `stratify=y`, garantizando que todos los modelos se evalúan sobre el **mismo conjunto de test**.
- `stratify=y` asegura proporciones iguales de clases: train tiene 40/40/40 y test tiene 10/10/10.

**Output:** `Train: 120 muestras | Test: 30 muestras`

---

## Bloque 13 — RF Baseline (experimento 6.1)

```python
inicio = time.time()
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_tr_orig, y_train)
t_train = time.time() - inicio

inicio = time.time()
y_pred_rf_base = rf_base.predict(X_te_orig)
t_pred = time.time() - inicio

acc = accuracy_score(y_test, y_pred_rf_base)
f1  = f1_score(y_test, y_pred_rf_base, average='macro')
cm  = confusion_matrix(y_test, y_pred_rf_base)
```

**¿Qué hace?**
- Mide el tiempo de entrenamiento y predicción por separado con `time.time()`.
- Imprime accuracy, F1-macro, reporte completo por clase y la matriz de confusión como heatmap.
- Calcula FP (Error Tipo I) y FN (Error Tipo II) por clase usando la matriz de confusión: `FP = cm.sum(axis=0)[i] - cm[i][i]`, `FN = cm.sum(axis=1)[i] - cm[i][i]`.
- Agrega un diccionario con todos los resultados a la lista `resultados`.

**Output:** Accuracy 0.9000 → 27/30 flores correctas. Los 3 errores son entre *versicolor* y *virginica*. *Setosa* no genera ningún error. Este patrón se repite en todos los experimentos.

**Este patrón (fit → predict → métricas → confusión → FP/FN → resultados.append) se repite en todos los experimentos.**

---

## Bloque 14 — RF Todas las features + importancia (experimento 6.2)

```python
rf_todas = RandomForestClassifier(n_estimators=100, random_state=42)
rf_todas.fit(X_tr_todas, y_train)
# ... métricas, confusión, FP/FN ...

importancias = pd.Series(rf_todas.feature_importances_, index=features_todas)
importancias = importancias.sort_values(ascending=False)
importancias.plot(kind='bar', ...)
```

**¿Qué hace?** Igual que el baseline pero con las 15 features. Además, extrae el atributo `.feature_importances_` del RF (importancia promedio de cada feature en todos los árboles), lo ordena y lo grafica. Esta importancia se reutiliza en el experimento 6.10 para seleccionar features.

**Output:** Accuracy 0.9667 → 29/30 flores correctas. Solo 1 error. Las 5 features más importantes son todas del pétalo o derivadas de él. El feature engineering funcionó.

---

## Bloque 15 — RF Solo pétalo (experimento 6.3)

Mismo patrón que el baseline, entrenado con `X_tr_pet` (4 features del pétalo). Evalúa si el pétalo solo es suficiente para igualar el resultado con las 15 features.

**Output:** Accuracy 0.9667 → mismo resultado que con 15 features usando solo 4. Hallazgo clave: el sépalo no aporta información discriminativa relevante.

---

## Bloque 16 — RF Solo binarias (experimento 6.4)

Mismo patrón con `X_tr_bin` (2 features binarias).

**Output:** Accuracy 0.6667 → 20/30. *Setosa* perfecta, *versicolor* con precision y recall = 0.00. Las variables binarias identifican *setosa* pero no pueden distinguir *versicolor* de *virginica* porque ambas tienen pétalo grande — los valores son idénticos para esas dos especies.

---

## Bloque 17 — RF Originales + Ratios (experimento 6.5)

Mismo patrón con `X_tr_ratio` (6 features: 4 originales + 2 ratios).

**Output:** Accuracy 0.9333 → 28/30. Mejora respecto al baseline (90%) pero no llega al nivel de las features del pétalo (96.67%). Los ratios aportan, pero menos que el área o las polinómicas.

---

## Bloque 18 — KNN: selección de k (experimento 6.6, parte 1)

```python
# Escalamos los datos (KNN es sensible a la escala)
scaler_knn = StandardScaler()
X_tr_scaled = scaler_knn.fit_transform(X_tr_orig)
X_te_scaled = scaler_knn.transform(X_te_orig)

# Probamos distintos valores de k con validacion cruzada
k_values  = [1, 3, 5, 7, 9, 11]
cv_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_tr_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[cv_scores.index(max(cv_scores))]
```

**¿Qué hace?**
- Escala los datos explícitamente: `.fit_transform()` sobre el train y `.transform()` sobre el test (sin re-ajustar). KNN mide distancias — sin escalar, la variable de mayor rango dominaría artificialmente.
- Itera sobre 6 valores de k y evalúa cada uno con validación cruzada de 5 folds sobre el conjunto de entrenamiento.
- Elige el k con mayor accuracy promedio.

**Output:** El mejor k es 5, con accuracy CV de 0.9667. k=1 sobreajusta (memoriza); k muy alto pierde precisión local.

---

## Bloque 19 — KNN: modelo final (experimento 6.6, parte 2)

```python
inicio = time.time()
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_tr_scaled, y_train)
t_train = time.time() - inicio

inicio = time.time()
y_pred_knn = knn.predict(X_te_scaled)
t_pred = time.time() - inicio
```

**¿Qué hace?** Entrena KNN con el `best_k` sobre los datos ya escalados y registra resultados con el mismo patrón que los demás experimentos.

**Output:** Accuracy 0.9333 → 28/30. Tiempo de entrenamiento: 0.5ms — el más rápido de todos porque KNN no "aprende" nada, solo guarda los datos. El costo real está en la predicción, donde calcula distancias contra todos los puntos de train.

---

## Bloque 20 — Gradient Boosting (experimento 6.7)

```python
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_tr_todas, y_train)
```

Mismo patrón. Parámetros: 100 árboles secuenciales, tasa de aprendizaje 0.1 (qué tanto corrige cada árbol), profundidad máxima 3. Usa todas las 15 features.

**Output:** Accuracy 0.9667 → 29/30. Mismo resultado que RF con todas las features pero tardando ~2.5x más (122ms vs 47ms). En Iris la mayor complejidad no suma.

---

## Bloque 21 — AdaBoost (experimento 6.8)

```python
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42, algorithm='SAMME')
ada.fit(X_tr_todas, y_train)
```

Mismo patrón. `algorithm='SAMME'` es la versión discreta de AdaBoost compatible con clasificación multiclase en versiones recientes de sklearn.

**Output:** Accuracy 0.9333 → 28/30. Por debajo del Gradient Boosting. AdaBoost usa árboles de profundidad 1 (stumps) que son menos expresivos para capturar la frontera sutil entre *versicolor* y *virginica*.

---

## Bloque 22 — XGBoost (experimento 6.9)

```python
if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                        random_state=42, eval_metric='mlogloss', verbosity=0)
    xgb.fit(X_tr_todas, y_train)
```

**¿Qué hace?** El bloque completo está dentro de un `if XGBOOST_AVAILABLE`, por lo que si la librería no está instalada simplemente imprime el mensaje de instalación y no ejecuta nada.

**Output:** Accuracy 0.9333 → 28/30. Mismo resultado que AdaBoost. Las ventajas de XGBoost (regularización, paralelización) no se notan en un dataset de 150 muestras.

---

## Bloque 23 — RF → GB tuneado (experimento 6.10)

Este experimento tiene 3 sub-bloques:

**Sub-bloque 1 — Selección de features:**
```python
top_features = []
for feature, valor in importancias_ordenadas.items():
    if valor > 0.05:
        top_features.append(feature)
```
Recorre la importancia de features del RF (bloque 14) y guarda en la lista solo las que superan el 5% de importancia. Resultado: **9 features**, todas derivadas del pétalo.

**Sub-bloque 2 — Búsqueda de hiperparámetros:**
```python
param_grid = [
    {'n_estimators': 50,  'learning_rate': 0.1,  'max_depth': 2},
    {'n_estimators': 100, 'learning_rate': 0.1,  'max_depth': 3},
    {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 3},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4},
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for params in param_grid:
    scores = cross_val_score(GradientBoostingClassifier(**params), X_tr_top, y_train, cv=cv, scoring='f1_macro')
```
Itera sobre 4 combinaciones de hiperparámetros. Usa `StratifiedKFold` explícito con `shuffle=True` y `random_state=42` para garantizar que cada fold tenga la misma proporción de clases y que los resultados sean reproducibles.

**Output de la búsqueda:** Las tres primeras combinaciones empatan en F1-macro CV de 0.9582. Se elige `n_estimators=50` porque con menos árboles se obtiene el mismo resultado — principio de parsimonia.

**Sub-bloque 3 — Modelo final:**
```python
gb_tuned = GradientBoostingClassifier(**best_params, random_state=42)
gb_tuned.fit(X_tr_top, y_train)
```
Entrena con los mejores parámetros sobre las 9 features seleccionadas.

**Output:** Accuracy 0.9333 → 28/30. A pesar del proceso de selección y tuneo, no supera al RF directo. En datasets pequeños, el tuneo agrega varianza sin beneficio de generalización.

---

## Bloque 24 — PCA + GB (experimento 6.11)

```python
# Escalamos las features originales
scaler_pca_gb = StandardScaler()
X_tr_orig_scaled = scaler_pca_gb.fit_transform(X_tr_orig)
X_te_orig_scaled = scaler_pca_gb.transform(X_te_orig)

# Reducimos a 2 componentes principales
pca_modelo = PCA(n_components=2)
X_tr_pca = pca_modelo.fit_transform(X_tr_orig_scaled)
X_te_pca = pca_modelo.transform(X_te_orig_scaled)

# Entrenamos Gradient Boosting sobre las componentes principales
gb_pca = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_pca.fit(X_tr_pca, y_train)
```

**¿Qué hace?** Tres pasos explícitos y separados:
1. Escala los datos con `StandardScaler` (ajustado solo en train).
2. Reduce a 2 componentes principales con `PCA` (ajustado solo en train).
3. Entrena Gradient Boosting sobre las componentes.

Cada transformación usa `.fit_transform()` en train y `.transform()` en test para evitar data leakage.

**Output:** Accuracy 0.8667 → 26/30. 4 errores, todos entre *versicolor* y *virginica*. El 4.2% de varianza descartada por PCA contiene información crítica para separar esas dos especies. Además, las features perdieron interpretabilidad: ya no sabemos qué significa PC1 o PC2 en términos de medidas de la flor.

---

## Bloque 25 — Curvas ROC

```python
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

modelos_roc = [
    ('RF Baseline',       rf_base,     X_te_orig),
    ('RF Todas features', rf_todas,    X_te_todas),
    ('KNN',               knn,         X_te_scaled),
    ('Gradient Boosting', gb,          X_te_todas),
    ('AdaBoost',          ada,         X_te_todas),
    ('RF->GB tuneado',    gb_tuned,    X_te_top),
    ('PCA + GB',          gb_pca,      X_te_pca),
]

for ax, class_idx, class_name in zip(axes, [0, 1, 2], CLASS_NAMES):
    for (nombre, modelo, X_te), color in zip(modelos_roc, colors):
        proba = modelo.predict_proba(X_te)[:, class_idx]
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{nombre} (AUC={roc_auc:.2f})')
```

**¿Qué hace?**
1. `label_binarize`: convierte `y_test` (vector de 0/1/2) en una matriz (30 × 3) de indicadores binarios, necesaria para el enfoque One-vs-Rest.
2. Define la lista de modelos con su respectivo conjunto de test (importante: cada modelo fue entrenado con un conjunto distinto, por eso cada uno tiene su `X_te` correspondiente).
3. Genera 3 paneles (uno por clase): para cada modelo extrae las probabilidades de la clase actual con `.predict_proba(X_te)[:, class_idx]` y calcula la curva ROC.
4. Grafica todas las curvas juntas con la línea del clasificador aleatorio (AUC = 0.50) como referencia.

**Output:** *Setosa*: AUC = 1.00 en todos los modelos — perfectamente separable. *Versicolor* y *Virginica*: AUC entre 0.97 y 1.00, con leve variación entre modelos.

---

## Bloque 26 — Tabla comparativa final

```python
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('F1-macro', ascending=False).reset_index(drop=True)
df_resultados.index += 1
display(df_resultados)
```

**¿Qué hace?**
- Convierte la lista `resultados` (acumulada en todos los experimentos) en un DataFrame.
- Lo ordena de mayor a menor F1-macro.
- Resetea el índice y suma 1 para que empiece en 1 en lugar de 0.
- Muestra la tabla completa con todas las métricas comparadas lado a lado.

**Output:** Tabla con 11 modelos ordenados por F1-macro. Los 3 primeros (RF todas, RF pétalo, GB) empatan en 0.9667. El último (RF binarias) tiene F1-macro de 0.5556.

---

## Flujo general del notebook

```
Carga → EDA → PCA → Feature Engineering → Splits
   ↓
Experimentos (6.1 a 6.11): cada uno sigue el patrón:
   fit → predict → accuracy/F1 → confusion matrix → FP/FN → resultados.append()
   ↓
Curvas ROC (comparación visual de todos los modelos)
   ↓
Tabla comparativa final (pd.DataFrame de resultados)
```

Después de cada bloque de código hay una celda de markdown que explica el output, los números concretos obtenidos y la interpretación del resultado en el contexto del análisis progresivo.

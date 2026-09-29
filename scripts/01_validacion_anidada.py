# ======================================================================
# 01 -- Nested validation with spatial blocks (main analysis)
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Run from the repository root:
#     python scripts/01_validacion_anidada.py
#
# Inputs:  data/municipal_dengue_dataset_mexico_2022.csv
# Outputs: resultados/*.csv, resultados/*.png  (~1 h 20 min)
#
# The code and its comments are in Spanish, as written by the authors.
# Only the data and output paths were changed when the analysis was
# organised as a repository; see scripts/rutas.py.
# ======================================================================

# ================================================================
# VALIDACIÓN ANIDADA CON BLOQUES ESPACIALES
# PARA REGRESIÓN ORDINAL CON CORRECCIÓN POR KRIGING
#
#
# 1. Validación espacial:
#    se agrega un esquema externo de bloques geográficos
#    (k-means sobre coordenadas) y se reporta en paralelo
#    con el esquema aleatorio estratificado.
#
# 2. Separación entre selección y evaluación final:
#    la selección del modelo ordinal y la del variograma
#    ocurren en un nivel interno que solo usa el conjunto
#    de entrenamiento de cada partición externa.
#
# 3. Fuga de información:
#    la selección de variables por correlación de Spearman
#    se calcula dentro de cada conjunto de entrenamiento,
#    nunca sobre la muestra completa.
#
# Predictores comparados en cada partición externa:
#
#   Base        regla de la mediana sobre P(Y=k)
#   Base EV     redondeo de E[Y] sin kriging
#   Método A    kriging sobre residuos discretos  + Base
#   Método B    kriging sobre residuos continuos  + E[Y]
#
# La comparación Base EV frente a Base aísla el efecto
# de la regla de decisión.
# La comparación Método B frente a Base EV aísla el
# efecto espacial puro.
# ================================================================


# ================================================================
# 1. LIBRERÍAS
# ================================================================

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import rankdata, wilcoxon
from scipy.special import expit

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

from statsmodels.miscmodels.ordinal_model import OrderedModel

from mord import LogisticAT, LogisticIT

from xgboost import XGBClassifier

from pykrige.ok import OrdinaryKriging

from libpysal.weights import KNN
from esda.moran import Moran

import rutas


warnings.filterwarnings("ignore")


# ================================================================
# 2. CONFIGURACIÓN
# ================================================================

GLOBAL_SEED = 123

N_CLASSES = 5


# ------------------------------------------------
# Modo de prueba
#
# Ejecutar primero con MODO_PRUEBA = True
# para verificar que todo el flujo corre sin errores.
# La ejecución tarda aproximadmanete una hora 20 minutos.
# ------------------------------------------------

# Puede activarse sin editar este archivo:
#     ORK_MODO_PRUEBA=1 python scripts/01_validacion_anidada.py
MODO_PRUEBA = os.environ.get(
    "ORK_MODO_PRUEBA", "0"
) not in ("0", "", "false", "False")


# ------------------------------------------------
# Esquemas externos de evaluación
# ------------------------------------------------

ESQUEMAS = [
    "aleatorio",
    "espacial"
]


# Esquema aleatorio estratificado.
#
# Se usan semillas distintas a las del estudio previo
# para que los conjuntos de prueba no coincidan.

N_REPETICIONES_ALEATORIO = 50

SEMILLA_BASE_ALEATORIO = 5000

TEST_SIZE = 0.20


# Esquema espacial.
#
# Total de particiones externas = N_BLOQUES * N_REPETICIONES_ESPACIAL

N_BLOQUES = 5

N_REPETICIONES_ESPACIAL = 10

SEMILLA_BASE_ESPACIAL = 7000


# ------------------------------------------------
# Nivel interno
# ------------------------------------------------

N_FOLDS_INTERNO = 5

N_FOLDS_OOF = 5

N_FOLDS_VARIOGRAMA = 3


# ------------------------------------------------
# Selección de variables
# ------------------------------------------------

UMBRAL_SPEARMAN = 0.10

MIN_VARIABLES = 3

PODAR_COLINEALES = True

UMBRAL_COLINEALIDAD = 0.95


# ------------------------------------------------
# Kriging
# ------------------------------------------------

VARIOGRAMAS_CANDIDATOS = [
    "linear",
    "exponential",
    "gaussian",
    "spherical"
]

N_LAGS = 12

N_CLOSEST_POINTS = 50


# ------------------------------------------------
# Umbral de conversion a nivel ordinal
#
# El redondeo estandar equivale a un umbral de 0.5.
# Se calibra dentro del entrenamiento el umbral t tal
# que la prediccion es floor(valor continuo + t).
# ------------------------------------------------

REJILLA_UMBRAL = np.arange(0.05, 0.96, 0.01)


# ------------------------------------------------
# Analisis de sensibilidad a la codificacion
#
# La formulacion e_i = Y_i - mu_i supone espaciamiento
# uniforme entre categorias. Como las categorias provienen
# de cuartiles de la tasa, ese espaciamiento no es uniforme
# en la escala original.
#
# Se repite el procedimiento con una codificacion
# alternativa basada en la tasa mediana observada dentro de
# cada categoria, transformada con log(1+x) y reescalada al
# mismo rango. Los puntajes se estiman dentro de cada
# conjunto de entrenamiento.
# ------------------------------------------------

EJECUTAR_SENSIBILIDAD = True


# ------------------------------------------------
# Modelos de referencia
# ------------------------------------------------

K_VECINOS_REFERENCIA = 8


# ------------------------------------------------
# Moran
# ------------------------------------------------

K_MORAN = 8

MORAN_PERMUTATIONS = 999


# ------------------------------------------------
# Salida
# ------------------------------------------------

CARPETA_SALIDA = rutas.CARPETA_RESULTADOS

os.makedirs(
    CARPETA_SALIDA,
    exist_ok=True
)


# ------------------------------------------------
# Reducción automática en modo de prueba
# ------------------------------------------------

if MODO_PRUEBA:

    N_REPETICIONES_ALEATORIO = 2

    N_REPETICIONES_ESPACIAL = 1

    N_BLOQUES = 3

    N_FOLDS_INTERNO = 2

    N_FOLDS_OOF = 3

    N_FOLDS_VARIOGRAMA = 2

    MORAN_PERMUTATIONS = 99

    VARIOGRAMAS_CANDIDATOS = [
        "exponential",
        "spherical"
    ]

    print(
        "MODO DE PRUEBA ACTIVO: "
        "configuración reducida.\n"
    )


# ================================================================
# 3. DATOS DE ENTRADA
# ================================================================
#
# Se requieren cuatro objetos ya presentes en el notebook:
#
#   X_full        matriz (n, p) con TODAS las covariables
#                 candidatas, sin filtrar por Spearman.
#                 No debe contener Rate_2022, DRL,
#                 Nivel_Dengue ni variables derivadas
#                 de la respuesta.
#
#   y_ordinal     vector (n,) con niveles enteros 0 a 4.
#
#   coords        matriz (n, 2), columna 0 = longitud,
#                 columna 1 = latitud, en grados decimales.
#
#   feature_names lista con los p nombres de columna.
#
#
# 
# El filtrado por Spearman ocurre dentro del bucle.
#
# Si en el notebook la tabla de covariables candidatas se
# llama df_covariables, la conversión sería:
#
#   feature_names = list(df_covariables.columns)
#   X_full        = df_covariables.to_numpy(dtype=float)
#   y_ordinal     = df_modelo["DRL"].to_numpy().astype(int)
#   coords        = df_coordenadas[["lon", "lat"]].to_numpy(dtype=float)
# ================================================================

df = pd.read_csv(rutas.ARCHIVO_DATOS)

COLUMNAS_EXCLUIDAS = [
    "Estado",
    "Municipio",
    "Rate_2022",
    "lon_mun",
    "lat_mun",
    "DRL"
]

df_covariables = df.drop(
    columns=COLUMNAS_EXCLUIDAS
)

feature_names = list(
    df_covariables.columns
)

X_full = df_covariables.to_numpy(
    dtype=float
)

y_ordinal = df["DRL"].to_numpy().ravel().astype(int)

# Tasa original, necesaria unicamente para construir la
# codificacion alternativa del analisis de sensibilidad.

rate_full = df["Rate_2022"].to_numpy(dtype=float)

coords = df[["lon_mun", "lat_mun"]].to_numpy(
    dtype=float
)


X_full = np.asarray(
    X_full,
    dtype=float
)

y_ordinal = np.asarray(
    y_ordinal
).ravel().astype(int)

coords = np.asarray(
    coords,
    dtype=float
)


try:

    feature_names = list(
        feature_names
    )

except NameError:

    feature_names = [
        f"X{j}"
        for j in range(X_full.shape[1])
    ]


# ================================================================
# 4. VALIDACIONES DE ENTRADA
# ================================================================

if X_full.ndim != 2:

    raise ValueError(
        "X_full debe ser una matriz bidimensional."
    )


if coords.ndim != 2 or coords.shape[1] != 2:

    raise ValueError(
        "coords debe tener dimensiones (n, 2)."
    )


if not (
    len(X_full)
    == len(y_ordinal)
    == len(coords)
):

    raise ValueError(
        "X_full, y_ordinal y coords no tienen "
        "el mismo número de observaciones."
    )


if len(feature_names) != X_full.shape[1]:

    raise ValueError(
        "feature_names no coincide con el número "
        "de columnas de X_full."
    )


if not np.isfinite(X_full).all():

    raise ValueError(
        "X_full contiene NaN o valores infinitos."
    )


if not np.isfinite(coords).all():

    raise ValueError(
        "coords contiene NaN o valores infinitos."
    )


if not np.array_equal(
    np.unique(y_ordinal),
    np.arange(N_CLASSES)
):

    raise ValueError(
        "y_ordinal debe contener exactamente "
        "los niveles 0, 1, 2, 3 y 4."
    )


# Coordenadas duplicadas:
# el kriging con exact_values=True puede fallar.

_coords_unicas = np.unique(
    coords,
    axis=0
)

N_DUPLICADAS = (
    len(coords)
    -
    len(_coords_unicas)
)


print("=" * 90)
print("COMPROBACIÓN DE DATOS")
print("=" * 90)

print(
    "Observaciones:",
    X_full.shape[0]
)

print(
    "Covariables candidatas:",
    X_full.shape[1]
)

print(
    "Coordenadas duplicadas:",
    N_DUPLICADAS
)

print()

print("Distribución de niveles:")

print(
    pd.Series(y_ordinal)
    .value_counts()
    .sort_index()
)

print()


if N_DUPLICADAS > 0:

    print(
        "Advertencia: existen coordenadas duplicadas. "
        "El kriging exacto puede fallar en algunas particiones."
    )
    print()


# ================================================================
# 5. MÉTRICAS ORDINALES
# ================================================================

METRIC_NAMES = [
    "MAE",
    "MZE",
    "Accuracy",
    "Dentro ±1",
    "Dentro ±2",
    "Subestimación",
    "Sobreestimación"
]

N_METRICS = len(METRIC_NAMES)


def calcular_metricas_ordinales(
    y_true,
    y_pred
):
    """
    Devuelve un vector con las siete métricas ordinales
    en el orden de METRIC_NAMES.
    """

    y_true = np.asarray(
        y_true
    ).ravel().astype(int)

    y_pred = np.asarray(
        y_pred
    ).ravel().astype(int)

    y_pred = np.clip(
        y_pred,
        0,
        N_CLASSES - 1
    )

    error_signed = y_pred - y_true

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    return np.array([
        mean_absolute_error(y_true, y_pred),
        1.0 - accuracy,
        accuracy,
        np.mean(np.abs(error_signed) <= 1),
        np.mean(np.abs(error_signed) <= 2),
        np.mean(error_signed < 0),
        np.mean(error_signed > 0)
    ])


# ================================================================
# 6. SELECCIÓN DE VARIABLES DENTRO DEL ENTRENAMIENTO
# ================================================================

def _spearman_columnas(
    X,
    y
):
    """
    Correlación de Spearman entre cada columna de X y el
    vector y, calculada como correlación de Pearson sobre
    los rangos. Equivalente a scipy.stats.spearmanr pero
    vectorizada.

    Las columnas constantes reciben rho = 0.
    """

    n, p = X.shape

    rangos_X = np.apply_along_axis(
        rankdata,
        0,
        X
    )

    rangos_y = rankdata(y)

    Xc = rangos_X - rangos_X.mean(axis=0)

    yc = rangos_y - rangos_y.mean()

    sx = np.sqrt(
        np.sum(Xc ** 2, axis=0)
    )

    sy = np.sqrt(
        np.sum(yc ** 2)
    )

    denominador = sx * sy

    rho = np.zeros(p)

    valido = denominador > 0

    rho[valido] = (
        (Xc[:, valido] * yc[:, None]).sum(axis=0)
        /
        denominador[valido]
    )

    return rho


def seleccionar_variables(
    X_train,
    y_train,
    umbral=UMBRAL_SPEARMAN,
    min_variables=MIN_VARIABLES,
    podar_colineales=PODAR_COLINEALES,
    umbral_colinealidad=UMBRAL_COLINEALIDAD
):
    """
    Devuelve los índices de las columnas seleccionadas.

    El criterio es |rho de Spearman con y| >= umbral,
    calculado exclusivamente con el conjunto de
    entrenamiento recibido.

    Si ninguna columna supera el umbral se conservan las
    min_variables de mayor magnitud, para que el ajuste
    no quede sin predictores.

    Opcionalmente se podan variables casi colineales entre
    sí, conservando en cada par la de mayor |rho| con y.
    Esto estabiliza los modelos paramétricos.
    """

    rho = _spearman_columnas(
        X_train,
        y_train
    )

    magnitud = np.abs(rho)

    seleccion = np.where(
        magnitud >= umbral
    )[0]

    if seleccion.size < min_variables:

        seleccion = np.argsort(
            -magnitud
        )[:min_variables]

    # Orden descendente por magnitud de correlación

    seleccion = seleccion[
        np.argsort(-magnitud[seleccion])
    ]

    if not podar_colineales or seleccion.size <= 1:

        return np.sort(seleccion)

    # ------------------------------------------------
    # Poda de colinealidad sobre las variables ya
    # seleccionadas, usando solo el entrenamiento.
    # ------------------------------------------------

    sub = X_train[:, seleccion]

    rangos_sub = np.apply_along_axis(
        rankdata,
        0,
        sub
    )

    with np.errstate(invalid="ignore"):

        matriz = np.corrcoef(
            rangos_sub,
            rowvar=False
        )

    matriz = np.nan_to_num(
        matriz,
        nan=0.0
    )

    conservadas = []

    for posicion in range(seleccion.size):

        redundante = False

        for previa in conservadas:

            if abs(matriz[posicion, previa]) >= umbral_colinealidad:

                redundante = True

                break

        if not redundante:

            conservadas.append(posicion)

    finales = seleccion[conservadas]

    if finales.size == 0:

        finales = seleccion[:1]

    return np.sort(finales)


# ================================================================
# 7. MODELOS ORDINALES CON INTERFAZ COMÚN
#
# Todos exponen:
#
#   fit(X, y)
#   predict_class_probabilities(X) -> (n, N_CLASSES)
#   predict(X)                     -> regla de la mediana
#   predict_expected_value(X)      -> E[Y|X] continuo
#
# La regla de la mediana se aplica de forma uniforme a los
# cinco modelos porque el MAE es el criterio principal y la
# mediana es el predictor óptimo bajo pérdida absoluta.
# ================================================================

class BaseOrdinalModel:

    def __init__(
        self,
        n_classes=N_CLASSES
    ):

        self.n_classes = n_classes

        self.classes_ = None


    # --------------------------------------------------------
    # A implementar por cada subclase
    # --------------------------------------------------------

    def fit(
        self,
        X,
        y
    ):

        raise NotImplementedError


    def predict_class_probabilities(
        self,
        X
    ):

        raise NotImplementedError


    # --------------------------------------------------------
    # Derivados comunes
    # --------------------------------------------------------

    def predict_cumulative_greater(
        self,
        X
    ):
        """
        Devuelve P(Y>0), ..., P(Y>K-2) a partir de las
        probabilidades por clase.
        """

        P = self.predict_class_probabilities(X)

        # acumulada[:, k] = P(Y >= k)

        acumulada = np.cumsum(
            P[:, ::-1],
            axis=1
        )[:, ::-1]

        return acumulada[:, 1:]


    def predict(
        self,
        X
    ):
        """
        Regla de la mediana:
        menor k tal que P(Y<=k) >= 0.5.
        """

        q = self.predict_cumulative_greater(X)

        return np.sum(
            q >= 0.5,
            axis=1
        ).astype(int)


    def predict_mode(
        self,
        X
    ):

        P = self.predict_class_probabilities(X)

        return np.argmax(
            P,
            axis=1
        ).astype(int)


    def predict_expected_value(
        self,
        X
    ):
        """
        E[Y|X] = sum_k k P(Y=k|X) = sum_k P(Y>k|X).
        """

        q = self.predict_cumulative_greater(X)

        return q.sum(axis=1).astype(float)


    # --------------------------------------------------------
    # Utilidades de mapeo de clases
    # --------------------------------------------------------

    def _mapear_y(
        self,
        y
    ):
        """
        Convierte las etiquetas observadas a enteros
        consecutivos 0..m-1 y guarda la correspondencia.
        """

        y = np.asarray(y).ravel().astype(int)

        self.classes_ = np.unique(y)

        return np.searchsorted(
            self.classes_,
            y
        )


    def _expandir(
        self,
        P_obs
    ):
        """
        Coloca las probabilidades de las clases observadas
        en la rejilla completa 0..N_CLASSES-1 y renormaliza.
        """

        P_obs = np.asarray(
            P_obs,
            dtype=float
        )

        if P_obs.ndim == 1:

            P_obs = P_obs.reshape(-1, 1)

        n = P_obs.shape[0]

        P = np.zeros(
            (n, self.n_classes),
            dtype=float
        )

        for columna, clase in enumerate(self.classes_):

            if columna < P_obs.shape[1]:

                P[:, int(clase)] = P_obs[:, columna]

        P = np.clip(P, 0.0, 1.0)

        suma = P.sum(
            axis=1,
            keepdims=True
        )

        suma[suma == 0] = 1.0

        return P / suma


# ----------------------------------------------------------------
# 7.1 Clasificador ordinal acumulativo (Random Forest, XGBoost)
# ----------------------------------------------------------------

class CumulativeOrdinalClassifier(BaseOrdinalModel):
    """
    Ajusta K-1 clasificadores binarios para Y > k y
    reconstruye las probabilidades por clase imponiendo
    monotonía sobre las probabilidades acumuladas.
    """

    def __init__(
        self,
        base_estimator,
        n_classes=N_CLASSES
    ):

        super().__init__(n_classes=n_classes)

        self.base_estimator = base_estimator

        self.models_ = []

        self.umbrales_degenerados_ = 0


    def fit(
        self,
        X,
        y
    ):

        from sklearn.base import clone

        X = np.asarray(X, dtype=float)

        y = np.asarray(y).ravel().astype(int)

        self.classes_ = np.arange(self.n_classes)

        self.models_ = []

        self.umbrales_degenerados_ = 0

        for umbral in range(self.n_classes - 1):

            y_binaria = (y > umbral).astype(int)

            clases_presentes = np.unique(y_binaria)

            if clases_presentes.size < 2:

                # Subproblema degenerado: no hay dos clases.
                # Se guarda la constante observada.

                self.umbrales_degenerados_ += 1

                self.models_.append(
                    ("constante", float(clases_presentes[0]))
                )

                continue

            modelo = clone(self.base_estimator)

            modelo.fit(X, y_binaria)

            self.models_.append(
                ("modelo", modelo)
            )

        return self


    def predict_threshold_probabilities(
        self,
        X
    ):

        X = np.asarray(X, dtype=float)

        n = X.shape[0]

        columnas = []

        for tipo, objeto in self.models_:

            if tipo == "constante":

                columnas.append(
                    np.full(n, objeto, dtype=float)
                )

                continue

            probabilidades = objeto.predict_proba(X)

            posiciones = {
                int(etiqueta): posicion
                for posicion, etiqueta
                in enumerate(objeto.classes_)
            }

            if 1 in posiciones:

                columnas.append(
                    probabilidades[:, posiciones[1]]
                )

            else:

                columnas.append(
                    np.zeros(n, dtype=float)
                )

        q = np.column_stack(columnas)

        # Consistencia ordinal:
        # P(Y>0) >= P(Y>1) >= ... >= P(Y>K-2)

        q = np.minimum.accumulate(
            q,
            axis=1
        )

        return np.clip(q, 0.0, 1.0)


    def predict_class_probabilities(
        self,
        X
    ):

        q = self.predict_threshold_probabilities(X)

        n = q.shape[0]

        P = np.zeros(
            (n, self.n_classes),
            dtype=float
        )

        P[:, 0] = 1.0 - q[:, 0]

        for k in range(1, self.n_classes - 1):

            P[:, k] = q[:, k - 1] - q[:, k]

        P[:, self.n_classes - 1] = q[:, -1]

        P = np.clip(P, 0.0, 1.0)

        suma = P.sum(
            axis=1,
            keepdims=True
        )

        suma[suma == 0] = 1.0

        return P / suma


# ----------------------------------------------------------------
# 7.2 Logit ordinal de odds proporcionales
# ----------------------------------------------------------------

class OrderedLogitOrdinal(BaseOrdinalModel):
    """
    Envoltura de statsmodels.OrderedModel con
    estandarización interna.

    OrderedModel no debe recibir columna constante.
    """

    def __init__(
        self,
        n_classes=N_CLASSES,
        maxiter=1000
    ):

        super().__init__(n_classes=n_classes)

        self.maxiter = maxiter

        self.scaler_ = None

        self.resultado_ = None


    def fit(
        self,
        X,
        y
    ):

        X = np.asarray(X, dtype=float)

        y_mapeada = self._mapear_y(y)

        if self.classes_.size < 2:

            raise ValueError(
                "OrderedModel requiere al menos dos clases."
            )

        self.scaler_ = StandardScaler().fit(X)

        X_esc = self.scaler_.transform(X)

        modelo = OrderedModel(
            endog=y_mapeada,
            exog=X_esc,
            distr="logit"
        )

        ultimo_error = None

        for metodo in ("bfgs", "lbfgs", "nm"):

            try:

                self.resultado_ = modelo.fit(
                    method=metodo,
                    maxiter=self.maxiter,
                    disp=False
                )

                return self

            except Exception as error:

                ultimo_error = error

        raise RuntimeError(
            f"OrderedModel no convergió: {ultimo_error}"
        )


    def predict_class_probabilities(
        self,
        X
    ):

        X_esc = self.scaler_.transform(
            np.asarray(X, dtype=float)
        )

        P_obs = self.resultado_.model.predict(
            self.resultado_.params,
            exog=X_esc,
            which="prob"
        )

        return self._expandir(
            np.asarray(P_obs)
        )


# ----------------------------------------------------------------
# 7.3 Modelos de umbral de mord
# ----------------------------------------------------------------

class MordOrdinal(BaseOrdinalModel):
    """
    Envoltura de mord.LogisticAT y mord.LogisticIT con
    estandarización interna.

    Se usa predict_proba cuando está disponible. Si la
    versión instalada no lo expone, las probabilidades se
    reconstruyen a partir de coef_ y theta_ bajo la
    parametrización acumulativa P(Y<=k) = sigma(theta_k - Xw).
    """

    def __init__(
        self,
        estimador,
        n_classes=N_CLASSES
    ):

        super().__init__(n_classes=n_classes)

        self.estimador = estimador

        self.modelo_ = None

        self.scaler_ = None


    def fit(
        self,
        X,
        y
    ):

        from sklearn.base import clone

        X = np.asarray(X, dtype=float)

        y_mapeada = self._mapear_y(y)

        if self.classes_.size < 2:

            raise ValueError(
                "mord requiere al menos dos clases."
            )

        self.scaler_ = StandardScaler().fit(X)

        X_esc = self.scaler_.transform(X)

        self.modelo_ = clone(self.estimador)

        self.modelo_.fit(
            X_esc,
            y_mapeada
        )

        return self


    def predict_class_probabilities(
        self,
        X
    ):

        X_esc = self.scaler_.transform(
            np.asarray(X, dtype=float)
        )

        P_obs = None

        if hasattr(self.modelo_, "predict_proba"):

            try:

                P_obs = np.asarray(
                    self.modelo_.predict_proba(X_esc),
                    dtype=float
                )

            except Exception:

                P_obs = None

        if P_obs is None:

            # Reconstrucción a partir de los umbrales

            eta = X_esc.dot(self.modelo_.coef_)

            theta = np.asarray(
                self.modelo_.theta_,
                dtype=float
            )

            # acumulada[:, k] = P(Y <= k), k = 0..m-2

            acumulada = expit(
                theta[None, :] - eta[:, None]
            )

            acumulada = np.maximum.accumulate(
                acumulada,
                axis=1
            )

            unos = np.ones(
                (acumulada.shape[0], 1)
            )

            ceros = np.zeros(
                (acumulada.shape[0], 1)
            )

            completa = np.hstack(
                [ceros, acumulada, unos]
            )

            P_obs = np.diff(
                completa,
                axis=1
            )

        return self._expandir(P_obs)


# ----------------------------------------------------------------
# 7.4 Fábrica de los cinco modelos
# ----------------------------------------------------------------

MODEL_NAMES = [
    "Ordered Logit",
    "Logistic All-Threshold",
    "Logistic Immediate-Threshold",
    "Ordinal Random Forest",
    "Ordinal XGBoost"
]


def construir_modelos(
    random_state
):
    """
    Devuelve un diccionario con instancias nuevas de los
    cinco modelos candidatos.

    El desbalance se trata de forma equivalente en los dos
    modelos de árbol: class_weight balanceado en Random
    Forest y ponderación implícita equivalente en XGBoost
    mediante scale_pos_weight calculado dentro de cada
    subproblema binario no es posible con clone, por lo que
    se usa la opción homogénea de no reponderar en ninguno
    y se documenta esta decisión.
    """

    base_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1
    )

    base_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1
    )

    return {

        "Ordered Logit":
            OrderedLogitOrdinal(),

        "Logistic All-Threshold":
            MordOrdinal(LogisticAT(alpha=1.0)),

        "Logistic Immediate-Threshold":
            MordOrdinal(LogisticIT(alpha=1.0)),

        "Ordinal Random Forest":
            CumulativeOrdinalClassifier(base_rf),

        "Ordinal XGBoost":
            CumulativeOrdinalClassifier(base_xgb)
    }


# ================================================================
# 8. UTILIDADES DE KRIGING
# ================================================================

def ajustar_kriging(
    coords_train,
    valores,
    modelo_variograma
):

    return OrdinaryKriging(
        coords_train[:, 0],
        coords_train[:, 1],
        valores,
        variogram_model=modelo_variograma,
        nlags=N_LAGS,
        weight=True,
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic",
        exact_values=True,
        pseudo_inv=True
    )


def predecir_kriging(
    modelo_ok,
    coords_pred,
    n_train
):

    n_cercanos = int(
        min(
            N_CLOSEST_POINTS,
            max(1, n_train - 1)
        )
    )

    z, _ = modelo_ok.execute(
        style="points",
        xpoints=coords_pred[:, 0],
        ypoints=coords_pred[:, 1],
        backend="loop",
        n_closest_points=n_cercanos
    )

    return np.asarray(z).ravel().astype(float)


def construir_escala(
    y_train,
    rate_train
):
    """
    Construye los puntajes representativos de cada categoria
    a partir de la tasa mediana observada dentro de ella,
    transformada con log(1+x) y reescalada al intervalo
    [0, K-1] para que la magnitud sea comparable con la
    codificacion uniforme.

    Se estima unicamente con datos de entrenamiento.

    La codificacion uniforme corresponde a los enteros
    0, 1, ..., K-1. Esta alternativa refleja que las
    categorias provienen de cuartiles de la tasa y que, por
    tanto, no estan igualmente espaciadas en la escala
    original.
    """

    puntajes = np.full(N_CLASSES, np.nan)

    for k in range(N_CLASSES):

        mascara = y_train == k

        if mascara.sum() > 0:

            puntajes[k] = np.log1p(
                np.median(rate_train[mascara])
            )

    posiciones = np.arange(N_CLASSES, dtype=float)

    observados = np.isfinite(puntajes)

    if observados.sum() < 2:

        return posiciones

    puntajes = np.interp(
        posiciones,
        posiciones[observados],
        puntajes[observados]
    )

    # Monotonia estricta

    for k in range(1, N_CLASSES):

        if puntajes[k] <= puntajes[k - 1]:

            puntajes[k] = puntajes[k - 1] + 1e-6

    rango = puntajes[-1] - puntajes[0]

    if rango <= 0:

        return posiciones

    return (
        (puntajes - puntajes[0])
        /
        rango
        *
        (N_CLASSES - 1)
    )


def escala_a_posicion(
    valores,
    puntajes
):
    """
    Convierte un puntaje continuo a la escala de posiciones
    0..K-1 mediante interpolacion lineal por tramos, con
    extrapolacion lineal fuera del rango.

    Esto permite reutilizar la calibracion del umbral sobre
    una escala comun, con independencia de la codificacion
    empleada para construir los residuos.
    """

    v = np.asarray(valores, dtype=float)

    posiciones = np.arange(
        len(puntajes),
        dtype=float
    )

    resultado = np.interp(
        v,
        puntajes,
        posiciones
    )

    izquierda = v < puntajes[0]

    derecha = v > puntajes[-1]

    if np.any(izquierda):

        ancho = puntajes[1] - puntajes[0]

        if ancho > 0:

            resultado[izquierda] = (
                (v[izquierda] - puntajes[0])
                /
                ancho
            )

    if np.any(derecha):

        ancho = puntajes[-1] - puntajes[-2]

        if ancho > 0:

            resultado[derecha] = (
                (N_CLASSES - 1)
                +
                (v[derecha] - puntajes[-1])
                /
                ancho
            )

    return resultado


def prediccion_trivial(
    y_train,
    n_test
):
    """
    Modelo de referencia sin covariables: predice siempre la
    mediana del entrenamiento, que es el predictor constante
    optimo bajo perdida absoluta.
    """

    constante = int(
        np.median(y_train)
    )

    return np.full(
        n_test,
        constante,
        dtype=int
    )


def prediccion_knn_espacial(
    coords_train,
    y_train,
    coords_test,
    k
):
    """
    Modelo de referencia que usa solo la geografia: predice
    la mediana de los k municipios de entrenamiento mas
    cercanos.

    Sirve para cuantificar cuanto del desempeno se explica
    por la posicion espacial sin recurrir a las covariables.
    """

    k_efectivo = int(
        min(k, len(y_train))
    )

    buscador = NearestNeighbors(
        n_neighbors=k_efectivo
    ).fit(coords_train)

    _, indices = buscador.kneighbors(coords_test)

    return np.clip(
        np.round(
            np.median(
                y_train[indices],
                axis=1
            )
        ),
        0,
        N_CLASSES - 1
    ).astype(int)


def kriging_oof_predicciones(
    coords_train,
    residuos,
    modelo_variograma,
    n_folds,
    semilla
):
    """
    Devuelve predicciones krigeadas fuera de muestra en los
    propios puntos de entrenamiento.

    Se necesitan para calibrar el umbral de conversion sin
    usar el conjunto de prueba. No pueden obtenerse del
    kriging ajustado sobre todo el entrenamiento, porque con
    exact_values=True la prediccion en un punto observado
    reproduce su residuo y la calibracion seria circular.
    """

    n = len(residuos)

    n_folds_efectivo = int(
        min(n_folds, max(2, n // 10))
    )

    kf = KFold(
        n_splits=n_folds_efectivo,
        shuffle=True,
        random_state=semilla
    )

    prediccion = np.full(n, np.nan)

    for idx_ajuste, idx_valida in kf.split(coords_train):

        try:

            modelo_ok = ajustar_kriging(
                coords_train[idx_ajuste],
                residuos[idx_ajuste],
                modelo_variograma
            )

            prediccion[idx_valida] = predecir_kriging(
                modelo_ok,
                coords_train[idx_valida],
                len(idx_ajuste)
            )

        except Exception:

            continue

    return prediccion


def calibrar_umbral(
    valores_continuos,
    y_true,
    rejilla=None
):
    """
    Busca el umbral t que minimiza el MAE de la conversion
    floor(valor continuo + t) sobre el conjunto recibido.

    El redondeo estandar corresponde a t = 0.5, que no es
    optimo bajo perdida absoluta cuando la distribucion del
    valor esperado esta sesgada.

    Ante empates se devuelve el punto medio de la meseta
    optima, que es mas estable que el primer valor.

    Se invoca unicamente con datos de entrenamiento.
    """

    if rejilla is None:

        rejilla = REJILLA_UMBRAL

    valores = np.asarray(
        valores_continuos,
        dtype=float
    ).ravel()

    y_true = np.asarray(
        y_true
    ).ravel().astype(int)

    validos = np.isfinite(valores)

    if validos.sum() < 20:

        return 0.5

    v = valores[validos]

    objetivo = y_true[validos]

    errores = np.empty(len(rejilla))

    for posicion, t in enumerate(rejilla):

        prediccion = np.clip(
            np.floor(v + t),
            0,
            N_CLASSES - 1
        )

        errores[posicion] = np.mean(
            np.abs(prediccion - objetivo)
        )

    minimo = errores.min()

    empatados = rejilla[
        errores <= minimo + 1e-12
    ]

    return float(np.median(empatados))


def seleccionar_variograma(
    coords_train,
    residuos,
    candidatos,
    n_folds,
    semilla
):
    """
    Elige el modelo de variograma por validación cruzada del
    propio kriging sobre los residuos de entrenamiento.

    El criterio es la raíz del error cuadrático medio en la
    predicción de los residuos retenidos. No interviene en
    ningún momento el conjunto de prueba externo.
    """

    n = len(residuos)

    n_folds_efectivo = int(
        min(n_folds, max(2, n // 10))
    )

    kf = KFold(
        n_splits=n_folds_efectivo,
        shuffle=True,
        random_state=semilla
    )

    rmse_por_modelo = {}

    for nombre_variograma in candidatos:

        errores = []

        exito = True

        for idx_ajuste, idx_valida in kf.split(coords_train):

            try:

                modelo_ok = ajustar_kriging(
                    coords_train[idx_ajuste],
                    residuos[idx_ajuste],
                    nombre_variograma
                )

                prediccion = predecir_kriging(
                    modelo_ok,
                    coords_train[idx_valida],
                    len(idx_ajuste)
                )

                errores.append(
                    residuos[idx_valida] - prediccion
                )

            except Exception:

                exito = False

                break

        if exito and errores:

            todos = np.concatenate(errores)

            todos = todos[np.isfinite(todos)]

            if todos.size > 0:

                rmse_por_modelo[nombre_variograma] = float(
                    np.sqrt(np.mean(todos ** 2))
                )

    if not rmse_por_modelo:

        return None, {}

    mejor = min(
        rmse_por_modelo,
        key=rmse_por_modelo.get
    )

    return mejor, rmse_por_modelo


def parametros_variograma(
    modelo_ok
):
    """
    Devuelve sill, rango y nugget cuando están disponibles.
    """

    try:

        parametros = np.asarray(
            modelo_ok.variogram_model_parameters,
            dtype=float
        ).ravel()

        if parametros.size >= 3:

            return (
                float(parametros[0]),
                float(parametros[1]),
                float(parametros[2])
            )

        if parametros.size == 2:

            return (
                float(parametros[0]),
                np.nan,
                float(parametros[1])
            )

    except Exception:

        pass

    return (np.nan, np.nan, np.nan)


# ================================================================
# 9. PARTICIONES EXTERNAS
# ================================================================

def generar_particiones_aleatorias(
    y,
    n_repeticiones,
    semilla_base,
    test_size
):
    """
    Particiones aleatorias estratificadas.
    """

    indices = np.arange(len(y))

    particiones = []

    for repeticion in range(n_repeticiones):

        idx_train, idx_test = train_test_split(
            indices,
            test_size=test_size,
            random_state=semilla_base + repeticion,
            stratify=y
        )

        particiones.append({
            "esquema": "aleatorio",
            "repeticion": repeticion,
            "bloque": -1,
            "idx_train": idx_train,
            "idx_test": idx_test
        })

    return particiones


def generar_particiones_espaciales(
    coordenadas,
    n_bloques,
    n_repeticiones,
    semilla_base
):
    """
    Bloques geográficos construidos con k-means sobre las
    coordenadas. En cada partición un bloque completo se
    retiene como conjunto de prueba, de modo que ningún
    municipio vecino del bloque queda en entrenamiento.

    El agrupamiento se repite con semillas distintas para
    no depender de una configuración particular.
    """

    indices = np.arange(len(coordenadas))

    particiones = []

    for repeticion in range(n_repeticiones):

        kmeans = KMeans(
            n_clusters=n_bloques,
            n_init=10,
            random_state=semilla_base + repeticion
        )

        etiquetas = kmeans.fit_predict(coordenadas)

        for bloque in range(n_bloques):

            mascara_test = etiquetas == bloque

            idx_test = indices[mascara_test]

            idx_train = indices[~mascara_test]

            if idx_test.size < 5 or idx_train.size < 30:

                continue

            particiones.append({
                "esquema": "espacial",
                "repeticion": repeticion,
                "bloque": bloque,
                "idx_train": idx_train,
                "idx_test": idx_test
            })

    return particiones


# ================================================================
# 10. NIVEL INTERNO: SELECCIÓN DEL MODELO ORDINAL
# ================================================================

def seleccionar_modelo(
    X_train,
    y_train,
    n_folds,
    semilla
):
    """
    Compara los cinco modelos candidatos mediante validación
    cruzada interna sobre el conjunto de entrenamiento.

    La selección de variables se repite dentro de cada
    pliegue interno, de modo que lo que se compara es el
    procedimiento completo y no un conjunto fijo de
    predictores.

    Devuelve el nombre ganador, la tabla de MAE interno y el
    número de pliegues en que cada modelo pudo ajustarse.
    """

    conteo_clases = np.bincount(
        y_train,
        minlength=N_CLASSES
    )

    minimo = conteo_clases[conteo_clases > 0].min()

    n_folds_efectivo = int(
        max(2, min(n_folds, minimo))
    )

    if minimo >= n_folds_efectivo:

        divisor = StratifiedKFold(
            n_splits=n_folds_efectivo,
            shuffle=True,
            random_state=semilla
        )

        generador = divisor.split(X_train, y_train)

    else:

        divisor = KFold(
            n_splits=n_folds_efectivo,
            shuffle=True,
            random_state=semilla
        )

        generador = divisor.split(X_train)

    acumulado = {
        nombre: []
        for nombre in MODEL_NAMES
    }

    for numero_fold, (idx_ajuste, idx_valida) in enumerate(
        generador,
        start=1
    ):

        X_ajuste = X_train[idx_ajuste]

        y_ajuste = y_train[idx_ajuste]

        X_valida = X_train[idx_valida]

        y_valida = y_train[idx_valida]

        if np.unique(y_ajuste).size < 2:

            continue

        columnas = seleccionar_variables(
            X_ajuste,
            y_ajuste
        )

        X_ajuste_sel = X_ajuste[:, columnas]

        X_valida_sel = X_valida[:, columnas]

        modelos = construir_modelos(
            random_state=semilla + numero_fold
        )

        for nombre, modelo in modelos.items():

            try:

                modelo.fit(
                    X_ajuste_sel,
                    y_ajuste
                )

                prediccion = modelo.predict(
                    X_valida_sel
                )

                acumulado[nombre].append(
                    mean_absolute_error(
                        y_valida,
                        prediccion
                    )
                )

            except Exception:

                # El modelo no pudo ajustarse en este
                # pliegue. Se registra como no disponible
                # en lugar de penalizarlo con un valor
                # arbitrario.

                continue

    resumen = {}

    for nombre in MODEL_NAMES:

        valores = acumulado[nombre]

        resumen[nombre] = {
            "mae_interno": (
                float(np.mean(valores))
                if valores
                else np.nan
            ),
            "folds_validos": len(valores)
        }

    disponibles = {
        nombre: datos["mae_interno"]
        for nombre, datos in resumen.items()
        if np.isfinite(datos["mae_interno"])
    }

    if not disponibles:

        raise RuntimeError(
            "Ningún modelo pudo ajustarse en el nivel interno."
        )

    ganador = min(
        disponibles,
        key=disponibles.get
    )

    return ganador, resumen


# ================================================================
# 11. RESIDUOS FUERA DE MUESTRA SOBRE EL ENTRENAMIENTO
# ================================================================

def calcular_residuos_oof(
    X_train_sel,
    y_train,
    nombre_modelo,
    n_folds,
    semilla
):
    """
    Genera predicciones fuera de muestra dentro del conjunto
    de entrenamiento externo, necesarias para que el kriging
    se ajuste sobre residuos honestos y no sobre residuos de
    ajuste, que serían artificialmente pequeños.

    Devuelve el residuo discreto (Método A) y el residuo
    continuo (Método B).
    """

    conteo_clases = np.bincount(
        y_train,
        minlength=N_CLASSES
    )

    minimo = conteo_clases[conteo_clases > 0].min()

    n_folds_efectivo = int(
        max(2, min(n_folds, minimo))
    )

    if minimo >= n_folds_efectivo:

        divisor = StratifiedKFold(
            n_splits=n_folds_efectivo,
            shuffle=True,
            random_state=semilla
        )

        generador = divisor.split(X_train_sel, y_train)

    else:

        divisor = KFold(
            n_splits=n_folds_efectivo,
            shuffle=True,
            random_state=semilla
        )

        generador = divisor.split(X_train_sel)

    pred_discreta = np.full(
        len(y_train),
        np.nan
    )

    pred_esperada = np.full(
        len(y_train),
        np.nan
    )

    for numero_fold, (idx_ajuste, idx_valida) in enumerate(
        generador,
        start=1
    ):

        if np.unique(y_train[idx_ajuste]).size < 2:

            continue

        modelo = construir_modelos(
            random_state=semilla + numero_fold
        )[nombre_modelo]

        modelo.fit(
            X_train_sel[idx_ajuste],
            y_train[idx_ajuste]
        )

        pred_discreta[idx_valida] = modelo.predict(
            X_train_sel[idx_valida]
        )

        pred_esperada[idx_valida] = modelo.predict_expected_value(
            X_train_sel[idx_valida]
        )

    completos = (
        np.isfinite(pred_discreta)
        &
        np.isfinite(pred_esperada)
    )

    if completos.sum() < 30:

        raise RuntimeError(
            "Muy pocas predicciones fuera de muestra válidas."
        )

    residuo_discreto = (
        y_train.astype(float)
        -
        pred_discreta
    )

    residuo_continuo = (
        y_train.astype(float)
        -
        pred_esperada
    )

    return residuo_discreto, residuo_continuo, completos


# ================================================================
# 12. MORAN SOBRE LOS RESIDUOS DE ENTRENAMIENTO
# ================================================================

def calcular_moran(
    coordenadas,
    valores
):

    n = len(valores)

    k_efectivo = int(
        min(K_MORAN, max(1, n - 1))
    )

    try:

        pesos = KNN.from_array(
            coordenadas,
            k=k_efectivo
        )

        pesos.transform = "R"

        resultado = Moran(
            valores,
            pesos,
            permutations=MORAN_PERMUTATIONS
        )

        return (
            float(resultado.I),
            float(resultado.p_sim),
            float(resultado.z_sim)
        )

    except Exception:

        return (np.nan, np.nan, np.nan)


# ================================================================
# 13. CICLO PRINCIPAL
# ================================================================

particiones = []

if "aleatorio" in ESQUEMAS:

    particiones += generar_particiones_aleatorias(
        y_ordinal,
        N_REPETICIONES_ALEATORIO,
        SEMILLA_BASE_ALEATORIO,
        TEST_SIZE
    )

if "espacial" in ESQUEMAS:

    particiones += generar_particiones_espaciales(
        coords,
        N_BLOQUES,
        N_REPETICIONES_ESPACIAL,
        SEMILLA_BASE_ESPACIAL
    )


N_PARTICIONES = len(particiones)

print("=" * 90)
print("PLAN DE EVALUACIÓN")
print("=" * 90)

print(
    "Particiones externas totales:",
    N_PARTICIONES
)

print(
    "  aleatorias:",
    sum(
        1 for p in particiones
        if p["esquema"] == "aleatorio"
    )
)

print(
    "  espaciales:",
    sum(
        1 for p in particiones
        if p["esquema"] == "espacial"
    )
)

print()


registros = []

predicciones_detalle = []

frecuencia_variables = np.zeros(
    X_full.shape[1],
    dtype=int
)

particiones_completadas = 0

errores_ejecucion = []

ruta_parcial = os.path.join(
    CARPETA_SALIDA,
    "resultados_parciales.csv"
)

tiempo_inicio = time.time()


for numero, particion in enumerate(particiones, start=1):

    etiqueta = (
        f"{particion['esquema']}"
        f" r{particion['repeticion']}"
        f" b{particion['bloque']}"
    )

    print(
        f"[{numero}/{N_PARTICIONES}] {etiqueta}",
        flush=True
    )

    try:

        idx_train = particion["idx_train"]

        idx_test = particion["idx_test"]

        X_train_completo = X_full[idx_train]

        X_test_completo = X_full[idx_test]

        y_train = y_ordinal[idx_train]

        y_test = y_ordinal[idx_test]

        coords_train = coords[idx_train]

        coords_test = coords[idx_test]

        if np.unique(y_train).size < 2:

            raise RuntimeError(
                "El entrenamiento contiene una sola clase."
            )

        semilla_particion = (
            GLOBAL_SEED
            +
            numero * 100
        )

        # ------------------------------------------------
        # 13.1 Nivel interno: selección del modelo
        # ------------------------------------------------

        nombre_modelo, resumen_interno = seleccionar_modelo(
            X_train_completo,
            y_train,
            N_FOLDS_INTERNO,
            semilla_particion
        )

        # ------------------------------------------------
        # 13.2 Selección de variables sobre todo el
        #      entrenamiento externo
        # ------------------------------------------------

        columnas = seleccionar_variables(
            X_train_completo,
            y_train
        )

        frecuencia_variables[columnas] += 1

        X_train_sel = X_train_completo[:, columnas]

        X_test_sel = X_test_completo[:, columnas]

        # ------------------------------------------------
        # 13.3 Residuos fuera de muestra
        # ------------------------------------------------

        (
            residuo_A,
            residuo_B,
            completos
        ) = calcular_residuos_oof(
            X_train_sel,
            y_train,
            nombre_modelo,
            N_FOLDS_OOF,
            semilla_particion
        )

        coords_oof = coords_train[completos]

        residuo_A = residuo_A[completos]

        residuo_B = residuo_B[completos]

        y_oof = y_train[completos]

        # Valor esperado fuera de muestra reconstruido a
        # partir del residuo continuo.

        esperado_oof = (
            y_oof.astype(float)
            -
            residuo_B
        )

        # ------------------------------------------------
        # 13.4 Moran de los residuos
        # ------------------------------------------------

        moran_A = calcular_moran(
            coords_oof,
            residuo_A
        )

        moran_B = calcular_moran(
            coords_oof,
            residuo_B
        )

        # ------------------------------------------------
        # 13.5 Nivel interno: selección del variograma
        # ------------------------------------------------

        variograma_A, rmse_A = seleccionar_variograma(
            coords_oof,
            residuo_A,
            VARIOGRAMAS_CANDIDATOS,
            N_FOLDS_VARIOGRAMA,
            semilla_particion
        )

        variograma_B, rmse_B = seleccionar_variograma(
            coords_oof,
            residuo_B,
            VARIOGRAMAS_CANDIDATOS,
            N_FOLDS_VARIOGRAMA,
            semilla_particion
        )

        # ------------------------------------------------
        # 13.6 Modelo final sobre todo el entrenamiento
        # ------------------------------------------------

        modelo_final = construir_modelos(
            random_state=semilla_particion
        )[nombre_modelo]

        modelo_final.fit(
            X_train_sel,
            y_train
        )

        pred_base = modelo_final.predict(
            X_test_sel
        ).astype(int)

        esperado_test = modelo_final.predict_expected_value(
            X_test_sel
        )

        # Control sin kriging: redondeo del valor esperado.
        # Sirve para separar el efecto de la regla de
        # decisión del efecto espacial.

        pred_base_ev = np.clip(
            np.rint(esperado_test),
            0,
            N_CLASSES - 1
        ).astype(int)

        # ------------------------------------------------
        # 13.7 Método A
        # ------------------------------------------------

        pred_A = pred_base.copy()

        sill_A = rango_A = nugget_A = np.nan

        if variograma_A is not None:

            try:

                ok_A = ajustar_kriging(
                    coords_oof,
                    residuo_A,
                    variograma_A
                )

                sill_A, rango_A, nugget_A = parametros_variograma(
                    ok_A
                )

                correccion_A = predecir_kriging(
                    ok_A,
                    coords_test,
                    len(coords_oof)
                )

                pred_A = np.clip(
                    np.rint(
                        pred_base.astype(float)
                        +
                        correccion_A
                    ),
                    0,
                    N_CLASSES - 1
                ).astype(int)

            except Exception as error:

                errores_ejecucion.append({
                    "particion": etiqueta,
                    "etapa": "kriging A",
                    "error": str(error)
                })

        # ------------------------------------------------
        # 13.8 Método B
        # ------------------------------------------------

        pred_B = pred_base_ev.copy()

        sill_B = rango_B = nugget_B = np.nan

        correccion_B = None

        if variograma_B is not None:

            try:

                ok_B = ajustar_kriging(
                    coords_oof,
                    residuo_B,
                    variograma_B
                )

                sill_B, rango_B, nugget_B = parametros_variograma(
                    ok_B
                )

                correccion_B = predecir_kriging(
                    ok_B,
                    coords_test,
                    len(coords_oof)
                )

                pred_B = np.clip(
                    np.rint(
                        esperado_test
                        +
                        correccion_B
                    ),
                    0,
                    N_CLASSES - 1
                ).astype(int)

            except Exception as error:

                errores_ejecucion.append({
                    "particion": etiqueta,
                    "etapa": "kriging B",
                    "error": str(error)
                })

        # ------------------------------------------------
        # 13.8b Umbral calibrado
        #
        # El Metodo B convierte el valor continuo a nivel
        # ordinal por redondeo, que equivale a un umbral
        # fijo de 0.5. Aqui ese umbral se calibra dentro
        # del entrenamiento.
        #
        # Se calibra tambien para Base EV, de modo que la
        # comparacion entre ambos siga siendo justa.
        # ------------------------------------------------

        umbral_base_ev = calibrar_umbral(
            esperado_oof,
            y_oof
        )

        pred_base_ev_cal = np.clip(
            np.floor(
                esperado_test
                +
                umbral_base_ev
            ),
            0,
            N_CLASSES - 1
        ).astype(int)

        umbral_metodo_b = 0.5

        pred_B_cal = pred_base_ev_cal.copy()

        if (
            variograma_B is not None
            and
            correccion_B is not None
        ):

            try:

                correccion_oof = kriging_oof_predicciones(
                    coords_oof,
                    residuo_B,
                    variograma_B,
                    N_FOLDS_VARIOGRAMA,
                    semilla_particion
                )

                validos_oof = np.isfinite(correccion_oof)

                if validos_oof.sum() >= 30:

                    umbral_metodo_b = calibrar_umbral(
                        esperado_oof[validos_oof]
                        +
                        correccion_oof[validos_oof],
                        y_oof[validos_oof]
                    )

                pred_B_cal = np.clip(
                    np.floor(
                        esperado_test
                        +
                        correccion_B
                        +
                        umbral_metodo_b
                    ),
                    0,
                    N_CLASSES - 1
                ).astype(int)

            except Exception as error:

                errores_ejecucion.append({
                    "particion": etiqueta,
                    "etapa": "umbral calibrado",
                    "error": str(error)
                })

        # ------------------------------------------------
        # 13.8c Modelos de referencia
        # ------------------------------------------------

        pred_trivial = prediccion_trivial(
            y_train,
            len(y_test)
        )

        pred_knn = prediccion_knn_espacial(
            coords_train,
            y_train,
            coords_test,
            K_VECINOS_REFERENCIA
        )

        # ------------------------------------------------
        # 13.8d Sensibilidad a la codificacion ordinal
        #
        # Se repite el Metodo B calibrado sustituyendo los
        # enteros 0..K-1 por puntajes derivados de la tasa
        # mediana de cada categoria en escala logaritmica.
        # ------------------------------------------------

        pred_B_log = pred_B_cal.copy()

        variograma_log = None

        umbral_log = np.nan

        moran_log = (np.nan, np.nan, np.nan)

        if EJECUTAR_SENSIBILIDAD:

            try:

                escala = construir_escala(
                    y_train,
                    rate_full[idx_train]
                )

                # Probabilidades por clase del modelo final

                P_test = modelo_final.predict_class_probabilities(
                    X_test_sel
                )

                esperado_log_test = P_test.dot(escala)

                # Residuos en la codificacion alternativa.
                #
                # El valor esperado fuera de muestra se
                # reconstruye desplazando el de la escala
                # uniforme a la escala alternativa.

                posicion_oof = np.clip(
                    esperado_oof,
                    0.0,
                    N_CLASSES - 1
                )

                esperado_log_oof = np.interp(
                    posicion_oof,
                    np.arange(N_CLASSES, dtype=float),
                    escala
                )

                residuo_log = (
                    escala[y_oof]
                    -
                    esperado_log_oof
                )

                moran_log = calcular_moran(
                    coords_oof,
                    residuo_log
                )

                variograma_log, _ = seleccionar_variograma(
                    coords_oof,
                    residuo_log,
                    VARIOGRAMAS_CANDIDATOS,
                    N_FOLDS_VARIOGRAMA,
                    semilla_particion
                )

                if variograma_log is not None:

                    ok_log = ajustar_kriging(
                        coords_oof,
                        residuo_log,
                        variograma_log
                    )

                    correccion_log = predecir_kriging(
                        ok_log,
                        coords_test,
                        len(coords_oof)
                    )

                    correccion_log_oof = kriging_oof_predicciones(
                        coords_oof,
                        residuo_log,
                        variograma_log,
                        N_FOLDS_VARIOGRAMA,
                        semilla_particion
                    )

                    validos_log = np.isfinite(
                        correccion_log_oof
                    )

                    umbral_log = 0.5

                    if validos_log.sum() >= 30:

                        posicion_cal = escala_a_posicion(
                            esperado_log_oof[validos_log]
                            +
                            correccion_log_oof[validos_log],
                            escala
                        )

                        umbral_log = calibrar_umbral(
                            posicion_cal,
                            y_oof[validos_log]
                        )

                    posicion_test = escala_a_posicion(
                        esperado_log_test
                        +
                        correccion_log,
                        escala
                    )

                    pred_B_log = np.clip(
                        np.floor(
                            posicion_test
                            +
                            umbral_log
                        ),
                        0,
                        N_CLASSES - 1
                    ).astype(int)

            except Exception as error:

                errores_ejecucion.append({
                    "particion": etiqueta,
                    "etapa": "sensibilidad codificacion",
                    "error": str(error)
                })

        # ------------------------------------------------
        # 13.9 Métricas
        # ------------------------------------------------

        metricas = {
            "Base": calcular_metricas_ordinales(y_test, pred_base),
            "Base EV": calcular_metricas_ordinales(y_test, pred_base_ev),
            "Metodo A": calcular_metricas_ordinales(y_test, pred_A),
            "Metodo B": calcular_metricas_ordinales(y_test, pred_B),
            "Base EV cal": calcular_metricas_ordinales(y_test, pred_base_ev_cal),
            "Metodo B cal": calcular_metricas_ordinales(y_test, pred_B_cal),
            "Metodo B log": calcular_metricas_ordinales(y_test, pred_B_log),
            "Trivial": calcular_metricas_ordinales(y_test, pred_trivial),
            "kNN espacial": calcular_metricas_ordinales(y_test, pred_knn)
        }

        error_base = np.abs(y_test - pred_base)

        error_A = np.abs(y_test - pred_A)

        error_B = np.abs(y_test - pred_B)

        registro = {
            "esquema": particion["esquema"],
            "repeticion": particion["repeticion"],
            "bloque": particion["bloque"],
            "n_train": len(idx_train),
            "n_test": len(idx_test),
            "modelo_seleccionado": nombre_modelo,
            "n_variables": int(columnas.size),
            "variograma_A": variograma_A,
            "variograma_B": variograma_B,
            "umbral_base_ev": umbral_base_ev,
            "umbral_log": umbral_log,
            "variograma_log": variograma_log,
            "moran_log_I": moran_log[0],
            "moran_log_p": moran_log[1],
            "umbral_metodo_b": umbral_metodo_b,
            "sill_B": sill_B,
            "rango_B": rango_B,
            "nugget_B": nugget_B,
            "moran_A_I": moran_A[0],
            "moran_A_p": moran_A[1],
            "moran_A_z": moran_A[2],
            "moran_B_I": moran_B[0],
            "moran_B_p": moran_B[1],
            "moran_B_z": moran_B[2],
            "cambiadas_A": int(np.sum(pred_A != pred_base)),
            "cambiadas_B": int(np.sum(pred_B != pred_base_ev)),
            "mejoradas_A": int(np.sum(error_A < error_base)),
            "empeoradas_A": int(np.sum(error_A > error_base)),
            "mejoradas_B": int(np.sum(error_B < error_base)),
            "empeoradas_B": int(np.sum(error_B > error_base))
        }

        for predictor, valores in metricas.items():

            for nombre_metrica, valor in zip(
                METRIC_NAMES,
                valores
            ):

                clave = (
                    f"{nombre_metrica}_{predictor}"
                    .replace(" ", "_")
                    .replace("±", "")
                )

                registro[clave] = float(valor)

        for nombre in MODEL_NAMES:

            registro[
                f"MAE_interno_{nombre}"
            ] = resumen_interno[nombre]["mae_interno"]

        registros.append(registro)

        # ------------------------------------------------
        # 13.10 Predicciones por municipio
        #
        # Necesarias para construir matrices de confusion y
        # desempeno por clase sin repetir la ejecucion.
        # ------------------------------------------------

        detalle_particion = pd.DataFrame({
            "esquema": particion["esquema"],
            "repeticion": particion["repeticion"],
            "bloque": particion["bloque"],
            "indice_municipio": idx_test,
            "y_true": y_test,
            "Base": pred_base,
            "Base EV": pred_base_ev,
            "Metodo A": pred_A,
            "Metodo B": pred_B,
            "Base EV cal": pred_base_ev_cal,
            "Metodo B cal": pred_B_cal,
            "Metodo B log": pred_B_log,
            "Trivial": pred_trivial,
            "kNN espacial": pred_knn
        })

        predicciones_detalle.append(detalle_particion)

        particiones_completadas += 1

        # Guardado incremental para no perder el avance

        pd.DataFrame(registros).to_csv(
            ruta_parcial,
            index=False
        )

    except Exception as error:

        print(
            f"    error: {error}"
        )

        errores_ejecucion.append({
            "particion": etiqueta,
            "etapa": "general",
            "error": str(error)
        })


tiempo_total = time.time() - tiempo_inicio

print()
print("=" * 90)
print("PROCESO FINALIZADO")
print("=" * 90)

print(
    f"Tiempo total: {tiempo_total / 60:.1f} minutos"
)

print(
    "Particiones completadas:",
    particiones_completadas,
    "de",
    N_PARTICIONES
)

print(
    "Incidencias registradas:",
    len(errores_ejecucion)
)

print()


if particiones_completadas == 0:

    raise RuntimeError(
        "Ninguna partición se completó. "
        "Revise las incidencias antes de continuar."
    )


resultados = pd.DataFrame(registros)


# ================================================================
# 14. RESUMEN POR ESQUEMA
# ================================================================

PREDICTORES = [
    "Trivial",
    "kNN espacial",
    "Base",
    "Base EV",
    "Metodo A",
    "Metodo B",
    "Base EV cal",
    "Metodo B cal",
    "Metodo B log"
]


def clave_metrica(
    nombre_metrica,
    predictor
):

    return (
        f"{nombre_metrica}_{predictor}"
        .replace(" ", "_")
        .replace("±", "")
    )


def resumen_por_esquema(
    tabla,
    esquema
):

    sub = tabla[
        tabla["esquema"] == esquema
    ]

    if sub.empty:

        return None

    filas = []

    for nombre_metrica in METRIC_NAMES:

        fila = {
            "Métrica": nombre_metrica
        }

        for predictor in PREDICTORES:

            valores = sub[
                clave_metrica(nombre_metrica, predictor)
            ].to_numpy(dtype=float)

            fila[predictor] = (
                f"{np.nanmean(valores):.4f}"
                f" ± {np.nanstd(valores, ddof=1):.4f}"
            )

        filas.append(fila)

    return pd.DataFrame(filas)


for esquema in ESQUEMAS:

    tabla_esquema = resumen_por_esquema(
        resultados,
        esquema
    )

    if tabla_esquema is None:

        continue

    n_particiones_esquema = int(
        (resultados["esquema"] == esquema).sum()
    )

    print("=" * 110)

    print(
        f"ESQUEMA {esquema.upper()} — "
        f"{n_particiones_esquema} particiones externas"
    )

    print("=" * 110)

    print(
        tabla_esquema.to_string(index=False)
    )

    print()


# ================================================================
# 15. COMPARACIÓN ENTRE ESQUEMAS
#
# La diferencia entre el esquema aleatorio y el espacial
# cuantifica el optimismo atribuible a la proximidad
# geográfica entre entrenamiento y prueba.
# ================================================================

if len(ESQUEMAS) > 1:

    filas_comparacion = []

    for predictor in PREDICTORES:

        columna = clave_metrica("MAE", predictor)

        mae_aleatorio = resultados.loc[
            resultados["esquema"] == "aleatorio",
            columna
        ].to_numpy(dtype=float)

        mae_espacial = resultados.loc[
            resultados["esquema"] == "espacial",
            columna
        ].to_numpy(dtype=float)

        if mae_aleatorio.size == 0 or mae_espacial.size == 0:

            continue

        filas_comparacion.append({
            "Predictor": predictor,
            "MAE aleatorio": np.nanmean(mae_aleatorio),
            "MAE espacial": np.nanmean(mae_espacial),
            "Optimismo": (
                np.nanmean(mae_espacial)
                -
                np.nanmean(mae_aleatorio)
            )
        })

    if filas_comparacion:

        print("=" * 110)
        print("OPTIMISMO DEL ESQUEMA ALEATORIO FRENTE AL ESPACIAL")
        print("=" * 110)

        print(
            pd.DataFrame(filas_comparacion)
            .round(4)
            .to_string(index=False)
        )

        print()


# ================================================================
# 16. MODELO Y VARIOGRAMA SELECCIONADOS
# ================================================================

print("=" * 110)
print("MODELO ORDINAL SELECCIONADO EN EL NIVEL INTERNO")
print("=" * 110)

tabla_modelos = (
    resultados
    .groupby(["esquema", "modelo_seleccionado"])
    .size()
    .rename("Frecuencia")
    .reset_index()
)

tabla_modelos["Porcentaje"] = (
    tabla_modelos
    .groupby("esquema")["Frecuencia"]
    .transform(lambda s: 100 * s / s.sum())
)

print(
    tabla_modelos
    .round(2)
    .to_string(index=False)
)

print()


print("=" * 110)
print("VARIOGRAMA SELECCIONADO (MÉTODO B)")
print("=" * 110)

tabla_variogramas = (
    resultados
    .groupby(["esquema", "variograma_B"])
    .size()
    .rename("Frecuencia")
    .reset_index()
)

print(
    tabla_variogramas.to_string(index=False)
)

print()


# ================================================================
# 16b. UMBRALES CALIBRADOS
#
# El redondeo estandar corresponde a 0.5. Un umbral menor
# desplaza las predicciones hacia niveles mas bajos.
# ================================================================

tabla_umbrales = (
    resultados
    .groupby("esquema")[[
        "umbral_base_ev",
        "umbral_metodo_b"
    ]]
    .agg(["mean", "std", "min", "max"])
    .round(4)
)

print("=" * 110)
print("UMBRALES CALIBRADOS DENTRO DEL ENTRENAMIENTO")
print("=" * 110)

print(tabla_umbrales.to_string())

print()


# ================================================================
# 17. FRECUENCIA DE SELECCIÓN DE VARIABLES
# ================================================================

tabla_variables = pd.DataFrame({
    "Variable": feature_names,
    "Veces seleccionada": frecuencia_variables,
    "Porcentaje": (
        100
        *
        frecuencia_variables
        /
        max(1, particiones_completadas)
    )
}).sort_values(
    "Veces seleccionada",
    ascending=False
).reset_index(drop=True)

print("=" * 110)
print("ESTABILIDAD DE LA SELECCIÓN DE VARIABLES")
print("=" * 110)

print(
    tabla_variables
    .head(30)
    .round(2)
    .to_string(index=False)
)

print()


# ================================================================
# 18. MORAN
#
# Se reportan mediana, rango intercuartílico y porcentaje de
# particiones con p < 0.05. Promediar valores p a través de
# conjuntos de entrenamiento traslapados no es interpretable.
# ================================================================

filas_moran = []

for esquema in ESQUEMAS:

    sub = resultados[
        resultados["esquema"] == esquema
    ]

    if sub.empty:

        continue

    for etiqueta_residuo, sufijo in [
        ("Discreto (A)", "A"),
        ("Continuo (B)", "B")
    ]:

        valores_I = sub[f"moran_{sufijo}_I"].to_numpy(dtype=float)

        valores_p = sub[f"moran_{sufijo}_p"].to_numpy(dtype=float)

        validos = np.isfinite(valores_I)

        if validos.sum() == 0:

            continue

        filas_moran.append({
            "Esquema": esquema,
            "Residuo": etiqueta_residuo,
            "I mediana": np.nanmedian(valores_I),
            "I Q1": np.nanpercentile(valores_I[validos], 25),
            "I Q3": np.nanpercentile(valores_I[validos], 75),
            "% p < 0.05": (
                100
                *
                np.nanmean(valores_p[validos] < 0.05)
            )
        })

if filas_moran:

    print("=" * 110)
    print("AUTOCORRELACIÓN ESPACIAL DE LOS RESIDUOS")
    print("=" * 110)

    print(
        pd.DataFrame(filas_moran)
        .round(4)
        .to_string(index=False)
    )

    print()


# ================================================================
# 19. FRECUENCIA DE MEJORA DEL MAE
# ================================================================

COMPARACIONES = [
    ("Metodo A vs Base",           "Metodo A",     "Base"),
    ("Base EV vs Base",            "Base EV",      "Base"),
    ("Metodo B vs Base EV",        "Metodo B",     "Base EV"),
    ("Metodo B vs Base",           "Metodo B",     "Base"),
    ("Metodo B vs Metodo A",       "Metodo B",     "Metodo A"),
    # Comparaciones con umbral calibrado
    ("Base EV cal vs Base EV",     "Base EV cal",  "Base EV"),
    ("Base EV cal vs Base",        "Base EV cal",  "Base"),
    ("Metodo B cal vs Base EV cal", "Metodo B cal", "Base EV cal"),
    ("Metodo B cal vs Metodo B",   "Metodo B cal", "Metodo B"),
    ("Metodo B cal vs Base",       "Metodo B cal", "Base"),
    # Contra los modelos de referencia
    ("Base vs Trivial",             "Base",         "Trivial"),
    ("Metodo B cal vs Trivial",     "Metodo B cal", "Trivial"),
    ("Metodo B cal vs kNN espacial", "Metodo B cal", "kNN espacial"),
    # Sensibilidad a la codificacion
    ("Metodo B log vs Metodo B cal", "Metodo B log", "Metodo B cal"),
    ("Metodo B log vs Base",        "Metodo B log", "Base")
]


filas_frecuencia = []

for etiqueta, predictor_1, predictor_2 in COMPARACIONES:

    for esquema in ESQUEMAS:

        sub = resultados[
            resultados["esquema"] == esquema
        ]

        if sub.empty:

            continue

        delta = (
            sub[clave_metrica("MAE", predictor_1)].to_numpy(dtype=float)
            -
            sub[clave_metrica("MAE", predictor_2)].to_numpy(dtype=float)
        )

        delta = delta[np.isfinite(delta)]

        if delta.size == 0:

            continue

        tolerancia = 1e-12

        filas_frecuencia.append({
            "Esquema": esquema,
            "Comparación": etiqueta,
            "Mejora %": 100 * np.mean(delta < -tolerancia),
            "Empeora %": 100 * np.mean(delta > tolerancia),
            "Sin cambio %": 100 * np.mean(np.abs(delta) <= tolerancia),
            "Delta medio": float(np.mean(delta))
        })

tabla_frecuencia = pd.DataFrame(filas_frecuencia)

print("=" * 110)
print("FRECUENCIA DE MEJORA DEL MAE")
print("=" * 110)

print(
    tabla_frecuencia
    .round(4)
    .to_string(index=False)
)

print()


# ================================================================
# 20. PRUEBAS PAREADAS DE WILCOXON
#
# ADVERTENCIA METODOLÓGICA
#
# Las particiones externas comparten observaciones, por lo
# que las diferencias pareadas no son independientes y el
# valor p resultante es anticonservador. Se reporta como
# evidencia descriptiva complementaria y no como prueba
# formal de significancia. La corrección de Nadeau y Bengio
# es la referencia habitual si se requiere una prueba
# ajustada.
# ================================================================

filas_wilcoxon = []

for etiqueta, predictor_1, predictor_2 in COMPARACIONES:

    for esquema in ESQUEMAS:

        sub = resultados[
            resultados["esquema"] == esquema
        ]

        if sub.empty:

            continue

        a = sub[clave_metrica("MAE", predictor_2)].to_numpy(dtype=float)

        b = sub[clave_metrica("MAE", predictor_1)].to_numpy(dtype=float)

        validos = np.isfinite(a) & np.isfinite(b)

        if validos.sum() < 5:

            continue

        try:

            # H1: el segundo predictor tiene menor MAE

            prueba = wilcoxon(
                a[validos],
                b[validos],
                alternative="greater",
                zero_method="pratt"
            )

            estadistico = float(prueba.statistic)

            p_valor = float(prueba.pvalue)

        except Exception:

            estadistico = np.nan

            p_valor = np.nan

        filas_wilcoxon.append({
            "Esquema": esquema,
            "Hipótesis": f"{predictor_1} mejora a {predictor_2}",
            "n": int(validos.sum()),
            "Estadístico": estadistico,
            "p unilateral": p_valor
        })

tabla_wilcoxon = pd.DataFrame(filas_wilcoxon)

print("=" * 110)
print("PRUEBAS PAREADAS DE WILCOXON SOBRE EL MAE")
print("=" * 110)

print(
    tabla_wilcoxon
    .round(6)
    .to_string(index=False)
)

print()

print(
    "Nota: las particiones comparten observaciones, por lo\n"
    "que estos valores p son anticonservadores y deben\n"
    "interpretarse como evidencia descriptiva."
)

print()


# ================================================================
# 21. EFECTO DE LAS CORRECCIONES
# ================================================================

tabla_cambios = (
    resultados
    .groupby("esquema")[[
        "cambiadas_A",
        "mejoradas_A",
        "empeoradas_A",
        "cambiadas_B",
        "mejoradas_B",
        "empeoradas_B",
        "n_test"
    ]]
    .mean()
    .reset_index()
)

print("=" * 110)
print("EFECTO PROMEDIO DE LAS CORRECCIONES POR KRIGING")
print("=" * 110)

print(
    tabla_cambios
    .round(2)
    .to_string(index=False)
)

print()


# ================================================================
# 21b. PREDICCIONES POR MUNICIPIO
# ================================================================

marca = datetime.now().strftime("%Y%m%d_%H%M")

esquemas_presentes = [
    esquema
    for esquema in ESQUEMAS
    if (resultados["esquema"] == esquema).any()
]

detalle = pd.concat(
    predicciones_detalle,
    ignore_index=True
)

ruta_detalle = os.path.join(
    CARPETA_SALIDA,
    f"predicciones_por_municipio_{marca}.csv"
)

detalle.to_csv(
    ruta_detalle,
    index=False
)


# ================================================================
# 21c. MATRICES DE CONFUSIÓN
#
# Se acumulan todas las predicciones de todas las
# particiones de cada esquema.
# ================================================================

PREDICTORES_CONFUSION = [
    "Base",
    "Metodo B cal",
    "Trivial"
]

for esquema in esquemas_presentes:

    sub_detalle = detalle[
        detalle["esquema"] == esquema
    ]

    if sub_detalle.empty:

        continue

    for predictor in PREDICTORES_CONFUSION:

        matriz = pd.crosstab(
            sub_detalle["y_true"],
            sub_detalle[predictor],
            rownames=["Observado"],
            colnames=["Predicho"],
            dropna=False
        )

        matriz = matriz.reindex(
            index=range(N_CLASSES),
            columns=range(N_CLASSES),
            fill_value=0
        )

        print("=" * 110)

        print(
            f"MATRIZ DE CONFUSIÓN — {esquema} — {predictor}"
        )

        print("=" * 110)

        print(matriz.to_string())

        print()

        matriz.to_csv(
            os.path.join(
                CARPETA_SALIDA,
                f"confusion_{esquema}_"
                f"{predictor.replace(' ', '_')}_{marca}.csv"
            )
        )


# ================================================================
# 21d. DESEMPEÑO POR CLASE
#
# Con categorias muy desbalanceadas, el promedio global
# puede ocultar un desempeno muy desigual entre niveles.
# ================================================================

filas_clase = []

for esquema in esquemas_presentes:

    sub_detalle = detalle[
        detalle["esquema"] == esquema
    ]

    if sub_detalle.empty:

        continue

    for predictor in PREDICTORES:

        for clase in range(N_CLASSES):

            mascara = sub_detalle["y_true"] == clase

            if mascara.sum() == 0:

                continue

            observado = sub_detalle.loc[
                mascara,
                "y_true"
            ].to_numpy()

            predicho = sub_detalle.loc[
                mascara,
                predictor
            ].to_numpy()

            error = np.abs(predicho - observado)

            filas_clase.append({
                "Esquema": esquema,
                "Predictor": predictor,
                "Clase": clase,
                "n": int(mascara.sum()),
                "MAE": float(np.mean(error)),
                "Acierto": float(np.mean(error == 0)),
                "Error grave": float(np.mean(error >= 2))
            })

tabla_clase = pd.DataFrame(filas_clase)

print("=" * 110)
print("DESEMPEÑO POR CLASE")
print("=" * 110)

for esquema in esquemas_presentes:

    sub_clase = tabla_clase[
        (tabla_clase["Esquema"] == esquema)
        &
        (tabla_clase["Predictor"].isin([
            "Trivial",
            "Base",
            "Metodo B cal"
        ]))
    ]

    if sub_clase.empty:

        continue

    print()

    print(f"Esquema {esquema}")

    print(
        sub_clase
        .pivot_table(
            index="Clase",
            columns="Predictor",
            values=["MAE", "Acierto", "Error grave"]
        )
        .round(4)
        .to_string()
    )

print()

tabla_clase.to_csv(
    os.path.join(
        CARPETA_SALIDA,
        f"desempeno_por_clase_{marca}.csv"
    ),
    index=False
)


# ================================================================
# 21e. TASAS DE ERROR GRAVE
#
# Se define error grave como una desviacion de dos o mas
# categorias respecto al nivel observado.
# ================================================================

filas_grave = []

for esquema in esquemas_presentes:

    sub_detalle = detalle[
        detalle["esquema"] == esquema
    ]

    if sub_detalle.empty:

        continue

    for predictor in PREDICTORES:

        error = np.abs(
            sub_detalle[predictor].to_numpy()
            -
            sub_detalle["y_true"].to_numpy()
        )

        filas_grave.append({
            "Esquema": esquema,
            "Predictor": predictor,
            "Error grave %": 100 * np.mean(error >= 2),
            "Error de 3 o mas %": 100 * np.mean(error >= 3),
            "Acierto exacto %": 100 * np.mean(error == 0)
        })

tabla_grave = pd.DataFrame(filas_grave)

print("=" * 110)
print("TASAS DE ERROR GRAVE")
print("=" * 110)

print(
    tabla_grave
    .round(3)
    .to_string(index=False)
)

print()

tabla_grave.to_csv(
    os.path.join(
        CARPETA_SALIDA,
        f"error_grave_{marca}.csv"
    ),
    index=False
)


# ================================================================
# 21f. SENSIBILIDAD A LA CODIFICACIÓN
# ================================================================

if EJECUTAR_SENSIBILIDAD:

    filas_sens = []

    for esquema in esquemas_presentes:

        sub = resultados[
            resultados["esquema"] == esquema
        ]

        if sub.empty:

            continue

        moran_uniforme = sub["moran_B_I"].to_numpy(dtype=float)

        moran_alterna = sub["moran_log_I"].to_numpy(dtype=float)

        filas_sens.append({
            "Esquema": esquema,
            "MAE codificación uniforme": np.nanmean(
                sub[clave_metrica("MAE", "Metodo B cal")]
            ),
            "MAE codificación log": np.nanmean(
                sub[clave_metrica("MAE", "Metodo B log")]
            ),
            "Moran I uniforme": np.nanmedian(moran_uniforme),
            "Moran I log": np.nanmedian(moran_alterna),
            "% p<0.05 log": 100 * np.nanmean(
                sub["moran_log_p"].to_numpy(dtype=float) < 0.05
            )
        })

    if filas_sens:

        print("=" * 110)
        print("SENSIBILIDAD A LA CODIFICACIÓN ORDINAL")
        print("=" * 110)

        print(
            pd.DataFrame(filas_sens)
            .round(4)
            .to_string(index=False)
        )

        print()


# ================================================================
# 22. GUARDADO
# ================================================================

# marca definida antes de las secciones de analisis

archivos = {
    f"resultados_por_particion_{marca}.csv": resultados,
    f"frecuencia_mejora_{marca}.csv": tabla_frecuencia,
    f"wilcoxon_{marca}.csv": tabla_wilcoxon,
    f"seleccion_variables_{marca}.csv": tabla_variables,
    f"modelos_seleccionados_{marca}.csv": tabla_modelos,
    f"variogramas_seleccionados_{marca}.csv": tabla_variogramas
}

if errores_ejecucion:

    archivos[f"incidencias_{marca}.csv"] = pd.DataFrame(
        errores_ejecucion
    )

print("=" * 110)
print("ARCHIVOS GUARDADOS")
print("=" * 110)

print(ruta_detalle)

for nombre_archivo, tabla in archivos.items():

    ruta = os.path.join(
        CARPETA_SALIDA,
        nombre_archivo
    )

    tabla.to_csv(
        ruta,
        index=False
    )

    print(ruta)

print()


# ================================================================
# 23. GRÁFICAS
# ================================================================

if esquemas_presentes:

    figura, ejes = plt.subplots(
        1,
        len(esquemas_presentes),
        figsize=(6 * len(esquemas_presentes), 5),
        squeeze=False
    )

    for posicion, esquema in enumerate(esquemas_presentes):

        sub = resultados[
            resultados["esquema"] == esquema
        ]

        datos = [
            sub[clave_metrica("MAE", predictor)].to_numpy(dtype=float)
            for predictor in PREDICTORES
        ]

        eje = ejes[0][posicion]

        eje.boxplot(
            datos,
            tick_labels=PREDICTORES
        )

        eje.set_title(
            f"MAE — esquema {esquema}"
        )

        eje.set_ylabel("MAE")

        eje.grid(
            axis="y",
            alpha=0.30
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CARPETA_SALIDA,
            f"mae_por_esquema_{marca}.png"
        ),
        dpi=150
    )

    plt.show()


# Distribución del cambio producido por el Método B
# respecto al control sin kriging.

figura, ejes = plt.subplots(
    1,
    len(esquemas_presentes),
    figsize=(6 * len(esquemas_presentes), 4.5),
    squeeze=False
)

for posicion, esquema in enumerate(esquemas_presentes):

    sub = resultados[
        resultados["esquema"] == esquema
    ]

    delta = (
        sub[clave_metrica("MAE", "Metodo B cal")].to_numpy(dtype=float)
        -
        sub[clave_metrica("MAE", "Base")].to_numpy(dtype=float)
    )

    delta = delta[np.isfinite(delta)]

    eje = ejes[0][posicion]

    if delta.size > 0:

        eje.hist(
            delta,
            bins=15
        )

    eje.axvline(
        x=0,
        linewidth=1,
        color="black"
    )

    eje.set_title(
        f"Efecto espacial puro — {esquema}"
    )

    eje.set_xlabel(
        "MAE Método B calibrado menos MAE Base"
    )

    eje.set_ylabel("Frecuencia")

    eje.grid(
        axis="y",
        alpha=0.30
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        CARPETA_SALIDA,
        f"efecto_espacial_{marca}.png"
    ),
    dpi=150
)

plt.show()

# ======================================================================
# 04 -- Scores of the alternative category coding, per partition
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Run from the repository root:
#     python scripts/04_puntajes_codificacion.py
#
# Inputs:  data/municipal_dengue_dataset_mexico_2022.csv
# Outputs: resultados/puntajes_codificacion_alternativa.csv
#
# The code and its comments are in Spanish, as written by the authors.
# Only the data and output paths were changed when the analysis was
# organised as a repository; see scripts/rutas.py.
# ======================================================================

# ======================================================================
# PUNTAJES DE LA CODIFICACIÓN ALTERNATIVA POR PARTICIÓN
#
# Reconstruye exactamente las mismas particiones externas del análisis
# principal (mismas semillas) y calcula, para cada una, los puntajes de
# la codificación alternativa:
#
#     score_k = log(1 + mediana de la tasa dentro del nivel k)
#
# reescalados al intervalo [0, K-1].
#
# No ajusta ningún modelo.
#
# Sirve para reportar en el manuscrito la variabilidad de los puntajes
# entre conjuntos de entrenamiento.
#
# ======================================================================

import os

import numpy as np
import pandas as pd

import rutas

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# Configuración: debe coincidir con el análisis principal
# ----------------------------------------------------------------------

ARCHIVO_DATOS = rutas.ARCHIVO_DATOS

COLUMNA_NIVEL = "DRL"
COLUMNA_TASA = "Rate_2022"

COLUMNA_LON = "lon_mun"
COLUMNA_LAT = "lat_mun"

N_CLASSES = 5

N_REPETICIONES_ALEATORIO = 50
SEMILLA_BASE_ALEATORIO = 5000
TEST_SIZE = 0.20

N_BLOQUES = 5
N_REPETICIONES_ESPACIAL = 10
SEMILLA_BASE_ESPACIAL = 7000


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------

df = pd.read_csv(ARCHIVO_DATOS)

y = df[COLUMNA_NIVEL].to_numpy().ravel().astype(int)

tasa = df[COLUMNA_TASA].to_numpy(dtype=float)

coords = df[[COLUMNA_LON, COLUMNA_LAT]].to_numpy(dtype=float)

print("Municipios:", len(y))
print()


# ----------------------------------------------------------------------
# Construcción de los puntajes (idéntica a la del análisis principal)
# ----------------------------------------------------------------------

def construir_escala(y_train, rate_train):

    puntajes = np.full(N_CLASSES, np.nan)

    for k in range(N_CLASSES):

        mascara = y_train == k

        if mascara.sum() > 0:

            puntajes[k] = np.log1p(np.median(rate_train[mascara]))

    posiciones = np.arange(N_CLASSES, dtype=float)

    observados = np.isfinite(puntajes)

    if observados.sum() < 2:

        return posiciones

    puntajes = np.interp(
        posiciones, posiciones[observados], puntajes[observados]
    )

    for k in range(1, N_CLASSES):

        if puntajes[k] <= puntajes[k - 1]:

            puntajes[k] = puntajes[k - 1] + 1e-6

    rango = puntajes[-1] - puntajes[0]

    if rango <= 0:

        return posiciones

    return (puntajes - puntajes[0]) / rango * (N_CLASSES - 1)


# ----------------------------------------------------------------------
# Particiones externas
# ----------------------------------------------------------------------

indices = np.arange(len(y))

particiones = []

for repeticion in range(N_REPETICIONES_ALEATORIO):

    idx_train, _ = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEMILLA_BASE_ALEATORIO + repeticion,
        stratify=y
    )

    particiones.append(("aleatorio", idx_train))

for repeticion in range(N_REPETICIONES_ESPACIAL):

    etiquetas = KMeans(
        n_clusters=N_BLOQUES,
        n_init=10,
        random_state=SEMILLA_BASE_ESPACIAL + repeticion
    ).fit_predict(coords)

    for bloque in range(N_BLOQUES):

        mascara_test = etiquetas == bloque

        idx_test = indices[mascara_test]

        idx_train = indices[~mascara_test]

        if idx_test.size < 5 or idx_train.size < 30:

            continue

        particiones.append(("espacial", idx_train))

print("Particiones reconstruidas:", len(particiones))

print(
    "  aleatorias:",
    sum(1 for e, _ in particiones if e == "aleatorio")
)

print(
    "  espaciales:",
    sum(1 for e, _ in particiones if e == "espacial")
)

print()


# ----------------------------------------------------------------------
# Puntajes por partición
# ----------------------------------------------------------------------

filas = []

for esquema, idx_train in particiones:

    escala = construir_escala(y[idx_train], tasa[idx_train])

    fila = {"esquema": esquema}

    for k in range(N_CLASSES):

        fila[f"s{k}"] = escala[k]

    filas.append(fila)

tabla = pd.DataFrame(filas)

columnas = [f"s{k}" for k in range(N_CLASSES)]


# ----------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------

print("=" * 78)
print("PUNTAJES POR ESQUEMA")
print("=" * 78)

print(
    tabla.groupby("esquema")[columnas]
    .agg(["mean", "std", "min", "max"])
    .round(4)
    .T.to_string()
)

print()
print("=" * 78)
print("PUNTAJES SOBRE TODAS LAS PARTICIONES")
print("=" * 78)

resumen = tabla[columnas].agg(["mean", "std", "min", "max"]).round(4)

print(resumen.T.to_string())

print()

media = tabla[columnas].mean().to_numpy()

print("Media de los puntajes:", np.round(media, 3))

print("Saltos entre niveles :", np.round(np.diff(media), 3))

print()

escala_completa = construir_escala(y, tasa)

print(
    "Puntajes sobre la muestra completa:",
    np.round(escala_completa, 3)
)

print(
    "Diferencia máxima respecto a la media entre particiones:",
    round(float(np.max(np.abs(escala_completa - media))), 4)
)

print()
print("Frase sugerida para el manuscrito:")

print(
    "  the scores are "
    + ", ".join(f"{m:.2f}" for m in media[:-1])
    + f" and {media[-1]:.2f} on average across partitions, with a "
    f"maximum standard deviation of "
    f"{tabla[columnas].std().max():.3f} across the five levels"
)

RUTA_PUNTAJES = os.path.join(
    rutas.CARPETA_RESULTADOS,
    "puntajes_codificacion_alternativa.csv"
)

tabla.to_csv(RUTA_PUNTAJES, index=False)

print()
print("Guardado:", RUTA_PUNTAJES)

# ======================================================================
# 05 -- SHAP for the cumulative ordinal Random Forest
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Run from the repository root:
#     python scripts/05_shap_random_forest.py
#
# Inputs:  data/... + resultados/seleccion_variables_*.csv
# Outputs: figuras/SHAP_Summary_DRL_greater_*.{pdf,png}, shap_importancia_rf.csv
#
# The code and its comments are in Spanish, as written by the authors.
# Only the data and output paths were changed when the analysis was
# organised as a repository; see scripts/rutas.py.
# ======================================================================

# ======================================================================
# SHAP PARA EL RANDOM FOREST ORDINAL ACUMULATIVO
#
# Recalcula el análisis de interpretabilidad sobre el modelo que el
# procedimiento anidado selecciona en la mayoría de las particiones.
#
# Se ajustan los cuatro clasificadores binarios
#
#     P(DRL > 0),  P(DRL > 1),  P(DRL > 2),  P(DRL > 3)
#
# sobre la muestra completa, con el conjunto de variables retenidas en
# al menos el 90% de las particiones del análisis principal, y se
# calculan los valores SHAP de cada uno.
#
# El análisis es descriptivo del modelo ajustado; no interviene en
# ninguna evaluación de desempeño.
#
# ======================================================================

import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from sklearn.ensemble import RandomForestClassifier

import rutas
import figuras_png


# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

ARCHIVO_DATOS = rutas.ARCHIVO_DATOS

COLUMNA_NIVEL = "DRL"

CARPETA_RESULTADOS = rutas.CARPETA_RESULTADOS

CARPETA_SALIDA = Path(rutas.CARPETA_FIGURAS)

GLOBAL_SEED = 123

N_CLASSES = 5

FRECUENCIA_MINIMA = 90.0

# Número de municipios cuyos valores SHAP se calculan.
# None = todos. Reducir si el cálculo resulta lento.

N_MUNICIPIOS_EXPLICADOS = None

# Número máximo de variables mostradas en cada gráfico.
# Como el conjunto estable tiene 12, se muestran todas.

MAX_VARIABLES_GRAFICO = 15


CARPETA_SALIDA.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------

df = pd.read_csv(ARCHIVO_DATOS)

y = df[COLUMNA_NIVEL].to_numpy().ravel().astype(int)

print("=" * 80)
print("DATOS")
print("=" * 80)
print("Municipios:", len(df))


# ----------------------------------------------------------------------
# Variables estables
# ----------------------------------------------------------------------

archivos = sorted(
    glob.glob(
        os.path.join(
            CARPETA_RESULTADOS, "seleccion_variables_*.csv"
        )
    )
)

if not archivos:

    raise SystemExit(
        "No se encontró la tabla de estabilidad de variables."
    )

tabla = pd.read_csv(archivos[-1])

variables = [
    v for v in tabla.loc[
        tabla["Porcentaje"] >= FRECUENCIA_MINIMA, "Variable"
    ].tolist()
    if v in df.columns
]

print("Tabla de estabilidad:", os.path.basename(archivos[-1]))
print(f"Variables (frecuencia >= {FRECUENCIA_MINIMA}%):", len(variables))
print(" ", ", ".join(variables))
print()

X = df[variables].to_numpy(dtype=float)

X_df = pd.DataFrame(X, columns=variables)


# ----------------------------------------------------------------------
# Municipios a explicar
# ----------------------------------------------------------------------

if N_MUNICIPIOS_EXPLICADOS is None:

    X_expl = X_df

else:

    rng = np.random.default_rng(GLOBAL_SEED)

    idx = rng.choice(
        len(X_df),
        size=min(N_MUNICIPIOS_EXPLICADOS, len(X_df)),
        replace=False
    )

    X_expl = X_df.iloc[np.sort(idx)]

print("Municipios explicados:", len(X_expl))
print()


# ----------------------------------------------------------------------
# Un clasificador binario por umbral
# ----------------------------------------------------------------------

def construir_rf(semilla):

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=semilla,
        n_jobs=-1
    )


def valores_shap_clase_positiva(valores):
    """
    Normaliza la salida de TreeExplainer a una matriz
    (n_municipios, n_variables) correspondiente a la clase 1.

    Según la versión de shap, la salida puede ser una lista de dos
    matrices o un arreglo tridimensional.
    """

    if isinstance(valores, list):

        return np.asarray(valores[1])

    valores = np.asarray(valores)

    if valores.ndim == 3:

        return valores[:, :, 1]

    return valores


resumen = []

for umbral in range(N_CLASSES - 1):

    print("=" * 80)
    print(f"CLASIFICADOR  P(DRL > {umbral})")
    print("=" * 80)

    y_binaria = (y > umbral).astype(int)

    print(
        f"Positivos: {y_binaria.sum()}  "
        f"Negativos: {len(y_binaria) - y_binaria.sum()}"
    )

    modelo = construir_rf(GLOBAL_SEED + umbral)

    inicio = time.time()

    modelo.fit(X, y_binaria)

    print(f"Ajuste: {time.time() - inicio:.1f} s")

    inicio = time.time()

    explicador = shap.TreeExplainer(modelo)

    shap_valores = valores_shap_clase_positiva(
        explicador.shap_values(X_expl, check_additivity=False)
    )

    print(
        f"SHAP: {time.time() - inicio:.1f} s  "
        f"(forma {shap_valores.shape})"
    )

    # ------------------------------------------------------------------
    # Importancia media por variable
    # ------------------------------------------------------------------

    importancia = np.abs(shap_valores).mean(axis=0)

    orden = np.argsort(-importancia)

    print("\nVariables por importancia media (|SHAP|):")

    for posicion in orden:

        resumen.append({
            "Clasificador": f"P(DRL>{umbral})",
            "Variable": variables[posicion],
            "Importancia": float(importancia[posicion])
        })

        print(
            f"  {variables[posicion]:>9}  "
            f"{importancia[posicion]:.5f}"
        )

    # ------------------------------------------------------------------
    # Gráfico de resumen
    # ------------------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_valores,
        X_expl,
        feature_names=variables,
        plot_type="dot",
        max_display=MAX_VARIABLES_GRAFICO,
        show=False
    )

    plt.title(
        f"SHAP Summary Plot - P(DRL > {umbral})",
        fontsize=14
    )

    plt.tight_layout()

    figura = plt.gcf()

    for extension in ("pdf", "png"):

        ruta = (
            CARPETA_SALIDA
            / f"SHAP_Summary_DRL_greater_{umbral}.{extension}"
        )

        figura.savefig(
            ruta,
            dpi=figuras_png.dpi_png(figura),
            bbox_inches="tight",
            facecolor="white"
        )

        if extension == "png":

            figuras_png.aplanar_png(ruta)

        print("  Guardado:", ruta)

    plt.close(figura)

    print()


# ----------------------------------------------------------------------
# Tabla resumen
# ----------------------------------------------------------------------

tabla_resumen = pd.DataFrame(resumen)

ruta_tabla = CARPETA_SALIDA / "shap_importancia_rf.csv"

tabla_resumen.to_csv(ruta_tabla, index=False)

print("=" * 80)
print("RESUMEN: tres variables más influyentes por clasificador")
print("=" * 80)

for clasificador in tabla_resumen["Clasificador"].unique():

    sub = tabla_resumen[
        tabla_resumen["Clasificador"] == clasificador
    ].nlargest(3, "Importancia")

    nombres = ", ".join(
        f"{r.Variable} ({r.Importancia:.4f})"
        for r in sub.itertuples()
    )

    print(f"  {clasificador}: {nombres}")

print()
print("Tabla completa:", ruta_tabla.resolve())

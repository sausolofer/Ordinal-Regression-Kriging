# ======================================================================
# 02 -- Row-normalised confusion matrices
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Run from the repository root:
#     python scripts/02_matrices_confusion.py
#
# Inputs:  resultados/predicciones_por_municipio_*.csv
# Outputs: resultados/confusion_norm_*.csv, resumen_confusion_*.csv, *.tex
#
# The code and its comments are in Spanish, as written by the authors.
# Only the data and output paths were changed when the analysis was
# organised as a repository; see scripts/rutas.py.
# ======================================================================

# ================================================================
# MATRICES DE CONFUSIÓN NORMALIZADAS
#
# Lee predicciones_por_municipio_*.csv y produce matrices
# expresadas como proporciones por fila, que indican la
# fraccion de municipios observados en cada categoria que
# recibio cada prediccion.
#
# Los conteos crudos acumulan todas las particiones, por lo
# que cada municipio aparece varias veces. La normalizacion
# por fila evita que los conteos se confundan con numero de
# municipios.
#
# No requiere volver a ejecutar el analisis.
# ================================================================

import glob
import os

import numpy as np
import pandas as pd

import rutas


CARPETA = rutas.CARPETA_RESULTADOS

N_CLASSES = 5

# Predictores a incluir, en el orden de presentacion

PREDICTORES = [
    "Trivial",
    "kNN espacial",
    "Base",
    "Metodo B",
    "Metodo B cal"
]

# Etiquetas para el manuscrito

ETIQUETAS_CLASE = [
    "0 sin casos",
    "1 bajo",
    "2 medio bajo",
    "3 medio alto",
    "4 alto"
]


# ----------------------------------------------------------------
# Localizar el archivo mas reciente
# ----------------------------------------------------------------

patron = os.path.join(
    CARPETA,
    "predicciones_por_municipio_*.csv"
)

archivos = sorted(
    glob.glob(patron)
)

if not archivos:

    raise SystemExit(
        f"No se encontro ningun archivo con el patron:\n{patron}"
    )

ruta = archivos[-1]

print(f"Archivo leido: {ruta}\n")

detalle = pd.read_csv(ruta)

marca = (
    os.path.basename(ruta)
    .replace("predicciones_por_municipio_", "")
    .replace(".csv", "")
)


# ----------------------------------------------------------------
# Comprobaciones
# ----------------------------------------------------------------

disponibles = [
    p for p in PREDICTORES
    if p in detalle.columns
]

faltantes = set(PREDICTORES) - set(disponibles)

if faltantes:

    print(
        "Advertencia: no estan en el archivo:",
        ", ".join(sorted(faltantes))
    )
    print()

esquemas = list(
    detalle["esquema"].unique()
)

print(
    "Predicciones totales:",
    len(detalle)
)

print(
    "Municipios distintos:",
    detalle["indice_municipio"].nunique()
)

for esquema in esquemas:

    sub = detalle[detalle["esquema"] == esquema]

    veces = (
        sub.groupby("indice_municipio")
        .size()
    )

    print(
        f"  {esquema}: {len(sub)} predicciones, "
        f"cada municipio evaluado entre "
        f"{veces.min()} y {veces.max()} veces"
    )

print()


# ----------------------------------------------------------------
# Construccion de las matrices
# ----------------------------------------------------------------

def matriz_normalizada(
    sub,
    predictor
):
    """
    Devuelve la matriz de confusion con filas que suman 1,
    junto con el numero de predicciones por fila.
    """

    conteos = pd.crosstab(
        sub["y_true"],
        sub[predictor]
    ).reindex(
        index=range(N_CLASSES),
        columns=range(N_CLASSES),
        fill_value=0
    )

    totales = conteos.sum(axis=1)

    proporciones = conteos.div(
        totales.replace(0, np.nan),
        axis=0
    )

    return conteos, proporciones, totales


resumen_filas = []

for esquema in esquemas:

    sub_esquema = detalle[
        detalle["esquema"] == esquema
    ]

    for predictor in disponibles:

        conteos, proporciones, totales = matriz_normalizada(
            sub_esquema,
            predictor
        )

        tabla = proporciones.copy()

        tabla.index = ETIQUETAS_CLASE

        tabla.columns = [
            str(k) for k in range(N_CLASSES)
        ]

        tabla.insert(
            0,
            "n",
            totales.values
        )

        print("=" * 78)

        print(
            f"{esquema.upper()} — {predictor}"
        )

        print(
            "Proporcion por fila; n es el numero de "
            "predicciones acumuladas"
        )

        print("=" * 78)

        print(
            tabla.round(3).to_string()
        )

        print()

        salida = os.path.join(
            CARPETA,
            f"confusion_norm_{esquema}_"
            f"{predictor.replace(' ', '_')}_{marca}.csv"
        )

        tabla.to_csv(salida)

        # Diagonal y masa fuera de la diagonal

        diagonal = np.diag(
            proporciones.fillna(0).to_numpy()
        )

        for k in range(N_CLASSES):

            fila = proporciones.iloc[k].fillna(0).to_numpy()

            resumen_filas.append({
                "Esquema": esquema,
                "Predictor": predictor,
                "Clase": k,
                "n": int(totales.iloc[k]),
                "Acierto": float(diagonal[k]),
                "Predicho por debajo": float(
                    fila[:k].sum()
                ),
                "Predicho por encima": float(
                    fila[k + 1:].sum()
                ),
                "Modo predicho": int(
                    np.argmax(fila)
                )
            })


# ----------------------------------------------------------------
# Resumen compacto
# ----------------------------------------------------------------

resumen = pd.DataFrame(resumen_filas)

print("=" * 78)
print("RESUMEN POR CLASE")
print("=" * 78)

for esquema in esquemas:

    print()

    print(f"Esquema {esquema}")

    sub_resumen = resumen[
        resumen["Esquema"] == esquema
    ]

    print(
        sub_resumen
        .pivot_table(
            index="Clase",
            columns="Predictor",
            values="Acierto"
        )
        .round(3)
        .to_string()
    )

print()

ruta_resumen = os.path.join(
    CARPETA,
    f"resumen_confusion_{marca}.csv"
)

resumen.to_csv(
    ruta_resumen,
    index=False
)


# ----------------------------------------------------------------
# Version en LaTeX del predictor principal
# ----------------------------------------------------------------

PREDICTOR_LATEX = "Metodo B cal"

if PREDICTOR_LATEX in disponibles:

    for esquema in esquemas:

        sub_esquema = detalle[
            detalle["esquema"] == esquema
        ]

        _, proporciones, totales = matriz_normalizada(
            sub_esquema,
            PREDICTOR_LATEX
        )

        tabla = proporciones.copy()

        tabla.index = ETIQUETAS_CLASE

        tabla.columns = [
            str(k) for k in range(N_CLASSES)
        ]

        ruta_tex = os.path.join(
            CARPETA,
            f"confusion_{esquema}_{marca}.tex"
        )

        with open(ruta_tex, "w", encoding="utf-8") as f:

            f.write(
                tabla.round(3).to_latex(
                    caption=(
                        f"Confusion matrix (row proportions) "
                        f"for the proposed method under "
                        f"{esquema} validation."
                    ),
                    label=f"tab:confusion_{esquema}"
                )
            )

        print(f"LaTeX guardado: {ruta_tex}")

print()
print(f"Resumen guardado: {ruta_resumen}")

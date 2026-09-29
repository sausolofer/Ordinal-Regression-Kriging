# ======================================================================
# 03 -- Effect of merging adjacent risk categories (supplementary)
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Run from the repository root:
#     python scripts/03_efecto_agrupacion.py
#
# Inputs:  resultados/predicciones_por_municipio_*.csv
# Outputs: resultados/efecto_agrupacion_*.csv
#
# The code and its comments are in Spanish, as written by the authors.
# Only the data and output paths were changed when the analysis was
# organised as a repository; see scripts/rutas.py.
# ======================================================================

# ================================================================
# EFECTO DE FUSIONAR CATEGORÍAS
#
# Evalua si agrupar niveles mejora la capacidad
# discriminante, usando las predicciones ya guardadas.
#
# ADVERTENCIA IMPORTANTE
#
# Esto NO equivale a reentrenar el modelo con menos
# categorias. Aqui se agrupan a posteriori las predicciones
# de un modelo ajustado con cinco niveles. El resultado
# indica si el ordenamiento producido por el modelo
# conserva informacion a una resolucion mas gruesa, lo cual
# es una senal util pero no definitiva.
#
# Si alguna agrupacion resulta claramente superior, el paso
# siguiente es reejecutar el analisis completo con la nueva
# codificacion, porque el modelo reentrenado puede
# comportarse de forma distinta.
#
# El MAE no es comparable entre agrupaciones, porque al
# reducir el numero de categorias disminuye la distancia
# maxima posible. Por eso se reporta tambien la mejora
# relativa respecto al predictor trivial de cada
# agrupacion, que si es comparable.
# ================================================================

import glob
import os

import numpy as np
import pandas as pd

import rutas


CARPETA = rutas.CARPETA_RESULTADOS

PREDICTORES = [
    "Trivial",
    "kNN espacial",
    "Base",
    "Metodo B",
    "Metodo B cal"
]

PREDICTOR_DETALLE = "Metodo B cal"


# Agrupaciones a evaluar.
# Cada diccionario asigna el nivel original al nuevo nivel.

AGRUPACIONES = {

    "5 niveles (original)":
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},

    "4 niveles: fusion 3+4":
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 3},

    "4 niveles: fusion 1+2":
        {0: 0, 1: 1, 2: 1, 3: 2, 4: 3},

    "3 niveles: 1+2 y 3+4":
        {0: 0, 1: 1, 2: 1, 3: 2, 4: 2},

    "2 niveles: 0 vs resto":
        {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
}


# ----------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------

archivos = sorted(
    glob.glob(
        os.path.join(
            CARPETA,
            "predicciones_por_municipio_*.csv"
        )
    )
)

if not archivos:

    raise SystemExit(
        "No se encontro el archivo de predicciones."
    )

ruta = archivos[-1]

detalle = pd.read_csv(ruta)

marca = (
    os.path.basename(ruta)
    .replace("predicciones_por_municipio_", "")
    .replace(".csv", "")
)

print(f"Archivo leido: {ruta}")
print()

disponibles = [
    p for p in PREDICTORES
    if p in detalle.columns
]

esquemas = list(
    detalle["esquema"].unique()
)


def aplicar(
    valores,
    mapa
):

    return np.vectorize(
        mapa.get
    )(
        np.asarray(valores, dtype=int)
    )


# ----------------------------------------------------------------
# Métricas por agrupación
# ----------------------------------------------------------------

filas = []

for esquema in esquemas:

    sub = detalle[
        detalle["esquema"] == esquema
    ]

    for nombre_agrupacion, mapa in AGRUPACIONES.items():

        y_agrupada = aplicar(
            sub["y_true"],
            mapa
        )

        n_niveles = len(set(mapa.values()))

        # Predictor trivial de esta agrupacion:
        # la mediana de la respuesta agrupada.

        constante = int(
            np.median(y_agrupada)
        )

        mae_trivial = float(
            np.mean(
                np.abs(y_agrupada - constante)
            )
        )

        for predictor in disponibles:

            pred_agrupada = aplicar(
                sub[predictor],
                mapa
            )

            error = np.abs(
                pred_agrupada - y_agrupada
            )

            mae = float(np.mean(error))

            filas.append({
                "Esquema": esquema,
                "Agrupación": nombre_agrupacion,
                "Niveles": n_niveles,
                "Predictor": predictor,
                "MAE": mae,
                "Acierto": float(np.mean(error == 0)),
                "Dentro ±1": float(np.mean(error <= 1)),
                "Error grave": float(np.mean(error >= 2)),
                "MAE trivial": mae_trivial,
                "Mejora sobre trivial %": (
                    100.0
                    *
                    (mae_trivial - mae)
                    /
                    mae_trivial
                    if mae_trivial > 0
                    else np.nan
                )
            })

tabla = pd.DataFrame(filas)


# ----------------------------------------------------------------
# Presentación
# ----------------------------------------------------------------

print("=" * 100)
print("COMPARACIÓN ENTRE AGRUPACIONES")
print()
print("El MAE no es comparable entre agrupaciones.")
print("La columna de mejora relativa sobre el trivial sí lo es.")
print("=" * 100)

for esquema in esquemas:

    print()

    print(f"Esquema {esquema}")

    print()

    sub_tabla = tabla[
        tabla["Esquema"] == esquema
    ]

    print(
        sub_tabla
        .pivot_table(
            index="Agrupación",
            columns="Predictor",
            values="Mejora sobre trivial %",
            sort=False
        )
        .round(2)
        .to_string()
    )

print()

print("=" * 100)
print(f"DETALLE PARA {PREDICTOR_DETALLE}")
print("=" * 100)

print()

print(
    tabla[
        tabla["Predictor"] == PREDICTOR_DETALLE
    ][[
        "Esquema",
        "Agrupación",
        "Niveles",
        "MAE",
        "MAE trivial",
        "Mejora sobre trivial %",
        "Acierto",
        "Error grave"
    ]]
    .round(4)
    .to_string(index=False)
)

print()


# ----------------------------------------------------------------
# Acierto por clase bajo cada agrupación
# ----------------------------------------------------------------

print("=" * 100)
print(f"ACIERTO POR CLASE — {PREDICTOR_DETALLE}")
print()
print("Muestra si la clase superior se vuelve identificable")
print("al agrupar niveles.")
print("=" * 100)

for esquema in esquemas:

    sub = detalle[
        detalle["esquema"] == esquema
    ]

    print()

    print(f"Esquema {esquema}")

    for nombre_agrupacion, mapa in AGRUPACIONES.items():

        y_agrupada = aplicar(sub["y_true"], mapa)

        pred_agrupada = aplicar(
            sub[PREDICTOR_DETALLE],
            mapa
        )

        n_niveles = len(set(mapa.values()))

        partes = []

        for k in range(n_niveles):

            mascara = y_agrupada == k

            if mascara.sum() == 0:

                partes.append(f"{k}: -")

                continue

            acierto = np.mean(
                pred_agrupada[mascara] == k
            )

            modo = int(
                np.bincount(
                    pred_agrupada[mascara],
                    minlength=n_niveles
                ).argmax()
            )

            marca_modo = "*" if modo == k else " "

            partes.append(
                f"{k}: {acierto:.3f}{marca_modo}"
            )

        print(
            f"  {nombre_agrupacion:<24} "
            + "  ".join(partes)
        )

print()
print("El asterisco indica que la clase es el nivel predicho")
print("mas frecuente para los municipios que la componen.")
print("Sin asterisco, el modelo asigna esa clase con mayor")
print("frecuencia a otro nivel.")
print()


# ----------------------------------------------------------------
# Matriz de confusión de las agrupaciones reducidas
# ----------------------------------------------------------------

for nombre_agrupacion in [
    "4 niveles: fusion 3+4",
    "3 niveles: 1+2 y 3+4"
]:

    mapa = AGRUPACIONES[nombre_agrupacion]

    n_niveles = len(set(mapa.values()))

    for esquema in esquemas:

        sub = detalle[
            detalle["esquema"] == esquema
        ]

        matriz = pd.crosstab(
            aplicar(sub["y_true"], mapa),
            aplicar(sub[PREDICTOR_DETALLE], mapa)
        ).reindex(
            index=range(n_niveles),
            columns=range(n_niveles),
            fill_value=0
        )

        proporciones = matriz.div(
            matriz.sum(axis=1).replace(0, np.nan),
            axis=0
        )

        print("=" * 78)

        print(
            f"{nombre_agrupacion} — {esquema} — "
            f"{PREDICTOR_DETALLE}"
        )

        print("=" * 78)

        print(
            proporciones.round(3).to_string()
        )

        print()


ruta_salida = os.path.join(
    CARPETA,
    f"efecto_agrupacion_{marca}.csv"
)

tabla.to_csv(
    ruta_salida,
    index=False
)

print(f"Guardado: {ruta_salida}")

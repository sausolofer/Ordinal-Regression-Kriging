# ======================================================================
# 09 -- Descriptive figures and the association table
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Run from the repository root:
#     python scripts/09_figuras_descriptivas.py
#
# Inputs:  data/municipal_dengue_dataset_mexico_2022.csv
# Outputs: figuras/figura01_incidencia_positiva.{png,pdf}   (Figure 1)
#          figuras/figura02_niveles_riesgo.{png,pdf}        (Figure 2)
#          figuras/figura03_matriz_spearman.{png,pdf}       (Figure 3)
#          resultados/tabla02_asociaciones.{csv,tex}        (Table 2)
#
# No ajusta ningun modelo y no depende de la salida del script 01:
# todo se calcula directamente del conjunto de datos, en segundos.
#
# ADVERTENCIA SOBRE LA TABLA 2
#
# Los coeficientes de esta tabla y de la Figura 3 se calculan sobre la
# muestra completa y son descriptivos. NO son el conjunto de predictores
# que usa el modelo: ese se re-selecciona dentro de cada conjunto de
# entrenamiento (Seccion 3.4 del manuscrito), y su frecuencia de
# seleccion la reporta el script 01 en seleccion_variables_*.csv.
#
# The code and its comments are in Spanish, as in the rest of the
# repository; paths come from scripts/rutas.py.
# ======================================================================

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rutas
import figuras_png


# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

COLUMNA_NIVEL = "DRL"

COLUMNA_TASA = "Rate_2022"

COLUMNAS_EXCLUIDAS = [
    "Estado",
    "Municipio",
    COLUMNA_TASA,
    "lon_mun",
    "lat_mun"
]

N_CLASSES = 5

UMBRAL_SPEARMAN = 0.10

# Paleta de niveles, la misma de los mapas (scripts 06 a 08)

colores_nivel = [
    "#2ca25f",   # Level 0
    "#fee08b",   # Level 1
    "#fdae61",   # Level 2
    "#f46d43",   # Level 3
    "#8b0000"    # Level 4
]

N_BINS_FIG1 = 40

# La resolucion de los PNG la calcula figuras_png.dpi_png() a partir
# del ancho al que la figura se imprime en la pagina.


def guardar(fig, nombre):

    for extension in ("png", "pdf"):

        ruta = os.path.join(
            rutas.CARPETA_FIGURAS,
            f"{nombre}.{extension}"
        )

        fig.savefig(
            ruta,
            dpi=figuras_png.dpi_png(fig),
            bbox_inches="tight",
            facecolor="white"
        )

        if extension == "png":

            figuras_png.aplanar_png(ruta)

        print("  Guardado:", ruta)

    plt.close(fig)


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------

df = pd.read_csv(rutas.ARCHIVO_DATOS)

tasa = df[COLUMNA_TASA].to_numpy(dtype=float)

y = df[COLUMNA_NIVEL].to_numpy().ravel().astype(int)

n = len(df)

positivos = tasa[tasa > 0]

print("=" * 78)
print("DATOS")
print("=" * 78)

print("Municipios:", n)

print(
    f"Con incidencia positiva: {len(positivos)} "
    f"({100 * len(positivos) / n:.2f}%)"
)

print(
    f"Sin casos reportados: {n - len(positivos)} "
    f"({100 * (n - len(positivos)) / n:.2f}%)"
)

cuartiles = np.quantile(positivos, [0.25, 0.50, 0.75])

print(
    "Cuartiles de la incidencia positiva: "
    + ", ".join(f"{c:.3f}" for c in cuartiles)
)

print()


# ======================================================================
# FIGURA 1: distribución de la incidencia positiva
# ======================================================================

print("=" * 78)
print("FIGURA 1: distribución de la incidencia positiva")
print("=" * 78)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    positivos,
    bins=N_BINS_FIG1,
    color="#1f77b4",
    edgecolor="white",
    linewidth=0.6,
    zorder=3
)

ax.set_xlabel(
    "Reported dengue incidence rate "
    "(cases per 100,000 inhabitants)",
    fontsize=11
)

ax.set_ylabel("Number of municipalities", fontsize=11)

ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.35,
    zorder=0
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()

guardar(fig, "figura01_incidencia_positiva")

print(
    f"  Rango: {positivos.min():.2f} a {positivos.max():.2f} "
    f"casos por 100,000 habitantes"
)

print()


# ======================================================================
# FIGURA 2: distribución de los niveles de riesgo
# ======================================================================

print("=" * 78)
print("FIGURA 2: distribución de los niveles de riesgo")
print("=" * 78)

conteos = np.bincount(y, minlength=N_CLASSES)

porcentajes = 100.0 * conteos / n

for nivel in range(N_CLASSES):

    print(
        f"  Nivel {nivel}: {conteos[nivel]:4d} municipios "
        f"({porcentajes[nivel]:.2f}%)"
    )

fig, ax = plt.subplots(figsize=(8, 5.5))

barras = ax.bar(
    np.arange(N_CLASSES),
    conteos,
    color=colores_nivel,
    edgecolor="black",
    linewidth=0.6,
    width=0.75,
    zorder=3
)

for barra, porcentaje in zip(barras, porcentajes):

    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 0.012 * conteos.max(),
        f"{porcentaje:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        zorder=4
    )

ax.set_xlabel("Dengue Risk Level (DRL)", fontsize=11)

ax.set_ylabel("Number of municipalities", fontsize=11)

ax.set_xticks(np.arange(N_CLASSES))

ax.set_ylim(0, conteos.max() * 1.10)

ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.35,
    zorder=0
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()

guardar(fig, "figura02_niveles_riesgo")

print()


# ======================================================================
# FIGURA 3: matriz de correlación de Spearman
#
# Triangular inferior, con la diagonal enmascarada, en el orden de
# columnas del conjunto de datos. La respuesta DRL conserva su posición
# natural entre MODPOV y SEVI.
# ======================================================================

print("=" * 78)
print("FIGURA 3: matriz de correlación de Spearman")
print("=" * 78)

columnas = [
    c for c in df.columns
    if c not in COLUMNAS_EXCLUIDAS
]

matriz = df[columnas].corr(method="spearman")

print("Variables en la matriz:", len(columnas))

valores = matriz.to_numpy()

mascara = np.triu(
    np.ones_like(valores, dtype=bool),
    k=0
)

visible = np.ma.array(valores, mask=mascara)

limite = float(np.nanmax(np.abs(valores[~mascara])))

# ----------------------------------------------------------------------
# Composición de la figura
#
# Las proporciones de abajo se midieron sobre la figura 3 del manuscrito
# enviado, para que esta reproducción coincida con la publicada. Todas
# están expresadas como fracción del lado de la matriz, así que la
# figura se reescala entera cambiando sólo LADO_MATRIZ.
#
# La barra de color es más alta que la matriz y queda separada de ella
# por un hueco. Eso no sale de fig.colorbar(ax=ax), que ajusta la barra
# a la caja de ejes; hay que colocar ambas cajas a mano.
# ----------------------------------------------------------------------

LADO_MATRIZ = 10.0          # pulgadas; sólo fija la escala

HUECO_BARRA = 0.0626        # separación entre la matriz y la barra

ANCHO_BARRA = 0.0578

ALTO_BARRA = 1.1537         # la barra sobresale arriba y abajo

# Tipografías, en puntos por pulgada de lado de matriz, para que
# conserven su tamaño relativo si se cambia LADO_MATRIZ.

PUNTOS_ANOTACION = 0.58

PUNTOS_ETIQUETA = 0.70

PUNTOS_BARRA = 0.65

# Margen izquierdo para las etiquetas de las variables, y derecho para
# los rótulos de la barra; medidos también sobre la figura publicada.
# El lienzo se ajusta a lo que ocupa el contenido, porque savefig recorta
# con bbox_inches="tight" y un lienzo holgado bajaría la resolución útil
# del PNG al recortarse.

MARGEN_ETIQUETAS = 0.069

MARGEN_ROTULOS = 0.046

ancho_figura = LADO_MATRIZ * (
    MARGEN_ETIQUETAS + 1 + HUECO_BARRA + ANCHO_BARRA + MARGEN_ROTULOS
)

alto_figura = LADO_MATRIZ * ALTO_BARRA * 1.02

fig = plt.figure(figsize=(ancho_figura, alto_figura))

borde_izquierdo = MARGEN_ETIQUETAS * LADO_MATRIZ / ancho_figura

borde_inferior = (1 - LADO_MATRIZ / alto_figura) / 2

ax = fig.add_axes([
    borde_izquierdo,
    borde_inferior,
    LADO_MATRIZ / ancho_figura,
    LADO_MATRIZ / alto_figura
])

eje_barra = fig.add_axes([
    borde_izquierdo + (1 + HUECO_BARRA) * LADO_MATRIZ / ancho_figura,
    borde_inferior - (ALTO_BARRA - 1) / 2 * LADO_MATRIZ / alto_figura,
    ANCHO_BARRA * LADO_MATRIZ / ancho_figura,
    ALTO_BARRA * LADO_MATRIZ / alto_figura
])

imagen = ax.imshow(
    visible,
    cmap="coolwarm",
    vmin=-limite,
    vmax=limite,
    aspect="equal"
)

# Anotación de cada celda visible

for i in range(len(columnas)):

    for j in range(i):

        valor = valores[i, j]

        ax.text(
            j,
            i,
            f"{valor:.2f}",
            ha="center",
            va="center",
            fontsize=PUNTOS_ANOTACION * LADO_MATRIZ,
            color=(
                "white"
                if abs(valor) > 0.60 * limite
                else "black"
            )
        )

ax.set_xticks(np.arange(len(columnas)))

ax.set_yticks(np.arange(len(columnas)))

ax.set_xticklabels(
    columnas,
    rotation=45,
    ha="right",
    fontsize=PUNTOS_ETIQUETA * LADO_MATRIZ
)

ax.set_yticklabels(
    columnas,
    fontsize=PUNTOS_ETIQUETA * LADO_MATRIZ
)

# Separadores blancos entre celdas

ax.set_xticks(
    np.arange(-0.5, len(columnas), 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, len(columnas), 1),
    minor=True
)

ax.grid(
    which="minor",
    color="white",
    linewidth=0.6
)

ax.tick_params(which="minor", length=0)

for lado in ("top", "right", "bottom", "left"):

    ax.spines[lado].set_visible(False)

barra = fig.colorbar(imagen, cax=eje_barra)

barra.ax.tick_params(labelsize=PUNTOS_BARRA * LADO_MATRIZ)

# Sin tight_layout: las cajas están puestas a mano y tight_layout las
# recolocaría, deshaciendo la composición medida arriba.

guardar(fig, "figura03_matriz_spearman")

print()


# ======================================================================
# TABLA 2: variables que satisfacen el criterio sobre la muestra completa
# ======================================================================

print("=" * 78)
print("TABLA 2: asociaciones con la respuesta")
print("=" * 78)

rho = matriz[COLUMNA_NIVEL].drop(COLUMNA_NIVEL)

tabla = (
    pd.DataFrame({
        "Variable": rho.index,
        "Spearman correlation": rho.to_numpy()
    })
    .assign(magnitud=lambda d: d["Spearman correlation"].abs())
    .sort_values("magnitud", ascending=False)
    .reset_index(drop=True)
)

tabla["Retained"] = tabla["magnitud"] >= UMBRAL_SPEARMAN

retenidas = tabla[tabla["Retained"]].drop(columns="magnitud")

print(
    f"Variables con |rho| >= {UMBRAL_SPEARMAN}: "
    f"{len(retenidas)} de {len(tabla)}"
)

print()

print(
    retenidas[["Variable", "Spearman correlation"]]
    .round(3)
    .to_string(index=False)
)

ruta_csv = os.path.join(
    rutas.CARPETA_RESULTADOS,
    "tabla02_asociaciones.csv"
)

tabla.drop(columns="magnitud").round(4).to_csv(
    ruta_csv,
    index=False
)

ruta_tex = os.path.join(
    rutas.CARPETA_RESULTADOS,
    "tabla02_asociaciones.tex"
)

with open(ruta_tex, "w", encoding="utf-8") as f:

    f.write(
        retenidas[["Variable", "Spearman correlation"]]
        .round(3)
        .to_latex(
            index=False,
            caption=(
                "Explanatory variables satisfying the selection "
                "criterion ($|\\rho| \\geq 0.10$), with their Spearman "
                "correlation with the Dengue Risk Level. Coefficients "
                "computed on the complete sample; the predictor set "
                "used by the ORK framework was re-selected within each "
                "training partition."
            ),
            label="tab:asociaciones"
        )
    )

print()
print("  Guardado:", ruta_csv)
print("  Guardado:", ruta_tex)

print()
print(
    "Recordatorio: estos coeficientes son descriptivos. El conjunto de "
    "predictores que usa el modelo se re-selecciona dentro de cada "
    "particion de entrenamiento; vease seleccion_variables_*.csv."
)

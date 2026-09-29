# ======================================================================
# 08 -- Choropleth of the observed municipal risk levels
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Run from the repository root:
#     python scripts/08_coropletas_observado.py
#
# Inputs:  data/ + data/shapefile/00mun.shp
# Outputs: figuras/figura11_observado_coropletas.*
#
# NOTE: the output file names below use the figure numbering of an
# earlier draft. See docs/reproducing.md for the mapping onto the
# figure numbers of the submitted manuscript.
#
# The code and its comments are in Spanish, as written by the authors.
# Only the data, shapefile and output paths were changed when the
# analysis was organised as a repository; see scripts/rutas.py.
# ======================================================================

# ======================================================================
# FIGURA 11
# MAPA DE NIVELES DE RIESGO OBSERVADOS
#
#
# La union entre la tabla de datos y el shapefile se hace por
# localizacion: cada cabecera municipal (lon_mun, lat_mun) se
# asigna al poligono que la contiene. Asi no se depende de
# que coincidan claves o nombres entre las dos fuentes.
# ======================================================================

from pathlib import Path

import numpy as np
import pandas as pd

import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import rutas
import figuras_png


# ======================================================================
# 1. CONFIGURACIÓN
# ======================================================================

ARCHIVO_DATOS = rutas.ARCHIVO_DATOS


CARPETA_SHAPE = Path(rutas.RUTA_SHAPEFILE).parent

RUTA_SHAPEFILE = Path(rutas.RUTA_SHAPEFILE)

CARPETA_SALIDA = Path(rutas.CARPETA_FIGURAS)

COLUMNA_NIVEL = "DRL"

columna_longitud = "lon_mun"
columna_latitud = "lat_mun"

N_CLASSES = 5

# ------------------------------------------------
# Simplificación de las geometrías para el dibujo
#
# Los polígonos del Marco Geoestadístico tienen 3.6 millones de vértices, que
# en un PDF vectorial pesan unos 30 MB por figura y hacen lento el documento.
# La simplificación se aplica únicamente a las geometrías que se dibujan,
# después de todas las uniones espaciales, así que no altera ningún resultado.
#
# La tolerancia está en grados: 0.002 equivale a unos 220 m y elimina el 94%
# de los vértices. A la escala de la página la diferencia no es perceptible.
# Poner None para dibujar las geometrías completas.
# ------------------------------------------------

TOLERANCIA_SIMPLIFICACION = 0.002


def simplificar(gdf):
    """Devuelve una copia con la geometría simplificada, o el original."""

    if TOLERANCIA_SIMPLIFICACION is None:

        return gdf

    copia = gdf.copy()

    return copia.set_geometry(
        copia.geometry.simplify(TOLERANCIA_SIMPLIFICACION)
    )





# ------------------------------------------------
# Contorno exterior del país
#
# dissolve() une los 2,469 municipios en una sola geometría, pero los
# polígonos del Marco Geoestadístico no comparten exactamente sus
# vértices a lo largo de los límites comunes. La unión deja por eso
# 13,473 huecos interiores: ranuras de unos pocos metros cuadrados
# entre municipios vecinos. Como superficie son invisibles, pero cada
# una es una curva cerrada, y boundary.plot() las dibuja con el mismo
# trazo negro grueso del contorno nacional. Donde los municipios son
# pequeños y numerosos -- el centro del país, Tlaxcala, Oaxaca -- se
# acumulan hasta verse como manchas negras sobre el mapa.
#
# Se conservan únicamente los huecos de área mayor o igual que
# AREA_MINIMA_HUECO. Con 1e-6 grados cuadrados (~1.2 ha) sobreviven los
# dos huecos reales del país (el estero entre Empalme y Guaymas,
# Sonora) y desaparecen las 13,471 ranuras. La ranura más grande mide
# 7.7e-9 y el hueco real más pequeño 1.8e-5, así que entre unas y otros
# hay tres órdenes de magnitud: el umbral no es delicado.
#
# El contorno debe construirse ANTES de simplificar. Simplificar cada
# municipio por separado y unirlos después abre ranuras de hasta el
# doble de la tolerancia, que ya no son despreciables.
# ------------------------------------------------

AREA_MINIMA_HUECO = 1e-6


def contorno_exterior(gdf):
    """Une las geometrías y descarta las ranuras que deja la unión."""

    from shapely.geometry import MultiPolygon, Polygon

    def sin_ranuras(geometria):

        partes = (
            list(geometria.geoms)
            if geometria.geom_type == "MultiPolygon"
            else [geometria]
        )

        limpias = [
            Polygon(
                parte.exterior,
                [
                    anillo
                    for anillo in parte.interiors
                    if Polygon(anillo).area >= AREA_MINIMA_HUECO
                ]
            )
            for parte in partes
        ]

        return (
            MultiPolygon(limpias)
            if len(limpias) > 1
            else limpias[0]
        )

    disuelto = gdf.dissolve()

    return disuelto.set_geometry(
        disuelto.geometry.apply(sin_ranuras)
    )


ETIQUETA_BARRA = "Observed dengue risk level"

TITULO_LEYENDA = "Observed risk levels"

TITULO_MAPA = (
    "Observed municipal dengue risk levels "
    "in Mexico 2022"
)

NOMBRE_SALIDA = "figura11_observado_coropletas"


CARPETA_SALIDA.mkdir(exist_ok=True)


# ======================================================================
# 2. PALETA DISCRETA
# ======================================================================

colores_nivel = [
    "#2ca25f",   # Level 0
    "#fee08b",   # Level 1
    "#fdae61",   # Level 2
    "#f46d43",   # Level 3
    "#8b0000"    # Level 4
]

cmap_niveles = mcolors.ListedColormap(colores_nivel)

norm_niveles = mcolors.BoundaryNorm(
    boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
    ncolors=cmap_niveles.N
)


# ======================================================================
# 3. DATOS
# ======================================================================

df = pd.read_csv(rutas.ARCHIVO_DATOS)


print("=" * 90)
print("DATOS")
print("=" * 90)

print("Municipios en la tabla:", len(df))

print()

print(
    df[COLUMNA_NIVEL]
    .value_counts()
    .sort_index()
)

print()


# ======================================================================
# 4. SHAPEFILE
# ======================================================================

if not RUTA_SHAPEFILE.exists():

    raise FileNotFoundError(
        f"No se encontró el shapefile en:\n"
        f"{RUTA_SHAPEFILE.resolve()}"
    )

gdf_municipios = gpd.read_file(RUTA_SHAPEFILE)

print("=" * 90)
print("SHAPEFILE")
print("=" * 90)

print("Polígonos:", len(gdf_municipios))

print("CRS original:", gdf_municipios.crs)

if gdf_municipios.crs is None:

    raise ValueError(
        "El shapefile no tiene un sistema de referencia "
        "definido."
    )

gdf_municipios = gdf_municipios.to_crs(epsg=4326)

print("CRS utilizado:", gdf_municipios.crs)

print()


# ======================================================================
# 5. UNIÓN ESPACIAL PUNTO-POLÍGONO
# ======================================================================

puntos = gpd.GeoDataFrame(
    df[[COLUMNA_NIVEL]].copy(),
    geometry=gpd.points_from_xy(
        df[columna_longitud],
        df[columna_latitud]
    ),
    crs="EPSG:4326"
)

union = gpd.sjoin(
    puntos,
    gdf_municipios[["geometry"]],
    how="left",
    predicate="within"
)

sin_poligono = union["index_right"].isna().sum()

print("=" * 90)
print("UNIÓN ESPACIAL")
print("=" * 90)

print(
    "Municipios asignados a un polígono:",
    len(union) - sin_poligono
)

print(
    "Sin polígono contenedor:",
    sin_poligono
)

if sin_poligono > 0:

    print(
        "\nAdvertencia: algunas cabeceras no caen dentro de "
        "ningún polígono.\nSuele deberse a cabeceras muy "
        "próximas al litoral o a diferencias de\nprecisión "
        "entre fuentes. Esos municipios se dibujarán como "
        "sin información."
    )

# Nivel observado por polígono.
# Si dos cabeceras cayeran en el mismo poligono se conserva
# la de mayor categoria, situacion que en la practica no
# deberia presentarse.

asignacion = (
    union
    .dropna(subset=["index_right"])
    .groupby("index_right")[COLUMNA_NIVEL]
    .max()
)

gdf_modelo = gdf_municipios.loc[
    asignacion.index.astype(int)
].copy()

gdf_modelo[COLUMNA_NIVEL] = asignacion.values.astype(int)

print(
    "\nPolígonos con categoría observada:",
    len(gdf_modelo)
)

print()

print(
    gdf_modelo[COLUMNA_NIVEL]
    .value_counts()
    .sort_index()
)

print()


# ======================================================================
# 6. EXTENSIÓN DEL MAPA
# ======================================================================

min_lon, min_lat, max_lon, max_lat = (
    gdf_municipios.total_bounds
)

margen = 0.5

min_lon -= margen
max_lon += margen
min_lat -= margen
max_lat += margen


# ======================================================================
# 6b. SIMPLIFICACIÓN PARA EL DIBUJO
#
# La unión espacial y la extensión del mapa ya se calcularon con las
# geometrías completas; lo que sigue es únicamente dibujo.
# ======================================================================

contorno_nacional = simplificar(contorno_exterior(gdf_municipios))

gdf_municipios = simplificar(gdf_municipios)

gdf_modelo = simplificar(gdf_modelo)


# ======================================================================
# 7. FIGURA
# ======================================================================

fig, ax = plt.subplots(figsize=(14, 9))


# Municipios sin información: fondo y límites

gdf_municipios.plot(
    ax=ax,
    facecolor="#f2f2f2",
    edgecolor="#c7c7c7",
    linewidth=0.12,
    zorder=1
)


# Municipios con categoría observada

gdf_modelo.plot(
    ax=ax,
    column=COLUMNA_NIVEL,
    cmap=cmap_niveles,
    norm=norm_niveles,
    edgecolor="#8c8c8c",
    linewidth=0.12,
    zorder=2
)


# Contorno nacional

contorno_nacional.boundary.plot(
    ax=ax,
    edgecolor="black",
    linewidth=0.70,
    zorder=3
)


# ----------------------------------------------------------------------
# Barra de color discreta
# ----------------------------------------------------------------------

sm = plt.cm.ScalarMappable(
    cmap=cmap_niveles,
    norm=norm_niveles
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation="vertical",
    fraction=0.030,
    pad=0.025
)

cbar.set_label(
    ETIQUETA_BARRA,
    fontsize=11
)

cbar.set_ticks([0, 1, 2, 3, 4])

cbar.set_ticklabels(["0", "1", "2", "3", "4"])

cbar.ax.tick_params(labelsize=9)


# ----------------------------------------------------------------------
# Leyenda de niveles
# ----------------------------------------------------------------------

leyenda = [
    Patch(
        facecolor=colores_nivel[nivel],
        edgecolor="#777777",
        linewidth=0.6,
        label=f"Level {nivel}"
    )
    for nivel in range(N_CLASSES)
]

ax.legend(
    handles=leyenda,
    title=TITULO_LEYENDA,
    loc="lower left",
    frameon=True,
    framealpha=0.95,
    fontsize=9,
    title_fontsize=10,
    borderpad=0.8,
    labelspacing=0.6
)


# ----------------------------------------------------------------------
# Títulos, ejes y cuadrícula
# ----------------------------------------------------------------------

ax.set_title(
    TITULO_MAPA,
    fontsize=16,
    fontweight="bold",
    pad=14
)

ax.set_xlabel("Longitude", fontsize=11)

ax.set_ylabel("Latitude", fontsize=11)

ax.set_xlim(min_lon, max_lon)

ax.set_ylim(min_lat, max_lat)

ax.grid(linewidth=0.25, alpha=0.30)

ax.tick_params(axis="both", labelsize=9)

ax.set_xticks([-115, -110, -105, -100, -95, -90])

ax.set_yticks([15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5])

plt.tight_layout()


# ======================================================================
# 8. GUARDADO
# ======================================================================

ruta_png = CARPETA_SALIDA / f"{NOMBRE_SALIDA}.png"

ruta_pdf = CARPETA_SALIDA / f"{NOMBRE_SALIDA}.pdf"

fig.savefig(
    ruta_png,
    dpi=figuras_png.dpi_png(fig),
    bbox_inches="tight",
    facecolor="white"
)

figuras_png.aplanar_png(ruta_png)

fig.savefig(
    ruta_pdf,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("=" * 90)
print("MAPA GUARDADO")
print("=" * 90)

print("PNG:", ruta_png.resolve())

print("PDF:", ruta_pdf.resolve())

print()

print(
    f"Para el pie de figura: {len(gdf_modelo)} de "
    f"{len(df)} municipios de la base de datos quedaron "
    f"representados; los polígonos restantes corresponden a "
    f"municipios sin información en el estudio."
)

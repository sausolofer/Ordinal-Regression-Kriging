# ======================================================================
# 07 -- Final-model maps: continuous score surface and out-of-sample choropleth
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Run from the repository root:
#     python scripts/07_mapas_modelo_final.py
#
# Inputs:  data/ + data/shapefile/00mun.shp + resultados/seleccion_variables_*.csv
# Outputs: figuras/mapaA_superficie_continua.*, mapaB_coropletas_oof.*
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
# MAPAS DEL MODELO FINAL
#
# Produce dos mapas complementarios con la metodologia
# actual (Random Forest ordinal acumulativo, variograma
# exponencial, umbral calibrado):
#
#   MAPA A  Superficie continua sobre malla regular.
#           Reproduce el enfoque del manuscrito: cada nodo
#           hereda las covariables del municipio que lo
#           contiene y recibe la correccion krigeada
#           evaluada en su propia ubicacion.
#           Es una representacion del ajuste;
#           en las cabeceras el
#           valor corregido reproduce la categoria
#           observada.
#
#   MAPA B  Coropletas con prediccion fuera de muestra.
#           Cada municipio recibe el valor esperado de un
#           modelo que no lo vio y una correccion krigeada
#           interpolada desde los demas. 
#
# VARIABLES EXPLICATIVAS
#
# Se fija el conjunto de variables con frecuencia de
# seleccion mayor o igual al umbral indicado en la tabla de
# estabilidad del analisis principal. 
# ======================================================================

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold

from pykrige.ok import OrdinaryKriging

import rutas
import figuras_png


# ======================================================================
# 1. CONFIGURACIÓN
# ======================================================================

ARCHIVO_DATOS = rutas.ARCHIVO_DATOS

CARPETA_SHAPE = Path(rutas.RUTA_SHAPEFILE).parent

RUTA_SHAPEFILE = Path(rutas.RUTA_SHAPEFILE)

CARPETA_RESULTADOS = rutas.CARPETA_RESULTADOS

CARPETA_SALIDA = Path(rutas.CARPETA_FIGURAS)


# Nombre de la columna de respuesta en el CSV

COLUMNA_NIVEL = "DRL"

columna_longitud = "lon_mun"
columna_latitud = "lat_mun"


# Columnas que no son covariables

COLUMNAS_EXCLUIDAS = [
    "Estado",
    "Municipio",
    "Rate_2022",
    columna_longitud,
    columna_latitud,
    COLUMNA_NIVEL
]


# Frecuencia mínima de selección, en por ciento

FRECUENCIA_MINIMA = 90.0


GLOBAL_SEED = 123

N_CLASSES = 5

N_FOLDS_OOF = 5

N_FOLDS_KRIGING_OOF = 3

VARIOGRAMA = "exponential"

N_LAGS = 12

N_VECINOS_KRIGING = 50

REJILLA_UMBRAL = np.arange(0.05, 0.96, 0.01)


# Malla del Mapa A

N_LON = 250
N_LAT = 250

CHUNK_KRIGING = 2000

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


CARPETA_SALIDA.mkdir(exist_ok=True)


# ======================================================================
# 2. PALETAS
# ======================================================================

# Continua, para el Mapa A

colores_continuos = [
    "#238443",
    "#78c679",
    "#fee08b",
    "#fdae61",
    "#d73027",
    "#7f0000"
]

cmap_riesgo = mcolors.LinearSegmentedColormap.from_list(
    "dengue_incidence",
    colores_continuos,
    N=256
)

cmap_riesgo.set_bad(alpha=0)


# Discreta, para el Mapa B

colores_nivel = [
    "#2ca25f",
    "#fee08b",
    "#fdae61",
    "#f46d43",
    "#8b0000"
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

y_ordinal = (
    df[COLUMNA_NIVEL].to_numpy().ravel().astype(int)
)

coords = df[
    [columna_longitud, columna_latitud]
].to_numpy(dtype=float)

print("=" * 90)
print("DATOS")
print("=" * 90)

print("Municipios:", len(df))

print()


# ======================================================================
# 4. VARIABLES EXPLICATIVAS ESTABLES
# ======================================================================

archivos_estabilidad = sorted(
    glob.glob(
        os.path.join(
            CARPETA_RESULTADOS,
            "seleccion_variables_*.csv"
        )
    )
)

if not archivos_estabilidad:

    raise SystemExit(
        "No se encontró la tabla de estabilidad de "
        "variables en "
        f"{CARPETA_RESULTADOS}."
    )

ruta_estabilidad = archivos_estabilidad[-1]

tabla_estabilidad = pd.read_csv(ruta_estabilidad)

seleccionadas = tabla_estabilidad.loc[
    tabla_estabilidad["Porcentaje"] >= FRECUENCIA_MINIMA,
    "Variable"
].tolist()

# Conservar solo las que existen en el CSV

variables_modelo = [
    v for v in seleccionadas
    if v in df.columns
]

faltantes = set(seleccionadas) - set(variables_modelo)

print("=" * 90)
print("VARIABLES EXPLICATIVAS")
print("=" * 90)

print("Tabla de estabilidad:", ruta_estabilidad)

print(
    f"Frecuencia mínima exigida: {FRECUENCIA_MINIMA}%"
)

print(
    f"Variables seleccionadas: {len(variables_modelo)}"
)

for nombre in variables_modelo:

    porcentaje = tabla_estabilidad.loc[
        tabla_estabilidad["Variable"] == nombre,
        "Porcentaje"
    ].iloc[0]

    print(f"  {porcentaje:6.2f}%  {nombre}")

if faltantes:

    print(
        "\nAdvertencia: no están en el CSV de datos: "
        + ", ".join(sorted(faltantes))
    )

if not variables_modelo:

    raise SystemExit(
        "Ninguna variable superó el umbral de frecuencia."
    )

X_full = df[variables_modelo].to_numpy(dtype=float)

if not np.isfinite(X_full).all():

    raise ValueError(
        "Las covariables contienen NaN o infinitos."
    )

print()


# ======================================================================
# 5. MODELO ORDINAL ACUMULATIVO
# ======================================================================

class CumulativeOrdinalRF:

    def __init__(self, random_state=GLOBAL_SEED):

        self.base = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1
        )

        self.models_ = []

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)

        y = np.asarray(y).ravel().astype(int)

        self.models_ = []

        for umbral in range(N_CLASSES - 1):

            y_binaria = (y > umbral).astype(int)

            presentes = np.unique(y_binaria)

            if presentes.size < 2:

                self.models_.append(
                    ("constante", float(presentes[0]))
                )

                continue

            modelo = clone(self.base)

            modelo.fit(X, y_binaria)

            self.models_.append(("modelo", modelo))

        return self

    def _umbral_probs(self, X):

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
                int(e): i
                for i, e in enumerate(objeto.classes_)
            }

            columnas.append(
                probabilidades[:, posiciones[1]]
                if 1 in posiciones
                else np.zeros(n)
            )

        q = np.column_stack(columnas)

        return np.clip(
            np.minimum.accumulate(q, axis=1), 0.0, 1.0
        )

    def predict_expected_value(self, X):

        return self._umbral_probs(X).sum(axis=1)


def valor_esperado_por_bloques(
    modelo,
    X,
    tamano=5000,
    etiqueta="Random Forest ordinal"
):

    salida = []

    total = len(X)

    for inicio in range(0, total, tamano):

        fin = min(inicio + tamano, total)

        salida.append(
            modelo.predict_expected_value(X[inicio:fin])
        )

        print(
            f"\r{etiqueta}: {fin:,}/{total:,}",
            end=""
        )

    print()

    return np.concatenate(salida)


# ======================================================================
# 6. KRIGING
# ======================================================================

def ajustar_kriging(coords_train, valores):

    return OrdinaryKriging(
        coords_train[:, 0],
        coords_train[:, 1],
        valores,
        variogram_model=VARIOGRAMA,
        nlags=N_LAGS,
        weight=True,
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic",
        exact_values=True,
        pseudo_inv=True
    )


def kriging_por_bloques(
    modelo_kriging,
    longitudes,
    latitudes,
    n_train,
    tamano=CHUNK_KRIGING
):

    n_cercanos = int(
        min(N_VECINOS_KRIGING, max(1, n_train - 1))
    )

    salida = []

    total = len(longitudes)

    for inicio in range(0, total, tamano):

        fin = min(inicio + tamano, total)

        z, _ = modelo_kriging.execute(
            style="points",
            xpoints=np.asarray(
                longitudes[inicio:fin], dtype=float
            ),
            ypoints=np.asarray(
                latitudes[inicio:fin], dtype=float
            ),
            backend="loop",
            n_closest_points=n_cercanos
        )

        salida.append(np.asarray(z, dtype=float).ravel())

        print(f"\rKriging: {fin:,}/{total:,}", end="")

    print()

    return np.concatenate(salida)


def calibrar_umbral(valores, y_true):

    valores = np.asarray(valores, dtype=float)

    y_true = np.asarray(y_true).astype(int)

    validos = np.isfinite(valores)

    v = valores[validos]

    objetivo = y_true[validos]

    errores = np.array([
        np.mean(
            np.abs(
                np.clip(np.floor(v + t), 0, N_CLASSES - 1)
                -
                objetivo
            )
        )
        for t in REJILLA_UMBRAL
    ])

    empatados = REJILLA_UMBRAL[
        errores <= errores.min() + 1e-12
    ]

    return float(np.median(empatados))


# ======================================================================
# 7. VALORES ESPERADOS FUERA DE MUESTRA Y RESIDUOS
# ======================================================================

print("=" * 90)
print("VALORES ESPERADOS FUERA DE MUESTRA")
print("=" * 90)

divisor = StratifiedKFold(
    n_splits=N_FOLDS_OOF,
    shuffle=True,
    random_state=GLOBAL_SEED
)

esperado_oof = np.full(len(y_ordinal), np.nan)

for numero, (idx_a, idx_v) in enumerate(
    divisor.split(X_full, y_ordinal),
    start=1
):

    modelo = CumulativeOrdinalRF(
        random_state=GLOBAL_SEED + numero
    )

    modelo.fit(X_full[idx_a], y_ordinal[idx_a])

    esperado_oof[idx_v] = modelo.predict_expected_value(
        X_full[idx_v]
    )

    print(f"  pliegue {numero} de {N_FOLDS_OOF}")

residuos = y_ordinal.astype(float) - esperado_oof

print(
    f"Residuos: media {residuos.mean():.3f}, "
    f"desviación estándar {residuos.std(ddof=1):.3f}"
)

print()


# ----------------------------------------------------------------------
# Kriging ajustado sobre los residuos fuera de muestra
# ----------------------------------------------------------------------

ok_residuos = ajustar_kriging(coords, residuos)

parametros = np.asarray(
    ok_residuos.variogram_model_parameters,
    dtype=float
).ravel()

print(
    f"Variograma {VARIOGRAMA}; "
    f"sill, rango y nugget: {np.round(parametros, 4)}"
)


# ----------------------------------------------------------------------
# Correcciones krigeadas fuera de muestra y umbral
# ----------------------------------------------------------------------

kf = KFold(
    n_splits=N_FOLDS_KRIGING_OOF,
    shuffle=True,
    random_state=GLOBAL_SEED
)

correccion_oof = np.full(len(residuos), np.nan)

for idx_a, idx_v in kf.split(coords):

    ok_parcial = ajustar_kriging(
        coords[idx_a], residuos[idx_a]
    )

    correccion_oof[idx_v] = kriging_por_bloques(
        ok_parcial,
        coords[idx_v, 0],
        coords[idx_v, 1],
        len(idx_a)
    )

umbral = calibrar_umbral(
    esperado_oof + correccion_oof,
    y_ordinal
)

print(f"Umbral calibrado: {umbral:.3f}")

print()


# ----------------------------------------------------------------------
# Modelo final con todos los municipios, para el Mapa A
# ----------------------------------------------------------------------

modelo_final = CumulativeOrdinalRF(
    random_state=GLOBAL_SEED
)

modelo_final.fit(X_full, y_ordinal)


# ======================================================================
# 8. SHAPEFILE Y UNIÓN CON LOS DATOS
# ======================================================================

if not RUTA_SHAPEFILE.exists():

    raise FileNotFoundError(
        f"No se encontró el shapefile en:\n"
        f"{RUTA_SHAPEFILE.resolve()}"
    )

gdf_municipios = gpd.read_file(RUTA_SHAPEFILE)

if gdf_municipios.crs is None:

    raise ValueError("El shapefile no tiene CRS definido.")

gdf_municipios = gdf_municipios.to_crs(epsg=4326)

gdf_mexico = contorno_exterior(gdf_municipios)

print("=" * 90)
print("SHAPEFILE")
print("=" * 90)

print("Polígonos:", len(gdf_municipios))


# Unión espacial de cabeceras a polígonos

puntos = gpd.GeoDataFrame(
    pd.DataFrame({"fila": np.arange(len(df))}),
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

union = union.dropna(subset=["index_right"])

union = union.drop_duplicates(subset=["fila"])

print(
    "Municipios asignados a un polígono:",
    len(union),
    "de",
    len(df)
)

print()

filas_datos = union["fila"].to_numpy(dtype=int)

indices_poligono = union["index_right"].to_numpy(dtype=int)


# ======================================================================
# 9. MAPA A: SUPERFICIE CONTINUA SOBRE MALLA
# ======================================================================

print("=" * 90)
print("MAPA A: superficie continua sobre malla")
print("=" * 90)

# gdf_modelo lleva las covariables de cada municipio

gdf_modelo = gdf_municipios.loc[indices_poligono].copy()

gdf_modelo = gdf_modelo.reset_index(drop=True)

for variable in variables_modelo:

    gdf_modelo[variable] = (
        df[variable].to_numpy()[filas_datos]
    )

gdf_modelo["nivel_observado"] = y_ordinal[filas_datos]


min_lon, min_lat, max_lon, max_lat = gdf_mexico.total_bounds

lon_grid = np.linspace(min_lon, max_lon, N_LON)

lat_grid = np.linspace(min_lat, max_lat, N_LAT)

grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)

grid_col, grid_row = np.meshgrid(
    np.arange(N_LON),
    np.arange(N_LAT)
)

gdf_grid = gpd.GeoDataFrame(
    pd.DataFrame({
        "lon_grid": grid_lon.ravel(),
        "lat_grid": grid_lat.ravel(),
        "grid_row": grid_row.ravel(),
        "grid_col": grid_col.ravel()
    }),
    geometry=gpd.points_from_xy(
        grid_lon.ravel(),
        grid_lat.ravel()
    ),
    crs="EPSG:4326"
)

print(f"Resolución: {N_LON} × {N_LAT}")

print(f"Nodos totales: {len(gdf_grid):,}")

gdf_grid_modelo = gpd.sjoin(
    gdf_grid,
    gdf_modelo[["geometry"] + variables_modelo],
    how="inner",
    predicate="within"
)

gdf_grid_modelo = (
    gdf_grid_modelo
    .sort_values(["grid_row", "grid_col"])
    .drop_duplicates(
        subset=["grid_row", "grid_col"],
        keep="first"
    )
    .reset_index(drop=True)
)

print(
    "Nodos dentro de municipios con datos:",
    f"{len(gdf_grid_modelo):,}",
    f"({100 * len(gdf_grid_modelo) / len(gdf_grid):.2f}%)"
)

X_grid = gdf_grid_modelo[
    variables_modelo
].to_numpy(dtype=float)

if not np.isfinite(X_grid).all():

    raise ValueError(
        "La malla contiene covariables faltantes."
    )

lon_prediccion = gdf_grid_modelo[
    "lon_grid"
].to_numpy(dtype=float)

lat_prediccion = gdf_grid_modelo[
    "lat_grid"
].to_numpy(dtype=float)

esperado_grid = valor_esperado_por_bloques(
    modelo_final,
    X_grid
)

correccion_grid = kriging_por_bloques(
    ok_residuos,
    lon_prediccion,
    lat_prediccion,
    len(coords)
)

prediccion_continua = np.clip(
    esperado_grid + correccion_grid,
    0.0,
    N_CLASSES - 1
)

print(
    f"Superficie continua: mínimo "
    f"{prediccion_continua.min():.3f}, máximo "
    f"{prediccion_continua.max():.3f}, media "
    f"{prediccion_continua.mean():.3f}"
)

raster_continuo = np.full((N_LAT, N_LON), np.nan)

raster_continuo[
    gdf_grid_modelo["grid_row"].to_numpy(dtype=int),
    gdf_grid_modelo["grid_col"].to_numpy(dtype=int)
] = prediccion_continua

print(
    "Celdas con predicción:",
    int(np.isfinite(raster_continuo).sum())
)

print()


# ----------------------------------------------------------------------
# Simplificación para el dibujo
#
# Todas las uniones espaciales ya ocurrieron; lo que sigue es dibujo.
# ----------------------------------------------------------------------

gdf_municipios = simplificar(gdf_municipios)

gdf_modelo = simplificar(gdf_modelo)

gdf_mexico = simplificar(gdf_mexico)


# ----------------------------------------------------------------------
# Figura del Mapa A
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 9))

imagen = ax.imshow(
    raster_continuo,
    extent=[min_lon, max_lon, min_lat, max_lat],
    origin="lower",
    cmap=cmap_riesgo,
    vmin=0,
    vmax=4,
    interpolation="bilinear",
    aspect="equal",
    zorder=1
)

gdf_modelo.boundary.plot(
    ax=ax,
    linewidth=0.15,
    color="black",
    alpha=0.40,
    zorder=2
)

gdf_mexico.boundary.plot(
    ax=ax,
    linewidth=1.0,
    color="black",
    zorder=3
)

cbar = fig.colorbar(
    imagen,
    ax=ax,
    orientation="vertical",
    fraction=0.030,
    pad=0.025
)

cbar.set_label(
    "Predicted continuous ordinal score",
    fontsize=11
)

cbar.set_ticks([0, 1, 2, 3, 4])

cbar.set_ticklabels(["0", "1", "2", "3", "4"])

cbar.ax.tick_params(labelsize=9)

ax.set_title(
    "Continuous spatial distribution of the ordinal score", 
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

plt.tight_layout()

for extension in ("png", "pdf"):

    ruta = CARPETA_SALIDA / f"mapaA_superficie_continua.{extension}"

    fig.savefig(
        ruta,
        dpi=figuras_png.dpi_png(fig),
        bbox_inches="tight",
        facecolor="white"
    )

    if extension == "png":

        figuras_png.aplanar_png(ruta)

    print("Guardado:", ruta.resolve())

plt.close(fig)

print()


# ======================================================================
# 10. MAPA B: COROPLETAS CON PREDICCIÓN FUERA DE MUESTRA
# ======================================================================

print("=" * 90)
print("MAPA B: coropletas fuera de muestra")
print("=" * 90)

prediccion_oof = np.clip(
    np.floor(esperado_oof + correccion_oof + umbral),
    0,
    N_CLASSES - 1
).astype(int)

mae_oof = float(
    np.mean(np.abs(prediccion_oof - y_ordinal))
)

acierto_oof = float(
    np.mean(prediccion_oof == y_ordinal)
)

print(f"MAE fuera de muestra: {mae_oof:.4f}")

print(f"Acierto exacto: {acierto_oof:.4f}")

print()

print("Distribución observada frente a predicha:")

print(
    pd.DataFrame({
        "Observado": pd.Series(y_ordinal).value_counts().sort_index(),
        "Predicho": pd.Series(prediccion_oof).value_counts().sort_index()
    }).fillna(0).astype(int).to_string()
)

print()

gdf_modelo["nivel_predicho"] = prediccion_oof[filas_datos]

fig, ax = plt.subplots(figsize=(14, 9))

gdf_municipios.plot(
    ax=ax,
    facecolor="#f2f2f2",
    edgecolor="#c7c7c7",
    linewidth=0.12,
    zorder=1
)

gdf_modelo.plot(
    ax=ax,
    column="nivel_predicho",
    cmap=cmap_niveles,
    norm=norm_niveles,
    edgecolor="#8c8c8c",
    linewidth=0.12,
    zorder=2
)

gdf_mexico.boundary.plot(
    ax=ax,
    edgecolor="black",
    linewidth=0.70,
    zorder=3
)

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
    "Predicted dengue risk level",
    fontsize=11
)

cbar.set_ticks([0, 1, 2, 3, 4])

cbar.set_ticklabels(["0", "1", "2", "3", "4"])

cbar.ax.tick_params(labelsize=9)

ax.legend(
    handles=[
        Patch(
            facecolor=colores_nivel[nivel],
            edgecolor="#777777",
            linewidth=0.6,
            label=f"Level {nivel}"
        )
        for nivel in range(N_CLASSES)
    ],
    title="Ordinal risk levels",
    loc="lower left",
    frameon=True,
    framealpha=0.95,
    fontsize=9,
    title_fontsize=10,
    borderpad=0.8,
    labelspacing=0.6
)

ax.set_title(
    "Predicted dengue "
    "risk levels by municipality (out-of-sample)",
    fontsize=16,
    fontweight="bold",
    pad=14
)

ax.set_xlabel("Longitude", fontsize=11)

ax.set_ylabel("Latitude", fontsize=11)

margen = 0.5

ax.set_xlim(min_lon - margen, max_lon + margen)

ax.set_ylim(min_lat - margen, max_lat + margen)

ax.grid(linewidth=0.25, alpha=0.30)

ax.tick_params(axis="both", labelsize=9)

ax.set_xticks([-115, -110, -105, -100, -95, -90])

ax.set_yticks([15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5])

plt.tight_layout()

for extension in ("png", "pdf"):

    ruta = CARPETA_SALIDA / f"mapaB_coropletas_oof.{extension}"

    fig.savefig(
        ruta,
        dpi=figuras_png.dpi_png(fig),
        bbox_inches="tight",
        facecolor="white"
    )

    if extension == "png":

        figuras_png.aplanar_png(ruta)

    print("Guardado:", ruta.resolve())

plt.close(fig)


# ======================================================================
# 11. DATOS PARA LOS PIES DE FIGURA
# ======================================================================

print()
print("=" * 90)
print("DATOS PARA LOS PIES DE FIGURA")
print("=" * 90)

print(
    f"Modelo: Random Forest ordinal acumulativo con "
    f"{len(variables_modelo)} covariables seleccionadas en "
    f"al menos el {FRECUENCIA_MINIMA}% de las particiones "
    f"del análisis principal."
)

print(
    f"Variograma {VARIOGRAMA}; sill, rango y nugget: "
    f"{np.round(parametros, 4)}."
)

print(
    f"Umbral de conversión calibrado dentro de la muestra: "
    f"{umbral:.3f}."
)

print(
    f"Mapa A: malla de {N_LON}×{N_LAT} nodos; "
    f"{len(gdf_grid_modelo):,} nodos dentro de municipios "
    f"con información; las covariables conservan su "
    f"resolución municipal y no se interpolan; el valor "
    f"mostrado es el puntaje ordinal continuo corregido."
)

print(
    f"Mapa B: predicción fuera de muestra por municipio; "
    f"MAE {mae_oof:.3f}, acierto exacto "
    f"{100 * acierto_oof:.1f}%."
)

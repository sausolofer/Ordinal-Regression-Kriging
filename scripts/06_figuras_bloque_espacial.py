# ======================================================================
# 06 -- Maps for a held-out spatial block, residual surface, observed levels
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Run from the repository root:
#     python scripts/06_figuras_bloque_espacial.py
#
# Inputs:  data/ (+ data/shapefile/00mun.shp, optional)
# Outputs: figuras/figura09a_*, figura09b_*, figura10_*, figura11_observado.*
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
# FIGURAS:
#
#
#   Figura 9a  Entrenamiento + bloque de prueba
#              observado
#   Figura 9b  Entrenamiento + bloque de prueba (Es el que se reporta en el manuscrito)
#              predicho por ORK
#   Figura 10  Superficie krigeada de residuos ordinales
#   Figura 11  Categorias observadas en todos los municipios
#
# CONFIGURACION DEL MODELO
#
# Se fija Random Forest ordinal acumulativo y variograma
# exponencial, opciones seleccionadas por el procedimiento
# anidado en 90 a 96 por ciento de las particiones.
#
# La particion de las Figuras 9a y 9b es un bloque espacial,
# no una particion aleatoria, para ilustrar el escenario de
# validacion mas exigente.
#
# La Figura 10 usa residuos fuera de muestra de todos los
# municipios.
# ======================================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import rankdata

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import NearestNeighbors

from pykrige.ok import OrdinaryKriging

import rutas
import figuras_png


# ======================================================================
# 1. CONFIGURACIÓN
# ======================================================================

ARCHIVO_DATOS = rutas.ARCHIVO_DATOS

# Ruta al shapefile municipal. Ajustar si esta en otra
# carpeta. Si no se encuentra, los mapas se generan sin
# fondo.

CARPETA_SHAPE = Path(rutas.RUTA_SHAPEFILE).parent

RUTA_SHAPEFILE = Path(rutas.RUTA_SHAPEFILE)

CARPETA_FIGURAS = rutas.CARPETA_FIGURAS


columna_longitud = "lon_mun"
columna_latitud = "lat_mun"


colores_nivel = {
    0: "#2ca25f",   # Green
    1: "#fee08b",   # Yellow
    2: "#fdae61",   # Orange
    3: "#f46d43",   # Red
    4: "#8b0000"    # Dark red
}

nombres_nivel = {
    0: "Level 0",
    1: "Level 1",
    2: "Level 2",
    3: "Level 3",
    4: "Level 4"
}

TITULO_LEYENDA_NIVEL = "Dengue risk level"

TAMANO_TRAIN = 22
TAMANO_TEST = 58


# ------------------------------------------------
# Modelo, identico al analisis principal
# ------------------------------------------------

GLOBAL_SEED = 123

N_CLASSES = 5

SEMILLA_BASE_ESPACIAL = 7000

N_BLOQUES = 5

REPETICION_FIGURA = 0

BLOQUE_FIGURA = 0

UMBRAL_SPEARMAN = 0.10

MIN_VARIABLES = 3

PODAR_COLINEALES = True

UMBRAL_COLINEALIDAD = 0.95

N_FOLDS_OOF = 5

VARIOGRAMA = "exponential"

N_LAGS = 12

N_CLOSEST_POINTS = 50

REJILLA_UMBRAL = np.arange(0.05, 0.96, 0.01)


# ------------------------------------------------
# Superficie de residuos
# ------------------------------------------------

NODOS_MALLA = 300

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


DISTANCIA_MAXIMA_GRADOS = 0.40


os.makedirs(CARPETA_FIGURAS, exist_ok=True)


# ======================================================================
# 2. DATOS
# ======================================================================

df = pd.read_csv(rutas.ARCHIVO_DATOS)

COLUMNAS_EXCLUIDAS = [
    "Estado",
    "Municipio",
    "Rate_2022",
    columna_longitud,
    columna_latitud,
    "DRL"
]

df_covariables = df.drop(columns=COLUMNAS_EXCLUIDAS)

feature_names = list(df_covariables.columns)

X_full = df_covariables.to_numpy(dtype=float)

y_ordinal = (
    df["DRL"].to_numpy().ravel().astype(int)
)

coords = df[
    [columna_longitud, columna_latitud]
].to_numpy(dtype=float)

print("=" * 90)
print("DATOS")
print("=" * 90)
print("Municipios:", len(y_ordinal))
print("Covariables candidatas:", X_full.shape[1])
print()


# ======================================================================
# 3. FONDO GEOGRÁFICO
# ======================================================================

gdf_fondo = None

contorno_nacional = None

try:

    import geopandas as gpd

    if RUTA_SHAPEFILE.exists():

        gdf_fondo = gpd.read_file(RUTA_SHAPEFILE)

        print("=" * 90)
        print("SHAPEFILE CARGADO")
        print("=" * 90)

        print("Número de municipios:", len(gdf_fondo))

        print("CRS original:", gdf_fondo.crs)

        if gdf_fondo.crs is None:

            raise ValueError(
                "El shapefile no tiene CRS definido."
            )

        gdf_fondo = gdf_fondo.to_crs(epsg=4326)

        contorno_nacional = simplificar(contorno_exterior(gdf_fondo))

        gdf_fondo = simplificar(gdf_fondo)

        print("CRS utilizado:", gdf_fondo.crs)

        print()

    else:

        print(
            f"Shapefile no encontrado en {RUTA_SHAPEFILE}.\n"
            "Los mapas se generarán sin fondo geográfico.\n"
        )

except ImportError:

    print(
        "geopandas no está instalado. "
        "Los mapas se generarán sin fondo geográfico.\n"
    )


# ======================================================================
# 4. FUNCIONES DE GRAFICADO
# ======================================================================

def dibujar_fondo(ax):
    """
    Dibuja los poligonos municipales y el contorno nacional.
    """

    if gdf_fondo is None:

        return

    gdf_fondo.plot(
        ax=ax,
        facecolor="#f7f7f7",
        edgecolor="#c7c7c7",
        linewidth=0.15,
        zorder=1
    )

    contorno_nacional.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=0.75,
        zorder=2
    )


def configurar_ejes(
    ax,
    longitudes,
    latitudes,
    margen_longitud=1.0,
    margen_latitud=0.7
):
    """
    Ajusta extension, etiquetas y proporcion del mapa.
    """

    ax.set_xlim(
        longitudes.min() - margen_longitud,
        longitudes.max() + margen_longitud
    )

    ax.set_ylim(
        latitudes.min() - margen_latitud,
        latitudes.max() + margen_latitud
    )

    ax.set_xlabel("Longitude", fontsize=11)

    ax.set_ylabel("Latitude", fontsize=11)

    ax.tick_params(axis="both", labelsize=9)

    ax.grid(
        linestyle="--",
        linewidth=0.45,
        alpha=0.30
    )

    ax.set_aspect(
        1 / np.cos(np.deg2rad(latitudes.mean()))
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)


def crear_leyenda_niveles():

    return [
        Patch(
            facecolor=colores_nivel[nivel],
            edgecolor="black",
            linewidth=0.5,
            label=nombres_nivel[nivel]
        )
        for nivel in range(N_CLASSES)
    ]


def leyenda_tipo_punto(etiqueta_test):

    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="0.55",
            markeredgecolor="none",
            markersize=5,
            alpha=0.50,
            label="Observed training municipality"
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=7,
            label=etiqueta_test
        )
    ]


def guardar_figura(fig, nombre_base):

    ruta = os.path.join(CARPETA_FIGURAS, nombre_base)

    fig.savefig(
        f"{ruta}.png",
        dpi=figuras_png.dpi_png(fig),
        bbox_inches="tight",
        facecolor="white"
    )

    figuras_png.aplanar_png(f"{ruta}.png")

    fig.savefig(
        f"{ruta}.pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"Guardados: {ruta}.png y {ruta}.pdf")

    plt.close(fig)


def dibujar_puntos_por_nivel(
    ax,
    longitudes,
    latitudes,
    niveles,
    tamano,
    con_borde,
    alpha,
    zorder
):

    for nivel in range(N_CLASSES):

        mascara = niveles == nivel

        if mascara.sum() == 0:

            continue

        ax.scatter(
            longitudes[mascara],
            latitudes[mascara],
            s=tamano,
            c=colores_nivel[nivel],
            marker="o",
            edgecolors="black" if con_borde else "none",
            linewidths=0.90 if con_borde else 0.0,
            alpha=alpha,
            zorder=zorder
        )


# ======================================================================
# 5. SELECCIÓN DE VARIABLES
# ======================================================================

def _spearman_columnas(X, y):

    rangos_X = np.apply_along_axis(rankdata, 0, X)

    rangos_y = rankdata(y)

    Xc = rangos_X - rangos_X.mean(axis=0)

    yc = rangos_y - rangos_y.mean()

    sx = np.sqrt(np.sum(Xc ** 2, axis=0))

    sy = np.sqrt(np.sum(yc ** 2))

    denominador = sx * sy

    rho = np.zeros(X.shape[1])

    valido = denominador > 0

    rho[valido] = (
        (Xc[:, valido] * yc[:, None]).sum(axis=0)
        /
        denominador[valido]
    )

    return rho


def seleccionar_variables(X_train, y_train):

    rho = _spearman_columnas(X_train, y_train)

    magnitud = np.abs(rho)

    seleccion = np.where(magnitud >= UMBRAL_SPEARMAN)[0]

    if seleccion.size < MIN_VARIABLES:

        seleccion = np.argsort(-magnitud)[:MIN_VARIABLES]

    seleccion = seleccion[np.argsort(-magnitud[seleccion])]

    if not PODAR_COLINEALES or seleccion.size <= 1:

        return np.sort(seleccion)

    sub = X_train[:, seleccion]

    rangos_sub = np.apply_along_axis(rankdata, 0, sub)

    with np.errstate(invalid="ignore"):

        matriz = np.corrcoef(rangos_sub, rowvar=False)

    matriz = np.nan_to_num(matriz, nan=0.0)

    conservadas = []

    for posicion in range(seleccion.size):

        redundante = any(
            abs(matriz[posicion, previa]) >= UMBRAL_COLINEALIDAD
            for previa in conservadas
        )

        if not redundante:

            conservadas.append(posicion)

    finales = seleccion[conservadas]

    if finales.size == 0:

        finales = seleccion[:1]

    return np.sort(finales)


# ======================================================================
# 6. MODELO ORDINAL ACUMULATIVO
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

                columnas.append(np.full(n, objeto, dtype=float))

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

    def predict(self, X):

        return np.sum(
            self._umbral_probs(X) >= 0.5, axis=1
        ).astype(int)


# ======================================================================
# 7. KRIGING Y CALIBRACIÓN
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


def predecir_kriging(modelo_ok, coords_pred, n_train):

    n_cercanos = int(
        min(N_CLOSEST_POINTS, max(1, n_train - 1))
    )

    z, _ = modelo_ok.execute(
        style="points",
        xpoints=coords_pred[:, 0],
        ypoints=coords_pred[:, 1],
        backend="loop",
        n_closest_points=n_cercanos
    )

    return np.asarray(z).ravel().astype(float)


def residuos_oof(X_sel, y, semilla):

    conteo = np.bincount(y, minlength=N_CLASSES)

    minimo = conteo[conteo > 0].min()

    n_folds = int(max(2, min(N_FOLDS_OOF, minimo)))

    if minimo >= n_folds:

        divisor = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=semilla
        )

        generador = divisor.split(X_sel, y)

    else:

        divisor = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=semilla
        )

        generador = divisor.split(X_sel)

    esperado = np.full(len(y), np.nan)

    for numero, (idx_a, idx_v) in enumerate(generador, start=1):

        if np.unique(y[idx_a]).size < 2:

            continue

        modelo = CumulativeOrdinalRF(
            random_state=semilla + numero
        )

        modelo.fit(X_sel[idx_a], y[idx_a])

        esperado[idx_v] = modelo.predict_expected_value(
            X_sel[idx_v]
        )

    validos = np.isfinite(esperado)

    return esperado, y.astype(float) - esperado, validos


def calibrar_umbral(valores, y_true):

    valores = np.asarray(valores, dtype=float)

    y_true = np.asarray(y_true).astype(int)

    validos = np.isfinite(valores)

    if validos.sum() < 20:

        return 0.5

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


def kriging_oof(coords_train, residuos, semilla, n_folds=3):

    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=semilla
    )

    prediccion = np.full(len(residuos), np.nan)

    for idx_a, idx_v in kf.split(coords_train):

        try:

            ok = ajustar_kriging(
                coords_train[idx_a], residuos[idx_a]
            )

            prediccion[idx_v] = predecir_kriging(
                ok, coords_train[idx_v], len(idx_a)
            )

        except Exception:

            continue

    return prediccion


# ======================================================================
# 8. AJUSTE PARA LAS FIGURAS 9a Y 9b
# ======================================================================

print("=" * 90)
print("FIGURAS 9a y 9b: bloque espacial retenido")
print("=" * 90)

kmeans = KMeans(
    n_clusters=N_BLOQUES,
    n_init=10,
    random_state=SEMILLA_BASE_ESPACIAL + REPETICION_FIGURA
)

etiquetas_bloque = kmeans.fit_predict(coords)

mascara_test = etiquetas_bloque == BLOQUE_FIGURA

idx_test = np.where(mascara_test)[0]

idx_train = np.where(~mascara_test)[0]

print(
    f"Bloque {BLOQUE_FIGURA}: {len(idx_test)} municipios "
    f"retenidos, {len(idx_train)} de entrenamiento"
)

X_train = X_full[idx_train]
y_train = y_ordinal[idx_train]
coords_train = coords[idx_train]

X_test = X_full[idx_test]
y_test = y_ordinal[idx_test]
coords_test = coords[idx_test]

columnas = seleccionar_variables(X_train, y_train)
print("Variables:", [feature_names[j] for j in columnas])

print("Variables seleccionadas:", len(columnas))

X_train_sel = X_train[:, columnas]
X_test_sel = X_test[:, columnas]

semilla = GLOBAL_SEED + REPETICION_FIGURA

esperado_oof, residuo_oof, validos = residuos_oof(
    X_train_sel, y_train, semilla
)

coords_oof = coords_train[validos]
residuo_oof_v = residuo_oof[validos]
esperado_oof_v = esperado_oof[validos]
y_oof = y_train[validos]

ok_train = ajustar_kriging(coords_oof, residuo_oof_v)

correccion_test = predecir_kriging(
    ok_train, coords_test, len(coords_oof)
)

correccion_oof = kriging_oof(
    coords_oof, residuo_oof_v, semilla
)

validos_cal = np.isfinite(correccion_oof)

umbral = calibrar_umbral(
    esperado_oof_v[validos_cal] + correccion_oof[validos_cal],
    y_oof[validos_cal]
)

print(f"Umbral calibrado: {umbral:.3f}")

modelo_final = CumulativeOrdinalRF(random_state=semilla)

modelo_final.fit(X_train_sel, y_train)

esperado_test = modelo_final.predict_expected_value(X_test_sel)

pred_test = np.clip(
    np.floor(esperado_test + correccion_test + umbral),
    0,
    N_CLASSES - 1
).astype(int)

mae_bloque = float(np.mean(np.abs(pred_test - y_test)))

print(f"MAE en el bloque: {mae_bloque:.4f}")
print()


# ----------------------------------------------------------------------
# Figuras 9a y 9b
# ----------------------------------------------------------------------

for sufijo, niveles_test, titulo, etiqueta_leyenda in [
    (
        "a",
        y_test,
        "Observed training levels and observed "
        "levels in the held-out spatial block",
        "Observed test municipality"
    ),
    (
        "b",
        pred_test,
        "Observed training levels and ORK predictions "
        "for the held-out spatial block",
        "ORK prediction for test municipality"
    )
]:

    fig, ax = plt.subplots(figsize=(10, 7.5))

    dibujar_fondo(ax)

    dibujar_puntos_por_nivel(
        ax,
        coords_train[:, 0],
        coords_train[:, 1],
        y_train,
        TAMANO_TRAIN,
        con_borde=False,
        alpha=0.48,
        zorder=3
    )

    dibujar_puntos_por_nivel(
        ax,
        coords_test[:, 0],
        coords_test[:, 1],
        niveles_test,
        TAMANO_TEST,
        con_borde=True,
        alpha=0.96,
        zorder=4
    )

    configurar_ejes(ax, coords[:, 0], coords[:, 1])

    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=12)

    leyenda_1 = ax.legend(
        handles=crear_leyenda_niveles(),
        title=TITULO_LEYENDA_NIVEL,
        loc="lower left",
        frameon=True,
        fontsize=9,
        title_fontsize=10
    )

    ax.add_artist(leyenda_1)

    ax.legend(
        handles=leyenda_tipo_punto(etiqueta_leyenda),
        title="Municipality type",
        loc="upper right",
        frameon=True,
        fontsize=9,
        title_fontsize=10
    )

    plt.tight_layout()

    guardar_figura(
        fig,
        f"figura09{sufijo}_bloque_espacial"
    )

print()


# ======================================================================
# 9. FIGURA 10: SUPERFICIE KRIGEADA DE RESIDUOS
# ======================================================================

print("=" * 90)
print("FIGURA 10: superficie krigeada de residuos")
print("=" * 90)

columnas_todas = seleccionar_variables(X_full, y_ordinal)

print("Variables seleccionadas:", len(columnas_todas))

X_sel_todas = X_full[:, columnas_todas]

_, residuo_todos, validos_todos = residuos_oof(
    X_sel_todas, y_ordinal, GLOBAL_SEED
)

coords_todos = coords[validos_todos]

residuo_todos = residuo_todos[validos_todos]

print(f"Residuos disponibles: {len(residuo_todos)}")

print(
    f"Rango de residuos: {residuo_todos.min():.2f} "
    f"a {residuo_todos.max():.2f}"
)

ok_todos = ajustar_kriging(coords_todos, residuo_todos)

margen = 0.3

malla_lon = np.linspace(
    coords[:, 0].min() - margen,
    coords[:, 0].max() + margen,
    NODOS_MALLA
)

malla_lat = np.linspace(
    coords[:, 1].min() - margen,
    coords[:, 1].max() + margen,
    NODOS_MALLA
)

rejilla_lon, rejilla_lat = np.meshgrid(malla_lon, malla_lat)

puntos_malla = np.column_stack([
    rejilla_lon.ravel(),
    rejilla_lat.ravel()
])

buscador = NearestNeighbors(n_neighbors=1).fit(coords_todos)

distancias, _ = buscador.kneighbors(puntos_malla)

cerca = distancias.ravel() <= DISTANCIA_MAXIMA_GRADOS

print(
    f"Nodos de malla: {len(puntos_malla)}; "
    f"se evaluarán {cerca.sum()} "
    f"({100 * cerca.mean():.1f} por ciento)"
)

print("Evaluando el kriging sobre la malla...")

valores_malla = np.full(len(puntos_malla), np.nan)

valores_malla[cerca] = predecir_kriging(
    ok_todos, puntos_malla[cerca], len(coords_todos)
)

superficie = np.ma.array(
    valores_malla.reshape(rejilla_lon.shape),
    mask=(~cerca).reshape(rejilla_lon.shape)
)

limite = float(
    np.nanpercentile(np.abs(residuo_todos), 95)
)

fig, ax = plt.subplots(figsize=(10, 7.5))

dibujar_fondo(ax)

imagen = ax.pcolormesh(
    rejilla_lon,
    rejilla_lat,
    superficie,
    cmap="RdBu_r",
    vmin=-limite,
    vmax=limite,
    shading="auto",
    zorder=3
)

if contorno_nacional is not None:

    contorno_nacional.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=0.75,
        zorder=4
    )

ax.scatter(
    coords_todos[:, 0],
    coords_todos[:, 1],
    c="none",
    edgecolors="0.25",
    linewidths=0.25,
    s=5,
    zorder=5
)

barra = fig.colorbar(imagen, ax=ax, shrink=0.80)

barra.set_label(
    "Kriged ordinal residual (categories)",
    fontsize=10
)

configurar_ejes(ax, coords[:, 0], coords[:, 1])

ax.set_title(
    "Kriged surface of out-of-fold ordinal residuals\n"
    "Positive values indicate observed categories above "
    "the model expectation",
    fontsize=13,
    pad=12
)

plt.tight_layout()

guardar_figura(fig, "figura10_superficie_residuos")

print()


# ======================================================================
# 10. FIGURA 11: CATEGORÍAS OBSERVADAS
# ======================================================================

print("=" * 90)
print("FIGURA 11: categorías observadas")
print("=" * 90)

fig, ax = plt.subplots(figsize=(10, 7.5))

dibujar_fondo(ax)

dibujar_puntos_por_nivel(
    ax,
    coords[:, 0],
    coords[:, 1],
    y_ordinal,
    34,
    con_borde=True,
    alpha=0.92,
    zorder=3
)

configurar_ejes(ax, coords[:, 0], coords[:, 1])

ax.set_title(
    "Observed reported dengue incidence levels "
    "by municipality, 2022",
    fontsize=16,
    fontweight="bold",
    pad=14
)

ax.legend(
    handles=crear_leyenda_niveles(),
    title=TITULO_LEYENDA_NIVEL,
    loc="lower left",
    frameon=True,
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout()

guardar_figura(fig, "figura11_observado")


# ======================================================================
# 11. DATOS PARA LOS PIES DE FIGURA
# ======================================================================

print()
print("=" * 90)
print("DATOS PARA LOS PIES DE FIGURA")
print("=" * 90)

print(
    f"Figuras 9a y 9b: bloque {BLOQUE_FIGURA} de la "
    f"repetición {REPETICION_FIGURA}; {len(idx_test)} "
    f"municipios retenidos; {len(columnas)} variables "
    f"seleccionadas dentro del entrenamiento; umbral "
    f"calibrado {umbral:.3f}; MAE del bloque "
    f"{mae_bloque:.3f}."
)

print(
    f"Figura 10: residuos fuera de muestra de "
    f"{len(residuo_todos)} municipios; variograma "
    f"{VARIOGRAMA}; escala de color recortada al percentil "
    f"95 (±{limite:.2f}); nodos a más de "
    f"{DISTANCIA_MAXIMA_GRADOS} grados del municipio más "
    f"cercano enmascarados."
)

print(
    "Configuración fijada a Random Forest ordinal "
    "acumulativo y variograma exponencial, opciones "
    "seleccionadas por el procedimiento anidado en la "
    "mayoría de las particiones."
)

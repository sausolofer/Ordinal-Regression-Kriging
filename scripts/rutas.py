# ======================================================================
# RUTAS DEL REPOSITORIO
#
# Único módulo añadido al reorganizar el análisis como repositorio.
# No contiene lógica del método: sólo resuelve dónde están los datos y
# dónde se escriben los resultados, de modo que los scripts funcionen
# desde cualquier directorio de trabajo y no sólo desde la raíz.
#
# Los scripts importan de aquí en lugar de declarar rutas relativas.
# ======================================================================

import os
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]

ARCHIVO_DATOS = str(
    RAIZ / "data" / "municipal_dengue_dataset_mexico_2022.csv"
)

CARPETA_RESULTADOS = str(RAIZ / "resultados")

CARPETA_FIGURAS = str(RAIZ / "figuras")

# Shapefile municipal del Marco Geoestadístico del INEGI, necesario
# únicamente para los mapas (scripts 06 a 08). No se distribuye con el
# repositorio: véase data/shapefile/README.md.

RUTA_SHAPEFILE = str(RAIZ / "data" / "shapefile" / "00mun.shp")


os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

os.makedirs(CARPETA_FIGURAS, exist_ok=True)


if not os.path.exists(ARCHIVO_DATOS):

    raise SystemExit(
        f"No se encontró el archivo de datos:\n  {ARCHIVO_DATOS}\n\n"
        "Descárguelo del depósito de Zenodo citado en el README y "
        "colóquelo en la carpeta data/."
    )

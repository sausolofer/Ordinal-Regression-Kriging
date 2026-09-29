# ======================================================================
# figuras_png -- resolución y formato de los PNG que entran al manuscrito
#
# Part of: Ordinal Regression Kriging (ORK) -- reproduction code for
# "Continuous Spatial Prediction of Dengue Risk Using Integrated
# Municipal-Level Data: An Ordinal Regression Kriging Approach"
# (Acta Tropica, ACTROP-D-26-01823).
#
# Por qué existe este módulo
#
# Los scripts guardaban los PNG con dpi=600. Una figura de 14 pulgadas
# de ancho sale entonces de 8,400 px, y el manuscrito la escala al ancho
# de la página, unas 5.8 pulgadas: 1,430 ppi sobre el papel. Elsevier
# pide 500 ppi para arte combinado, así que eran casi tres veces más
# píxeles por lado de los necesarios: 43 megapíxeles por mapa.
#
# El archivo PDF no crece tanto, porque Flate comprime bien las zonas
# planas, pero eso engaña: lo que cuenta para la fluidez no es el tamaño
# del archivo sino cuántos píxeles tiene que descomprimir el lector cada
# vez que la figura entra en pantalla. Con tres mapas así el manuscrito
# obliga a reconstruir medio gigabyte de mapa de bits, y por eso se
# traba justo al pasar por ellos.
#
# Hay un segundo costo. matplotlib guarda los PNG en RGBA, y pdfTeX
# incrusta el canal alfa como una imagen separada (SMask) que el lector
# descomprime y compone encima de la otra: un tercio más de trabajo por
# una transparencia que aquí siempre es opaca, porque las figuras se
# guardan con fondo blanco.
#
# Este módulo resuelve las dos cosas: dpi_png() calcula la resolución a
# partir del ancho al que la figura se imprime, y aplanar_png() quita el
# canal alfa después de guardar.
#
# Nada de esto toca los PDF vectoriales ni ningún resultado numérico.
# ======================================================================

# Resolución objetivo sobre la página impresa. Elsevier pide 500 ppi
# para arte combinado (imagen con líneas y texto encima), que es lo que
# son estos mapas; 600 deja margen.

PPI_EN_PAGINA = 600

# Ancho al que el manuscrito imprime una figura a todo el ancho de la
# columna, en pulgadas. Medido sobre el PDF compilado: 8,340 px a
# 1,430 ppi.

ANCHO_EN_PAGINA = 5.8


def dpi_png(fig):
    """dpi que deja la figura en PPI_EN_PAGINA sobre la página impresa.

    Depende del ancho de la figura en pulgadas: una de 14 pulgadas se
    reduce mucho más al colocarla en la página que una de 8, así que
    necesita menos dpi para llegar a la misma resolución impresa.
    """

    return PPI_EN_PAGINA * ANCHO_EN_PAGINA / fig.get_size_inches()[0]


def aplanar_png(ruta):
    """Quita el canal alfa del PNG, sobre fondo blanco.

    Pillow viene con matplotlib, así que no agrega ninguna dependencia.
    """

    from PIL import Image

    imagen = Image.open(ruta)

    if imagen.mode not in ("RGBA", "LA", "P"):

        return

    rgba = imagen.convert("RGBA")

    fondo = Image.new("RGB", rgba.size, "white")

    fondo.paste(rgba, mask=rgba.split()[-1])

    fondo.save(ruta, optimize=True)

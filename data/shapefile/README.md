# Municipal boundary geometries

The map scripts (`06`, `07` and `08`) draw municipal polygons and the national
outline. Those geometries are **not** distributed with this repository, because
they belong to a separate public product with its own terms of use and weigh far
more than the analysis dataset.

Place here the municipal layer of INEGI's *Marco Geoestadístico*, so that this
folder contains:

```
data/shapefile/00mun.shp     (~55 MB)
data/shapefile/00mun.dbf
data/shapefile/00mun.shx
data/shapefile/00mun.prj
data/shapefile/00mun.cpg
```

All five components are needed: geopandas fails without the `.dbf` or the
`.shx`, reprojects using the `.prj`, and reads attribute text with the encoding
declared in the `.cpg`.

## Edition used for the published figures

**Marco Geoestadístico 2020 Integrado** (`MG_2020_Integrado`), INEGI, files
dated February 2021, distributed through the *Marco Geoestadístico. Censo de
Población y Vivienda 2020* package. Inside that download the layer sits at
`MG_2020_Integrado/conjunto_de_datos/00mun.*`.

The package is large — over 2 GB, because it contains every layer for the whole
country — but only these five files are needed. They can be extracted without
unpacking the rest:

```bash
unzip -j <package>.zip "*00mun.*" -d data/shapefile/
```

Record the edition if you substitute another one. Municipal boundaries change
between editions, so the maps are not strictly reproducible without it, even
with the random seeds fixed.

## How the join works

The scripts join the analysis table to these polygons **by location**: each
municipal seat (`lon_mun`, `lat_mun`) is assigned to the polygon that contains
it. No key or name matching is involved, so the layer does not need to share
identifiers with the dataset.

With the edition above, the join was verified as follows:

- the layer contains **2,469** municipal polygons, in `MEXICO_ITRF_2008_LCC`
  (EPSG:6372), which the scripts reproject to EPSG:4326 before drawing;
- **all 1,050** municipal seats of the dataset fall inside a polygon, so no
  municipality is lost for lack of a containing geometry;
- **one collision**: Calkiní and Dzitbalché, in Campeche, both fall inside
  `CVEGEO 04001` (Calkiní). Dzitbalché is a recently created municipality that
  this edition of the Marco Geoestadístico does not yet separate, while the
  dataset treats it as an independent unit.

The consequence of that collision is small but worth knowing: the choropleths
draw **1,049** polygons rather than 1,050. Both municipalities are at risk level
0 with zero reported incidence, so the colour of the shared polygon is the same
either way, but a figure caption claiming that all 1,050 municipalities are
shown would be off by one. Script `08` prints the actual number at the end of
its run; script `08` resolves the collision by keeping the higher category and
script `07` by keeping the first match.

## Behaviour without the shapefile

Script `06` degrades gracefully and produces its figures without a geographic
background. Scripts `07` and `08` stop with an explicit error, because the
choropleths have nothing to draw without polygons.

## Figure size and geometry simplification

At full resolution these polygons carry about 3.56 million vertices, which in a
vector PDF amounts to roughly 30 MB per map and makes a manuscript slow to
render. Scripts `06`, `07` and `08` therefore simplify the geometry **for
drawing only**, through the constant `TOLERANCIA_SIMPLIFICACION`, applied after
every spatial join so that no analysis result changes.

The default tolerance is 0.002 degrees, about 220 m. Measured on this edition:

| Tolerance | Vertices | Reduction | PDF |
|---|---|---|---|
| none | 3,563,912 | — | 29.7 MB |
| 0.002 | 214,300 | 94.0% | 2.5 MB |
| 0.005 | 111,403 | 96.9% | 1.4 MB |
| 0.01 | 65,191 | 98.2% | 0.9 MB |

At the width a figure occupies on a journal page, 0.002 degrees is about
0.01 mm, far below what print or a PDF viewer can resolve; the difference only
becomes visible when zooming to a few degrees of longitude across the screen.
Set the constant to `None` to draw the complete geometry.

## The national outline: slivers left by the dissolve

The maps draw a thicker black line for the outline of the country, obtained by
dissolving the municipal layer. The polygons of this edition do not share their
vertices exactly along common boundaries, so the union leaves **13,473 interior
holes**: slivers a few square metres wide between neighbouring municipalities.
As area they are invisible — the median is 1.7e-13 square degrees, about two
square metres — but each is a closed curve, and `boundary.plot` strokes it with
the same heavy black line as the coastline. Where municipalities are small and
numerous (central Mexico, Tlaxcala, Oaxaca) they pile up into what look like
black smudges scattered over the map.

Scripts `06`, `07` and `08` therefore build the outline with
`contorno_exterior()`, which dissolves the layer and then keeps only the holes
of area at least `AREA_MINIMA_HUECO`. At 1e-6 square degrees, about 1.2 ha, the
two genuine holes of the country survive — the estuary between Empalme and
Guaymas, Sonora, of 7.5 and 0.2 km² — and all 13,471 slivers disappear. The
largest sliver measures 7.7e-9 and the smaller genuine hole 1.8e-5, three orders
of magnitude apart, so the threshold is not a delicate choice. As a side effect
the outline drops from 66,401 vertices to 10,019.

The order matters. The outline is built from the **complete** geometry, before
simplification: simplifying each municipality separately and dissolving
afterwards opens gaps of up to twice the tolerance, which are no longer
negligible — measured on this edition, 10,157 of them would exceed the
threshold.

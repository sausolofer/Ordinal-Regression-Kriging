# Reproduction notes

## Order of execution

Script 09 needs only the dataset and runs in seconds. Script 01 produces everything that scripts 02, 03, 05 and 07 read. Those four locate the most recent output of 01 by its timestamp, so running them after a second execution of 01 reads the second set. If you want a specific run, delete or move the newer files. Scripts 04, 06 and 08 read only the dataset and can run at any time.

```bash
make test     # a few minutes: confirms every stage executes
make all      # the real run, about two hours in total
make maps     # scripts 06 to 08 only
```

Scripts 07 and 08 need the municipal geometries described in `data/shapefile/README.md`. Script 06 runs without them and omits the geographic background.

## From article element to code

### Tables

| Table | Script | Output |
|---|---|---|
| 1 | — | written by hand |
| 2 | 09 | `tabla02_asociaciones.csv`, `tabla02_asociaciones.tex` |
| 3, 4, 7 | 01 | printed to stdout; recomputable from `resultados_por_particion_*.csv` |
| 5 | 01 | `wilcoxon_*.csv`, `frecuencia_mejora_*.csv` |
| 6 | 01 | `modelos_seleccionados_*.csv`, `variogramas_seleccionados_*.csv` |
| 8 | 01 | `desempeno_por_clase_*.csv` |
| 9, 10 | 02 | `confusion_norm_*.csv`, `confusion_*.tex` |
| 11 | 03 | `efecto_agrupacion_*.csv` |
| 12, 13 | 01 | printed; recomputable from the `moran_*` columns of `resultados_por_particion_*.csv` |
| 14 | 05 | `figuras/shap_importancia_rf.csv` |
| 15 | 01 | `seleccion_variables_*.csv` |

### Figures

The file names produced by scripts 06 to 08 come from an earlier draft in which the figures were numbered differently. The correspondence with the submitted manuscript is:

| Figure | Script | Output file |
|---|---|---|
| 1 — distribution of positive incidence rates | 09 | `figura01_incidencia_positiva.{png,pdf}` |
| 2 — distribution of risk levels | 09 | `figura02_niveles_riesgo.{png,pdf}` |
| 3 — Spearman correlation matrix | 09 | `figura03_matriz_spearman.{png,pdf}` |
| 4 — training levels and ORK predictions for the held-out block | 06 | `figura09b_bloque_espacial.{png,pdf}` |
| 5 — out-of-sample predictions for every municipality | 07 | `mapaB_coropletas_oof.{png,pdf}` |
| 6 — observed municipal distribution | 08 | `figura11_observado_coropletas.{png,pdf}` |
| 7 — continuous ordinal score surface | 07 | `mapaA_superficie_continua.{png,pdf}` |
| 8 to 11 — SHAP summaries | 05 | `SHAP_Summary_DRL_greater_{0,1,2,3}.{png,pdf}` |

Four more figures are produced but do not appear in the manuscript: `figura09a_bloque_espacial` (the same block with its observed rather than predicted levels, the natural companion of Figure 4), `figura10_superficie_residuos` (the kriged surface of out-of-fold residuals), `figura11_observado` (the observed levels as points rather than polygons, superseded by Figure 6), and the two diagnostics of script 01, `mae_por_esquema` and `efecto_espacial`.

## Reproducibility

The master seed is 123. The outer partitions derive from two fixed seed bases, 5000 for the random scheme and 7000 for the spatial one, so partition *i* is identical between runs and between scripts: `04_puntajes_codificacion.py` reconstructs exactly the same partitions as `01_validacion_anidada.py` without refitting anything, and `06_figuras_bloque_espacial.py` reconstructs block 0 of repetition 0 of the spatial scheme.

The seed bases differ from those of the earlier version of the study, so the test sets of the two analyses do not coincide. This is deliberate; the results reported in the revision are not a re-scoring of the previous partitions.

Outputs are timestamped rather than overwritten, so a rerun never destroys an earlier result. Clean them with `make clean` when they accumulate.

The figures, unlike the tables, depend on an external product: municipal boundaries change between editions of the Marco Geoestadístico, so record which edition was used. Without that, the maps are not strictly reproducible even with the seeds fixed.

The map scripts simplify the polygons before drawing, through `TOLERANCIA_SIMPLIFICACION`, which keeps the vector figures to a few megabytes instead of thirty. The simplification is applied after every spatial join, so it affects only what is drawn and never a reported number; `data/shapefile/README.md` documents the measured trade-off, and the slivers the dissolve leaves in the national outline.

## Figure resolution

Every figure is written twice, as PDF and as PNG. The PDF is vector and its size is governed by the simplification above. The PNG is raster, and its resolution comes from `scripts/figuras_png.py`, which sets the dpi so that the figure lands at `PPI_EN_PAGINA` (600) once the manuscript scales it to `ANCHO_EN_PAGINA` (5.8 inches). A fixed dpi does not work here, because the figures differ in width: 600 dpi on a 14-inch map gives 1,430 ppi on the page, nearly three times what Elsevier asks for combination artwork, and 43 megapixels that a PDF reader has to decompress every time the figure scrolls into view. That, rather than the file size, is what makes a manuscript stutter on the map pages.

The same module strips the alpha channel after saving. matplotlib writes PNGs as RGBA; pdfTeX embeds the alpha as a separate soft-mask image that the reader decompresses and composites on top, a third more work for a transparency that is always opaque, since the figures are saved on a white background.

## Two caveats the code states about itself

The Wilcoxon tests compare predictors across outer partitions that share observations, so the partitions are not independent and the p-values are descriptive summaries of how consistently one predictor beats another, not strict inference. Section 20 of script 01 says so in place.

Script 03 regroups the predictions of a model trained on five levels; it does not retrain on fewer categories. It indicates whether the ordering the model produces survives at a coarser resolution, which is suggestive but not conclusive. If a grouping looks clearly better, the analysis has to be rerun with the new coding before anything is claimed. The MAE is not comparable across groupings, which is why the improvement over each grouping's own trivial predictor is reported alongside it.

A third caveat belongs to the maps. Map A of script 07 evaluates the final model, fitted on all 1,050 municipalities, over a grid; at an observed municipal seat the kriging predictor is exact, so the corrected score there reproduces the observed category by construction. That map describes the fit, not out-of-sample behaviour. Map B is the one that shows prediction: every municipality receives the expectation of a model that did not see it, plus a correction interpolated from the others.

## Before archiving

- [ ] `make all` completes from a clean checkout
- [ ] `make lock` run, `environment.lock.yml` and `requirements.lock.txt` committed
- [ ] Runtime, hardware and the Marco Geoestadístico edition recorded in the README
- [ ] Figures and tables in `resultados/` and `figuras/` match those in the submitted manuscript
- [ ] Zenodo release created from a tagged release and the DOI added to `README.md` and `CITATION.cff`
- [ ] The dataset DOI in `CITATION.cff` matches the one cited in the manuscript
- [ ] The "Code availability" statement in the manuscript points to the archived DOI, not to the branch

# Ordinal Regression Kriging (ORK)

Code and data reproducing the analysis reported in *Continuous Spatial Prediction of Dengue Risk Using Integrated Municipal-Level Data: An Ordinal Regression Kriging Approach* (Acta Tropica, ACTROP-D-26-01823).

ORK extends the regression kriging paradigm to ordered categorical responses. An ordinal regression model maps covariates to a probability distribution over ordered risk levels; that distribution is summarised by its conditional expectation, and ordinary kriging interpolates the continuous out-of-fold residuals of the model. The two components are added on the risk-level scale and the corrected score is decoded back to a category. The framework is applied here to municipal dengue risk in Mexico for 2022, but the decomposition is not specific to dengue and applies to any georeferenced outcome expressed as ordered categories.

The code and its comments are written in Spanish; this README, the reproduction notes and the output structure are in English.

---

## Citation

> Wences, G., Solorio-Fernández, S., Herrera-Zúñiga, L. D., and Hernández-López, E. Continuous Spatial Prediction of Dengue Risk Using Integrated Municipal-Level Data: An Ordinal Regression Kriging Approach. *Acta Tropica* (under review).

`CITATION.cff` carries the same information in machine-readable form, and GitHub surfaces it through the "Cite this repository" button.

---

## Data

`data/municipal_dengue_dataset_mexico_2022.csv` contains 1,050 Mexican municipalities and 36 columns: two identifiers (`Estado`, `Municipio`), the 2022 dengue incidence rate (`Rate_2022`), the municipal centroid (`lon_mun`, `lat_mun`), the ordinal response (`DRL`) and 30 candidate covariates, including the two composite vulnerability indices `SEVI` and `SEAVI`. There are no missing values.

The response has five ordered levels. Level 0 collects the 580 municipalities with no reported cases; the 470 with positive incidence are split at the empirical quartiles of the positive distribution into levels 1 to 4, which contain 118, 117, 117 and 118 municipalities respectively. Level 0 means *no reported cases*, not absence of transmission, and the quartile cut-points are relative to this sample and this year rather than official epidemiological thresholds.

`data/data_dictionary.csv` documents the role, units, source, reference year and observed range of every column. Six agricultural variables (`AWCROP`, `OFANN`, `PRANN`, `OFPER`, `PRPER`, `STRCROP`) are planted areas in hectares, not counts of crops, despite how they are described in Table 1 of the article.

The map scripts additionally need municipal boundary geometries, which are **not** distributed here: place INEGI's *Marco Geoestadístico* municipal layer in `data/shapefile/` as `00mun.shp`. See `data/shapefile/README.md`. Script 06 runs without it and simply omits the background; scripts 07 and 08 stop with an explicit error, since a choropleth has nothing to draw without polygons.

The dataset is included here so the repository runs out of the box. The citable version of record is the Zenodo deposit:

> https://doi.org/10.5281/zenodo.23004274 (version 3)

Earlier versions of that deposit (10.5281/zenodo.21987858 and 10.5281/zenodo.22777558) contain only the fifteen covariates retained by a full-sample feature selection and cannot reproduce this analysis, which recomputes the selection within each training partition from all thirty candidates.

---

## Installation

```bash
conda env create -f environment.yml
conda activate ork
```

or, with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**One caveat worth knowing before you start.** `mord`, which provides the two threshold-based ordinal models, is an old package whose source distribution does not build under recent setuptools; the failure is an `AttributeError: install_layout` during the wheel build. It installs cleanly with the legacy distutils backend:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib pip install mord
```

Everything else installs normally. The pinned versions are those reported in Section 3.6 of the article — Python 3.11.5, scikit-learn 1.9.0, xgboost 3.2.0, statsmodels 0.14.6, mord 0.7, PyKrige 1.7.3, libpysal 4.14.1 and esda 2.9.0 — so the environment described in the paper and the one built from these files are the same. Run `make lock` after a successful run to record the complete resolved environment, including the packages the article does not list.

---

## Running the analysis

```bash
make all             # scripts 01 through 05, in order
```

or one at a time, from the repository root:

```bash
python scripts/01_validacion_anidada.py
```

Script 01 takes roughly **1 hour 20 minutes** and the map scripts add some twenty minutes more; it fits five ordinal models inside an inner cross-validation for each of about 100 outer partitions, and also fits kriging for every candidate variogram. The other four scripts take seconds to a few minutes and read the outputs of 01.

Before committing to the full run, verify the pipeline end to end in a reduced configuration:

```bash
make test            # or: ORK_MODO_PRUEBA=1 python scripts/01_validacion_anidada.py
```

Test mode cuts the run to 2 random repetitions, 3 spatial blocks, 2 inner folds and 2 candidate variograms, and finishes in a few minutes. Its numbers are not the published ones; it exists to confirm that every stage executes.

---

## Repository layout

```
data/       municipal_dengue_dataset_mexico_2022.csv, data_dictionary.csv
            shapefile/  municipal geometries (not distributed; see its README)
scripts/    01-09, in execution order, plus rutas.py (path resolution)
            and figuras_png.py (resolution and format of the PNG outputs)
resultados/ tables and figures written by the scripts (generated)
figuras/    SHAP figures (generated)
docs/       reproduction notes
```

`scripts/rutas.py` and `scripts/figuras_png.py` are the only modules added when the analysis was organised as a repository. The first resolves the data and output paths relative to the repository root, so the scripts work from any working directory. The second sets the resolution of the PNG outputs from the width at which a figure is printed, and removes their alpha channel. Neither contains analysis logic.

---

## What each script does

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `01_validacion_anidada.py` | Main analysis: nested validation under two outer schemes, model and variogram selection, Moran's I, threshold calibration, Wilcoxon tests, coding-sensitivity analysis, per-municipality predictions | `data/` | `resultados/*.csv`, `resultados/*.png` |
| `02_matrices_confusion.py` | Row-normalised confusion matrices per scheme and predictor, plus a LaTeX version of the main one | `resultados/predicciones_por_municipio_*.csv` | `resultados/confusion_norm_*.csv`, `resumen_confusion_*.csv`, `confusion_*.tex` |
| `03_efecto_agrupacion.py` | Supplementary: whether merging adjacent risk levels improves discrimination, evaluated post hoc on the saved predictions | same | `resultados/efecto_agrupacion_*.csv` |
| `04_puntajes_codificacion.py` | Scores of the alternative category coding, recomputed per partition, for reporting their variability | `data/` | `resultados/puntajes_codificacion_alternativa.csv` |
| `05_shap_random_forest.py` | SHAP for the four cumulative binary classifiers of the ordinal Random Forest, fitted on the variables retained in at least 90% of partitions | `data/` + `resultados/seleccion_variables_*.csv` | `figuras/SHAP_Summary_DRL_greater_*.{pdf,png}`, `shap_importancia_rf.csv` |
| `06_figuras_bloque_espacial.py` | Maps for one held-out spatial block (observed and ORK-predicted), the kriged residual surface, and the observed levels as points | `data/` (+ shapefile, optional) | `figuras/figura09a_*`, `figura09b_*`, `figura10_*`, `figura11_observado.*` |
| `07_mapas_modelo_final.py` | Final-model maps: the continuous ordinal score over a 250×250 grid, and the out-of-sample choropleth for every municipality | `data/` + shapefile + `resultados/seleccion_variables_*.csv` | `figuras/mapaA_superficie_continua.*`, `mapaB_coropletas_oof.*` |
| `08_coropletas_observado.py` | Choropleth of the observed municipal risk levels | `data/` + shapefile | `figuras/figura11_observado_coropletas.*` |
| `09_figuras_descriptivas.py` | Descriptive figures and the association table: incidence distribution, risk-level distribution, Spearman matrix, and the variables meeting the screening criterion on the complete sample | `data/` | `figuras/figura01_*`, `figura02_*`, `figura03_*`, `resultados/tabla02_asociaciones.{csv,tex}` |

Scripts 02, 03, 05 and 07 locate the most recent output of script 01 by timestamp, so 01 must run first. Scripts 04, 06, 08 and 09 read only the dataset; 09 in particular runs in seconds and is a quick way to check that the data are in place.

The output file names of scripts 06 to 08 use the figure numbering of an earlier draft and do not match the numbering of the submitted manuscript; `docs/reproducing.md` gives the mapping.

---

## Predictors compared

Each outer partition scores nine predictors. The names in the third column are those used in the output tables.

| | Predictor | Name in the code |
|---|---|---|
| Baselines | Constant: the median of the training response | `Trivial` |
| | Geography only: median of the 8 nearest training municipalities | `kNN espacial` |
| Covariates only | Median rule applied to P(Y = k) | `Base` |
| | Rounded expected value E[Y], no kriging | `Base EV` |
| | Rounded E[Y] with the calibrated threshold | `Base EV cal` |
| Covariates + kriging | Kriging of discrete residuals, added to `Base` | `Metodo A` |
| | Kriging of continuous residuals, added to E[Y] — **this is ORK** | `Metodo B` |
| | ORK with the calibrated threshold | `Metodo B cal` |
| | ORK under the alternative category coding | `Metodo B log` |

The set is built so that each comparison isolates one thing. `Base EV` against `Base` isolates the effect of the decision rule; `Metodo B` against `Base EV` isolates the spatial contribution with the decision rule held fixed; `kNN espacial` against the covariate models shows how much of the performance geography alone accounts for; and `Trivial` establishes the floor, which matters because 55% of municipalities sit in level 0 and an exact-agreement metric flatters a constant predictor.

---

## Analysis design

**Two outer schemes, reported side by side.** The random scheme draws 50 stratified 80/20 partitions and estimates performance in a gap-filling regime, where the target municipality lies among municipalities with observed data. The spatial scheme partitions the municipal centroids into 5 contiguous blocks by *k*-means and holds one block out at a time, repeating the clustering 10 times with different seeds; it estimates performance when an entire region is unobserved. Blocks leaving fewer than 5 test or 30 training municipalities are skipped. Because the kriged correction is local by construction, its advantage is expected to be larger under the random scheme, and reporting only one of the two would misrepresent the method in one direction or the other.

**Selection is separated from evaluation.** Within each outer training set, an inner cross-validation selects the ordinal model among five candidates, the theoretical variogram among four, and the retained predictors. The outer test set is used once, for evaluation. No held-out municipality participates in any selection step, and in particular the Spearman screening is recomputed inside every training set rather than once on the complete sample.

**Distances are great-circle.** Kriging runs in PyKrige with `coordinates_type="geographic"`, so distances are computed on the sphere from longitude and latitude in decimal degrees. No planar projection is applied.

---

## Parameters

Everything below is declared at the top of `scripts/01_validacion_anidada.py`.

| | Value |
|---|---|
| Global seed | 123 |
| Ordered levels | 5 |
| Random scheme | 50 repetitions, 20% test, stratified, seed base 5000 |
| Spatial scheme | 5 *k*-means blocks × 10 repetitions, seed base 7000 |
| Inner folds, model selection | 5 |
| Folds for out-of-fold residuals | 5 |
| Folds for variogram selection | 3 |
| Predictor screening | Spearman \|ρ\| ≥ 0.10, minimum 3 variables |
| Collinearity pruning | rank correlation ≥ 0.95, keeping the stronger predictor |
| Candidate variograms | linear, exponential, gaussian, spherical |
| Variogram lags | 12, weighted fit |
| Kriging | ordinary, geographic coordinates, exact values, pseudo-inverse, 50 nearest points |
| Threshold calibration grid | 0.05 to 0.95 in steps of 0.01, prediction = floor(score + t) |
| Spatial kNN baseline | k = 8 |
| Moran's I | k = 8 nearest neighbours, row-standardised, 999 permutations |
| Random Forest | 500 trees, max depth 20, min samples leaf 2, sqrt features |
| XGBoost | 400 rounds, max depth 4, learning rate 0.03, subsample 0.80, colsample 0.80, min child weight 3, L2 = 1.0 |
| Ordered Logit | statsmodels `OrderedModel`, logit link, standardised inputs |
| LogisticAT / LogisticIT | mord, α = 1.0, standardised inputs |

Hyperparameters are fixed a priori and are never tuned on the data, so the only data-dependent choices are predictor screening, model selection and variogram selection — all three made inside the inner loop.

---

## Outputs

Script 01 timestamps its outputs as `*_YYYYMMDD_HHMM.csv`, so successive runs accumulate rather than overwrite. The downstream scripts always read the most recent set.

| File | Contents |
|---|---|
| `resultados_por_particion_*.csv` | Every metric for every predictor in every outer partition |
| `predicciones_por_municipio_*.csv` | Per-municipality predictions, the input to scripts 02 and 03 |
| `seleccion_variables_*.csv` | How often each covariate survived screening |
| `modelos_seleccionados_*.csv` | Which ordinal model the inner loop chose, and how often |
| `variogramas_seleccionados_*.csv` | Same, for the theoretical variogram |
| `frecuencia_mejora_*.csv` | Proportion of partitions in which each correction reduced the MAE |
| `wilcoxon_*.csv` | Paired signed-rank comparisons between predictors |
| `desempeno_por_clase_*.csv`, `error_grave_*.csv` | Per-level performance and the rate of errors spanning two or more levels |
| `mae_por_esquema_*.png`, `efecto_espacial_*.png` | Figures |

The Wilcoxon tables carry a methodological caveat stated in the script itself: outer partitions share observations, so the tests are not independent and the p-values should be read as descriptive summaries of consistency across partitions rather than as strict inference.

---

## License

MIT, see `LICENSE`. The dataset is distributed under the terms stated in its Zenodo deposit.

---

## Contact

Saúl Solorio-Fernández — sausolofer@inaoep.mx

Instituto Nacional de Astrofísica, Óptica y Electrónica, and Faculty of Mathematics No. 2, Autonomous University of Guerrero.

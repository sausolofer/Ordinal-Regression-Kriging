PYTHON ?= python
S := scripts

.PHONY: help env test all maps main confusion grouping scores shap blockmaps finalmaps choropleth descriptive lock clean

help:
	@echo "make env         Create the conda environment"
	@echo "make test        Run the main analysis in reduced configuration (minutes)"
	@echo "make all         Run the full analysis, scripts 01-09 (~2 h)"
	@echo "make descriptive Figures 1-3 and Table 2 only (seconds, dataset only)"
	@echo "make main        Script 01 only (~1 h 20 min)"
	@echo "make maps        Figure scripts 06-08 only (needs data/shapefile/00mun.shp)"
	@echo "make lock        Pin the exact environment versions"
	@echo "make clean       Remove generated results and figures"

env:
	conda env create -f environment.yml
	@echo "Now run: conda activate ork"
	@echo "If mord failed: SETUPTOOLS_USE_DISTUTILS=stdlib pip install mord"

# Verifies that every stage executes before committing to the full run.
test:
	ORK_MODO_PRUEBA=1 $(PYTHON) $(S)/01_validacion_anidada.py

all: descriptive main confusion grouping scores shap blockmaps finalmaps choropleth

maps: blockmaps finalmaps choropleth

main:
	$(PYTHON) $(S)/01_validacion_anidada.py

confusion:
	$(PYTHON) $(S)/02_matrices_confusion.py

grouping:
	$(PYTHON) $(S)/03_efecto_agrupacion.py

scores:
	$(PYTHON) $(S)/04_puntajes_codificacion.py

shap:
	$(PYTHON) $(S)/05_shap_random_forest.py

# Los tres siguientes dibujan mapas. 07 y 08 requieren el shapefile
# municipal en data/shapefile/; 06 funciona sin él, sin fondo geográfico.
blockmaps:
	$(PYTHON) $(S)/06_figuras_bloque_espacial.py

finalmaps:
	$(PYTHON) $(S)/07_mapas_modelo_final.py

choropleth:
	$(PYTHON) $(S)/08_coropletas_observado.py

# Sólo necesita el conjunto de datos: corre en segundos y en cualquier momento.
descriptive:
	$(PYTHON) $(S)/09_figuras_descriptivas.py

lock:
	conda env export --no-builds > environment.lock.yml
	pip freeze > requirements.lock.txt

clean:
	rm -rf resultados/* figuras/* $(S)/__pycache__
	touch resultados/.gitkeep figuras/.gitkeep

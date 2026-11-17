# Highland Lakes water mapping pipeline.  `make all` reproduces every result from public data.
# Requires uv (https://docs.astral.sh/uv/). GPU optional but strongly recommended for training.
# On a free Colab T4 expect roughly 3-4x the timings in README.md; set ITERS lower to go faster.

PY      := uv run python -W ignore -m
FOLDS   := holdout_travis holdout_buchanan
SEEDS   := 0 1 2
ITERS   ?= 2000
WORKERS ?= 5

.PHONY: all env data aoi levels scenes select download labels label-tool human-score recheck-score pond-audit-tool pond-audit-score blogfigs train train-s2 train-ablation \
        teacher train-s1 evaluate timeline analyze web weights transfer ponds test lint clean

all: data train evaluate timeline analyze weights web blogfigs transfer ponds

env:
	uv sync

# ---- data ---------------------------------------------------------------------------------
data: aoi levels scenes select download

aoi:            ## lake zones from JRC GSW + static weak-label layers
	$(PY) hlw.aoi
	$(PY) hlw.labels_fetch

levels:         ## TWDB daily lake level / area / storage
	$(PY) hlw.levels

scenes:         ## S2 clear-sky inventory per lake (SCL), S1 day list. Slow (~30 min).
	$(PY) hlw.scenes

select:         ## deterministic scene choice -> data/scenes/selected.json
	$(PY) hlw.selection

download:       ## selected S2 + S1 scenes onto lake grids
	$(PY) hlw.download

labels:         ## (re)render contact sheets for auditing the hand labels
	$(PY) hlw.handlabel sheets

label-tool:     ## images for the browser labelling tool; then open http://localhost:8000
	$(PY) hlw.handlabel chips
	@echo "Serve with: uv run python -m http.server 8000 -d labels/tool"

LABELS ?= labels/hand_points_human.csv
human-score:    ## agreement with the AI labels + re-score all methods on LABELS (results/human_labels/)
	$(PY) hlw.handlabel agree --human $(LABELS)
	$(PY) hlw.evaluate --labels $(LABELS) --fill-shipped

recheck-score:  ## merge the blind second look (tool ?set=recheck) and re-score on the final labels
	$(PY) hlw.handlabel merge-recheck
	$(PY) hlw.handlabel agree --human labels/hand_points_human_final.csv
	$(PY) hlw.evaluate --labels labels/hand_points_human_final.csv --fill-shipped

pond-audit-tool: ## images for the stock-tank outline check; then open http://localhost:8000/ponds.html
	$(PY) hlw.ponds audit-tool
	@echo "Serve with: uv run python -m http.server 8000 -d labels/tool  (open /ponds.html)"

pond-audit-score: ## compare your pond check with the AI's and re-run the pond analysis
	$(PY) hlw.ponds audit-compare analyze

blogfigs:       ## map figures (results/figures/blog_*.png)
	$(PY) hlw.blogfigs

# ---- training -----------------------------------------------------------------------------
train: train-s2 train-ablation teacher train-s1

train-s2:
	for f in $(FOLDS); do for s in $(SEEDS); do \
	  $(PY) hlw.train --sensor s2 --fold $$f --seed $$s --iters $(ITERS) --workers $(WORKERS) || exit 1; \
	done; done

train-ablation: ## no pretraining, and static JRC labels instead of per-scene SCL labels
	for f in $(FOLDS); do \
	  $(PY) hlw.train --sensor s2 --fold $$f --init none --iters $(ITERS) --workers $(WORKERS) || exit 1; \
	  $(PY) hlw.train --sensor s2 --fold $$f --label jrc50 --iters $(ITERS) --workers $(WORKERS) || exit 1; \
	done

teacher:
	for f in $(FOLDS); do $(PY) hlw.teacher --fold $$f || exit 1; done

train-s1:
	for f in $(FOLDS); do \
	  $(PY) hlw.train --sensor s1 --fold $$f --iters $(ITERS) --workers $(WORKERS) || exit 1; \
	done

# ---- evaluation & products ----------------------------------------------------------------
evaluate:       ## hand-labelled point metrics, weak-label noise, calibration
	$(PY) hlw.evaluate

timeline:       ## map every usable S2/S1 date, streamed (slow: ~2-3 h, network bound)
	$(PY) hlw.timeline
	$(PY) hlw.timeline --fix-index   # NDWI areas (+ MNDWI at the current cutoffs), ~15 min

analyze:        ## area vs official level, cloud table, figures
	$(PY) hlw.analyze

weights:        ## copy the shipped models into weights/
	$(PY) hlw.export_weights

web:            ## build data for the static web app in web/
	$(PY) hlw.webexport

transfer:       ## shipped models on 10 small "pseudo-ungauged" reservoirs (configs/transfer.yaml)
	$(PY) hlw.transfer zones levels scan timeline fix-cutoffs analyze

ponds:          ## shipped models on ungauged stock tanks, scored against NAIP aerial photos (configs/ponds.yaml)
	$(PY) hlw.ponds naip s2 s1 analyze

test:
	uv run pytest -q -m "not network"

lint:
	uv run ruff check src tests

clean:
	rm -rf data/raw data/runs data/outputs data/teacher data/label_sheets

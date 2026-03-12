# Usage:
#   make scrape                           # scrape default dataset
#   make process DATASET=pokemondb-gen3   # tokenize a specific dataset
#   make preview DATASET=pokemondb-gen3   # tokenize + contact sheet
#   make upload  DATASET=pokemondb-gen3   # upload raw + processed + config to HF

HF_REPO = akirashengchang/pixel-sprites
DATASET ?= pokemondb-gen3

.PHONY: scrape process preview upload upload-raw upload-processed

scrape:
	python scripts/$(DATASET)/scrape.py

process:
	python scripts/process.py $(DATASET)

preview:
	python scripts/process.py $(DATASET) --preview

upload: upload-raw upload-processed
	hf upload $(HF_REPO) data/$(DATASET)/dataset.json $(DATASET)/dataset.json --repo-type dataset

upload-raw:
	hf upload $(HF_REPO) data/$(DATASET)/raw/ $(DATASET)/raw/ --repo-type dataset

upload-processed:
	hf upload $(HF_REPO) data/$(DATASET)/processed/ $(DATASET)/processed/ --repo-type dataset

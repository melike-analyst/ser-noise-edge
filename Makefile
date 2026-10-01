# Usage:  make demo                      (no dataset needed, synthetic data, plumbing check)
#         make experiments DATA=data/ravdess
DATA ?= data/ravdess
EPOCHS ?= 30
TAG ?= $(shell hostname)

test:
	python -m pytest -q

demo:
	python -m ser.synthetic --out data/synthetic
	python -m ser.train --data_root data/synthetic --model cnn --out runs/demo --epochs 3
	python -m ser.export_onnx --run runs/demo
	python -m ser.evaluate --data_root data/synthetic --run runs/demo --variant onnx_int8 --snrs 10 0 --noise_kinds white cabin
	python -m ser.benchmark --models runs/demo/model.onnx runs/demo/model.int8.onnx --runs 50 --tag demo

experiments:
	python -m ser.baseline --data_root $(DATA)
	python -m ser.train --data_root $(DATA) --model cnn  --out runs/cnn      --epochs $(EPOCHS)
	python -m ser.train --data_root $(DATA) --model cnn  --out runs/cnn_aug  --epochs $(EPOCHS) --aug_noise
	python -m ser.train --data_root $(DATA) --model crnn --out runs/crnn_aug --epochs $(EPOCHS) --aug_noise
	for r in cnn cnn_aug crnn_aug; do \
	  python -m ser.export_onnx --run runs/$$r; \
	  for v in torch onnx_fp32 onnx_int8; do python -m ser.evaluate --data_root $(DATA) --run runs/$$r --variant $$v; done; \
	done
	python -m ser.benchmark --models $$(ls runs/*/model.onnx runs/*/model.int8.onnx) --tag $(TAG)
	python -m ser.plots

.PHONY: test demo experiments

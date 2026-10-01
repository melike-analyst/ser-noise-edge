
**Speech Emotion Recognition (SER) under noise, from training to edge deployment.**

Most SER tutorials stop at "accuracy on a clean test set". This project asks the
questions that matter when a model has to run for real, for example on a device in a car:

1. How much does accuracy drop when the audio is noisy?
2. Does training with noise augmentation help?
3. After converting to **ONNX** and quantizing to **INT8**, how much accuracy do we lose,
   and what do we gain in model size and speed?
4. Does it run in real time on a small device?

Every number below comes from a file in `results/` or `runs/`, and everything can be
reproduced with the commands in this README.

```
 wav -> trim/pad 3 s -> [+ noise at chosen SNR] -> log-mel (NumPy) -> model -> emotion
                                                      |
            train / evaluate (PyTorch)    export -> ONNX FP32 -> INT8 -> benchmark (ONNX Runtime)
```

## Status

| Experiment | Status |
|---|---|
| MFCC + SVM baseline, clean and noisy test | done |
| CNN **without** noise augmentation, clean and noisy test | done |
| CNN **with** noise augmentation, clean and noisy test (ONNX FP32 and INT8) | done |
| ONNX FP32 / INT8 export, parity check | done |
| Latency / memory benchmark on an x86-64 laptop | done |
| Real noise recordings (`--noise_dir`) | not run yet |
| CRNN (CNN + GRU) | not run yet |
| Several random seeds, longer training | not run yet |
| Benchmark on an ARM64 device | **not done**: no ARM64 hardware was available (see [Benchmarking on ARM64](#benchmarking-on-arm64-not-done-here-how-to-do-it)) |

## Results

Setup: RAVDESS speech, 8 emotions, **speaker-independent** split (actors 1-16 train,
17-20 validation, 21-24 test = 240 test clips). One training run per model (seed 42),
30 epochs. Chance level is UAR = 0.125. UAR = unweighted average recall (mean of per-class recall).
The noisy test clips are generated with fixed seeds, so every model sees identical inputs.

**How much to trust the numbers.** With 240 test clips from only 4 speakers, a UAR value has a
95% uncertainty of roughly +/- 0.06 (a rough estimate that treats clips as independent, so the
real uncertainty is probably larger). Differences smaller than that, from a single run, should
not be read as real effects.

### 1. Clean test set

| Model | Noise aug. | UAR | Macro-F1 | Accuracy | Best val UAR |
|---|---|---|---|---|---|
| MFCC + SVM | no | 0.348 | 0.351 | 0.346 | - |
| CNN (98k parameters) | no | 0.422 | 0.383 | 0.421 | 0.508 |
| CNN (98k parameters) | yes | 0.441 | 0.362 | 0.438 | 0.426 |
| CNN, ONNX FP32 | yes | 0.441 | 0.362 | 0.438 | - |
| CNN, ONNX INT8 | yes | 0.438 | 0.360 | 0.433 | - |

Sources: `runs/*/metrics.json`, `results/robustness_*.csv`.

### 2. Robustness to noise

![Robustness to noise](results/robustness.png)

UAR at selected SNRs (dB). Each cell: **CNN without aug. / CNN with aug. (ONNX FP32) / MFCC + SVM**.
Clean test: 0.422 / 0.441 / 0.348.

| Noise | 20 dB | 10 dB | 0 dB | -5 dB |
|---|---|---|---|---|
| white | 0.152 / 0.465 / 0.191 | 0.125 / 0.449 / 0.129 | 0.125 / 0.355 / 0.129 | 0.125 / 0.324 / 0.125 |
| pink | 0.160 / 0.473 / 0.234 | 0.129 / 0.465 / 0.172 | 0.125 / 0.371 / 0.160 | 0.125 / 0.324 / 0.129 |
| brown | 0.391 / 0.480 / 0.293 | 0.367 / 0.492 / 0.277 | 0.270 / 0.473 / 0.281 | 0.242 / 0.469 / 0.223 |
| cabin (synthetic) | 0.336 / 0.465 / 0.254 | 0.281 / 0.438 / 0.266 | 0.152 / 0.430 / 0.148 | 0.133 / 0.441 / 0.145 |

Mean UAR over all 24 noisy conditions (4 noise types x 6 SNR levels):
CNN without aug. **0.21**, MFCC + SVM **0.20**, CNN with aug. **0.44** (FP32) / **0.44** (INT8).
The augmented CNN scores higher than the non-augmented one in all 24 conditions, by 0.09 to 0.34.

### 3. Which emotions are confused (clean test)

| CNN without noise aug. | CNN with noise aug. (ONNX FP32) |
|---|---|
| ![](results/confusion_cnn_torch.png) | ![](results/confusion_cnn_aug_onnx_fp32.png) |

Per-class recall, without aug. -> with aug.: surprised 0.59 -> 0.94, calm 0.78 -> 0.84,
angry 0.72 -> 0.66, neutral 0.44 -> 0.50, disgust 0.38 -> 0.41, happy 0.25 -> 0.12,
fearful 0.19 -> 0.06, **sad 0.03 -> 0.00**. Neither model really predicts "sad". Frequent confusions in both:
sad -> calm (about 0.5), disgust -> angry (0.5-0.6), happy -> angry / surprised; fearful is spread over several classes (angry, disgust, happy).
(The INT8 model's matrix is in `results/confusion_cnn_aug_onnx_int8.png` and is very close to the FP32 one.)

### 4. Size, speed and memory (x86-64 laptop, 1 thread, 3 s clip, 200 runs)

| Model | Size (MB) | Features p50 (ms) | Model p50 (ms) | Total p50 / p95 (ms) | RTF | Peak RAM (MB) | Clean UAR |
|---|---|---|---|---|---|---|---|
| CNN FP32 | 0.396 | 2.22 | 1.18 | 3.40 / 3.77 | 0.0011 | 66.5 | 0.441 |
| CNN INT8 | 0.111 | 2.52 | 1.65 | 4.17 / 4.62 | 0.0014 | 68.2 | 0.438 |

Machine: Intel CPU (`Intel64 Family 6 Model 141`), Windows (which reports x86-64 as "AMD64"),
Python 3.11.5, ONNX Runtime 1.30.0. Source: `results/benchmark_laptop.csv`.
RTF = real-time factor = latency / audio duration (below 1 is faster than real time).

![Accuracy vs latency](results/tradeoff.png)

## Findings

**Supported by the data**

- **Noise augmentation is what makes the CNN robust.** Without it, the CNN is as fragile as the classical
  baseline (mean UAR over noisy conditions 0.21 vs 0.20 for the SVM) and drops to chance (about 0.125) already at
  10 dB white or pink noise. With it, UAR stays around 0.32-0.37 even at 0 to -5 dB white/pink noise and
  0.43-0.49 for brown and cabin noise. The gaps (0.09-0.34 in all 24 conditions) are larger than the
  uncertainty of a single measurement. This also settles the earlier confound: the robustness of the CNN comes from
  the augmentation, not from the architecture alone.
- **On clean audio the CNN beats the baseline** (UAR 0.42-0.44 vs 0.35), a modest but likely real gap.
- **Damage depends on the noise spectrum.** White and pink noise are the most harmful. Brown and the synthetic
  cabin noise are much less harmful, even for the CNN without augmentation (mean UAR 0.33 for brown, 0.24 for cabin,
  vs 0.13 for white and pink). A plausible explanation is that white noise spreads energy over all frequencies and masks the
  mid/high mel bands, while brown and cabin noise put most of their energy at very low frequencies, below where most
  emotion cues are. This explanation was not tested directly.
- **INT8 costs no measurable accuracy and makes the file 3.6x smaller.** Clean UAR 0.438 (INT8) vs 0.441 (FP32),
  about 1 clip of 240. Across all 24 noisy conditions the INT8-minus-FP32 difference averages -0.003 and stays
  between -0.023 and +0.016, i.e. no systematic loss. ONNX FP32 reproduces the PyTorch result exactly (UAR 0.441;
  max output difference 2.9e-06).
- **INT8 was slower, not faster, on this laptop.** Model time +40% (1.18 -> 1.65 ms), total time +23%,
  peak RAM about the same (+1.7 MB). Dynamic quantization adds quantize/dequantize work at run time, and on this CPU that
  cost outweighs the integer arithmetic gain. In the accuracy-vs-latency plot the FP32 model is therefore at least as good
  on both axes; INT8 wins only on **file size** (relevant for storage or over-the-air updates). This is one machine; another
  CPU, especially an ARM64 one, may behave differently, which is why the benchmark needs to be repeated on the
  target hardware.
- **Feature extraction costs more than the model.** NumPy log-mel takes 60-65% of total time. If latency ever
  mattered, that is the first thing to optimize, not the network.
- **Both pipelines are far faster than real time on this laptop** (roughly 700-900x). Peak RAM (about 66-68 MB) is
  dominated by the Python/ONNX Runtime process, not by the 0.4 MB model.
- **Errors fall between emotions with similar energy** (sad/calm, happy/fearful/surprised/angry), typical for SER.
  Neither model really predicts "sad" (recall 0.00-0.03), so this is not an augmentation artifact.

**Not supported yet (do not claim these)**

- *"Noise augmentation improves clean accuracy."* UAR 0.422 vs 0.441 is a difference of about 5 clips, inside the
  uncertainty, and macro-F1 goes the other way (0.383 vs 0.362). The per-class pattern also changes (surprised, calm
  better; happy, fearful worse). The data shows no clear clean-accuracy cost of augmentation, but also no clear gain.
- *"The model generalizes to real-world noise."* The test noise types (white, pink, brown, cabin) are generated by the
  same code as the training augmentation, so these results are for **noise types seen in training**. Real recordings
  were not tested.
- *"Noisy audio is easier than clean audio for the augmented model."* Several noisy conditions score above clean
  (for example brown noise at every SNR: 0.46-0.49 vs 0.44). Each gap is within the uncertainty, and the FP32 and INT8
  results are the same model, so they are not independent confirmations. A possible reason is that half of the training
  clips were noisy, which makes moderately noisy audio closer to the training data. This was not tested.
- *"It would work in a car."* RAVDESS is acted studio speech, and "cabin" is synthetic noise.

## Limitations of this study

- Single training run, single seed, small test set (240 clips, 4 speakers). The validation UAR
  jumps by 0.1 or more between epochs (`runs/*/history.csv`), and the best-epoch choice on a
  small validation set is itself noisy. For example the CNN without augmentation scores 0.508 on validation but 0.422
  on test, so the clean-test ranking of the two CNNs could change with another seed or split.
- Training had not converged: the loss was still decreasing at epoch 30 and validation UAR was
  still rising for the CNN without augmentation. Longer training or a larger model would
  probably score higher. The absolute scores (UAR about 0.44) are modest for RAVDESS.
- Almost no "sad" predictions (recall 0.00-0.03). This should be investigated (more epochs,
  other seeds, class weighting, a larger model) before trusting per-class conclusions.
- Acted English speech by 24 North American actors, recorded in a studio. Results will not
  transfer directly to spontaneous speech, other languages or real in-car audio.
- "cabin" noise is a synthetic approximation, not a recording, and the evaluation noise types equal the training
  noise types.
- SNR is computed over the whole 3 s clip, including any zero padding.
- Latency was measured on one x86-64 laptop with one thread. **There is no ARM64 measurement** because no ARM64
  hardware was available, so statements about edge devices are limited to what this laptop shows. INT8 in particular
  may behave differently on ARM64.

## Next steps

The open questions, in order of value:

```bash
# 1) noise the model has NOT seen in training (download e.g. ESC-50 / DEMAND / MUSAN, check the license)
python -m ser.evaluate --data_root data/ravdess --run runs/cnn_aug --variant onnx_fp32 --noise_dir path/to/noise_wavs
python -m ser.evaluate --data_root data/ravdess --run runs/cnn     --variant torch     --noise_dir path/to/noise_wavs
python -m ser.baseline --data_root data/ravdess --noise_dir path/to/noise_wavs

# 2) other seeds and longer training (to check the clean-accuracy comparison and the "sad" problem)
python -m ser.train --data_root data/ravdess --model cnn --aug_noise --epochs 60 --seed 1 --out runs/cnn_aug_s1
python -m ser.train --data_root data/ravdess --model cnn            --epochs 60 --seed 1 --out runs/cnn_s1

python -m ser.plots
```

ARM64 measurements are described in [Benchmarking on ARM64](#benchmarking-on-arm64-not-done-here-how-to-do-it).
Further ideas: the CRNN (`--model crnn`), pre-trained encoders (wav2vec 2.0, WavLM, emotion2vec, Whisper
encoder) as feature extractors on the same noise sweep, arousal / valence regression with CCC (needs a
dataset such as MSP-Podcast, check its access terms), static INT8 with calibration data, and a
cross-corpus test.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

make test      # unit tests + end-to-end smoke test on synthetic data (~1-2 min on CPU)
make demo      # tiny synthetic dataset, checks the whole pipeline, no download needed
```

The synthetic demo only proves that the code runs. Its scores mean nothing about real speech.

### Run on real data (RAVDESS)

1. Download `Audio_Speech_Actors_01-24.zip` from the RAVDESS page on Zenodo
   (https://zenodo.org/records/1188976) and unzip it into `data/ravdess/`.
   The dataset is licensed **CC BY-NC-SA 4.0** (non-commercial, share-alike, attribution).
   Do not commit the audio to the repository.
2. Run the experiments:

```bash
python -m ser.baseline  --data_root data/ravdess                               # MFCC + SVM reference
python -m ser.train     --data_root data/ravdess --model cnn --out runs/cnn    # clean training
python -m ser.train     --data_root data/ravdess --model cnn --out runs/cnn_aug --aug_noise
python -m ser.export_onnx --run runs/cnn_aug                                   # ONNX + INT8
python -m ser.evaluate  --data_root data/ravdess --run runs/cnn_aug --variant onnx_int8
python -m ser.benchmark --models runs/cnn_aug/model.onnx runs/cnn_aug/model.int8.onnx --tag laptop
python -m ser.plots
```

Or `make experiments DATA=data/ravdess EPOCHS=30 TAG=laptop` for the full set.

For real noise, download a set such as ESC-50, DEMAND or MUSAN, check its license, and add
`--noise_dir path/to/noise_wavs` to `ser.evaluate` and `ser.baseline`. Results then include a `real` noise type.

## Benchmarking on ARM64 (not done here, how to do it)

The benchmark in this repo was **only run on an x86-64 laptop**, because the author has no ARM64 device.
The tooling is ready, so anyone with access to ARM64 hardware can repeat it. Inference needs only
`numpy`, `pandas` and `onnxruntime` (no PyTorch). The command is the same on every machine, only `--tag` changes:

```bash
python -m ser.benchmark --models models/cnn_aug/model.onnx models/cnn_aug/model.int8.onnx --tag my-arm-device --threads 1
```

The CSV records the architecture (`arch`), CPU name, ONNX Runtime version and thread count, so results from
different machines can be compared with `python -m ser.plots`.

Ways to get ARM64 numbers, from closest to a real edge device to least:

1. **Raspberry Pi 4/5 or a similar single-board computer.** Closest to a real edge scenario. Copy the repo and the
   two `.onnx` files, install the three packages, run the command above.
2. **GitHub Actions ARM64 runner (free, no hardware).** The optional workflow
   `.github/workflows/benchmark-arm64.yml` runs the benchmark on `ubuntu-24.04-arm` and uploads the CSV as an
   artifact. It requires a **public** repository and the exported models committed under
   `models/cnn_aug/` (`model.onnx`, `model.int8.onnx`, together about 0.5 MB). Trigger it manually from the
   Actions tab. This is a shared cloud VM, so treat the numbers as "ARM64 server-class CPU", not as proof that
   the model runs on a small board. The commands in the workflow were dry-run on x86-64 only.
3. **An Apple Silicon Mac** is ARM64, but it is a fast laptop-class chip, not an edge device.
4. **Cloud ARM instances** (for example AWS Graviton). Check current pricing and free tiers.
5. **Docker with QEMU emulation** is fine for checking that the code runs on ARM64, but the timings are **not
   meaningful**, because the CPU is emulated.

When reporting ARM64 results, keep the same settings (1 thread, 3 s clip, 200 runs) and compare FP32 against
INT8 on that machine. Do not assume the laptop result (INT8 slower) carries over.

## What is inside

| Part | What it does |
|---|---|
| Data | RAVDESS speech (8 emotions), **speaker-independent** split by actor |
| Features | Log-mel spectrogram in plain NumPy (no torchaudio needed at inference) |
| Models | MFCC + SVM baseline, small CNN, CNN + bidirectional GRU (CRNN), all in PyTorch |
| Noise | White, pink, brown, synthetic "cabin" noise, and optional real noise files, mixed at exact SNR levels |
| Metrics | UAR, macro-F1, accuracy, confusion matrix |
| Export | PyTorch -> ONNX (with numerical parity check) -> dynamic INT8 quantization |
| Benchmark | Latency p50/p95, real-time factor, peak memory, model size, saved with CPU/architecture info |
| Tests | `pytest`, including an end-to-end run on synthetic data |

## Design decisions (and why)

- **Split by actor, not by file.** Clips of the same speaker in train and test leak voice identity
  and inflate scores. Gender-balanced: actors 1-16 train, 17-20 validation, 21-24 test.
- **UAR as the main metric.** Class sizes are not equal (RAVDESS has fewer "neutral" clips), and UAR
  is not dominated by the biggest class.
- **Model selection on validation only.** The test set is used once per model.
- **Same noisy test clips for every model.** Noise is generated with fixed seeds, so FP32, INT8 and
  the baseline are compared on identical inputs.
- **Features in NumPy.** One implementation for training and deployment, and no PyTorch in the
  inference process, which keeps memory measurements honest.
- **Each benchmark in its own process.** Peak memory of one model is not affected by the others.
- **A baseline first.** A neural network is only interesting if it beats MFCC + SVM.

## Project structure

```
src/ser/
  data.py         RAVDESS parsing, actor split, loading
  features.py     log-mel and MFCC statistics (NumPy)
  noise.py        noise types + exact-SNR mixing
  models.py       SmallCNN, CRNN
  train.py        training (optional noise augmentation)
  baseline.py     MFCC + SVM with the same noise sweep
  evaluate.py     clean + noisy evaluation for torch / onnx_fp32 / onnx_int8
  export_onnx.py  ONNX export, parity check, INT8 quantization
  benchmark.py    latency / RTF / memory / size on the current machine
  plots.py        robustness and trade-off figures
  synthetic.py    tiny fake dataset for smoke tests
tests/            pytest (unit + end-to-end)
results/          CSVs and figures produced by the experiments
models/           (optional) exported ONNX files, needed only for the ARM64 workflow
.github/workflows/benchmark-arm64.yml   (optional) manual ARM64 benchmark on GitHub Actions
```

## License

Code: MIT (see `LICENSE`). Datasets keep their own licenses.


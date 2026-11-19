# AI Reimplementation of RSLDA+ for Handwritten Digit Recognition

This repository contains an AI-generated NumPy reimplementation of the alternating optimization in RSLDA+ and a reproducible handwritten-digit experiment.

## Sources

- Liu, Feng, Xiu, and Liu (2024), *Towards robust and sparse linear discriminant analysis for image classification*, Pattern Recognition 153, 110512: https://doi.org/10.1016/j.patcog.2024.110512
- Authors' MATLAB reference code: https://github.com/EMXlight/RSLDAplus
- UCI Optical Recognition of Handwritten Digits: https://doi.org/10.24432/C50P49

The Python code is an educational AI-generated translation guided by the paper and authors' source. It is not the authors' official implementation. Dataset files download from UCI automatically and are kept out of this repository.

## Run

Requires Python 3.10+ and NumPy. Install with `python -m pip install -r requirements.txt`, then run `python run_experiment.py` from the repository root. The script standardizes using training statistics only, selects parameters on a stratified validation split from the official training data, refits on the full training set, and writes CSV/JSON metrics under `results/`.

## Results

| Method | Dimensions | Clean accuracy | 10% salt-and-pepper accuracy |
|---|---:|---:|---:|
| Raw pixels + 5-NN | 64 | 96.49% | 51.14% |
| PCA + 5-NN | 9 | 91.93% | 48.08% |
| Fisher LDA + 5-NN | 9 | 95.94% | 61.83% |
| RSLDA+ + 5-NN | 9 | 93.99% | 80.24% |

The corruption test replaces 10% of test pixel-block values with random black or white values; all models are trained on clean data. RSLDA+ had the highest accuracy in this specific corruption condition, while raw pixels and Fisher LDA were stronger on clean images. The final RSLDA+ run reached the 200-iteration cap without meeting the 1e-7 residual tolerance (final reconstruction residual infinity norm 1.236e-4), so its result is from a capped approximate solution.

Full metrics, parameter search, and run settings are in `results/metrics.csv` and `results/summary.json`.

## AI use

AI was used to search the method, translate the reference update structure, generate the experiment code, run the evaluation, and prepare this repository. The authors' paper and MATLAB source are credited above.

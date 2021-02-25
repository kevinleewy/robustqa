# CS224N Default Final Project (RobustQA track)

## Starter code for robustqa track

- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`

## Training

### Baseline

```bash
python train.py --do-train --run-name baesline
```

### Adversarial

```bash
python train.py --do-train --run-name adversarial --adversarial
```

### Category

```bash
python train.py --do-train --run-name baseline-cat0 --category during --load-dir save/baseline-01
```

### Category + Adversarial

```bash
python train.py --do-train --run-name adversarial-cat0 --adversarial --category during --load-dir save/adversarial-01
```

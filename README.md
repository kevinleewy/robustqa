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
python train.py --do-train --run-name baseline
```

### Adversarial

```bash
python train.py --do-train --run-name adversarial --adversarial --num-epochs 5 --batch-size 32 --visualize-predictions
```

### Category

```bash
python train.py --do-train --run-name baseline-cat0 --category during --load-dir save/baseline-01 --num-epochs 10  --lr 1e-6 --eval-every 200
```

### Category + Adversarial

```bash
python train.py --do-train --run-name adversarial-cat0 --adversarial --category during --load-dir save/adversarial-01 --num-epochs 10  --eval-every 2000
```

### Baseline + Finetune by Category

```bash
python train.py --do-train --run-name baseline-finetune --load-dir save/baseline-01 --num-epochs 10 --lr 1e-6 --eval-every 200
```

### Adversarial + OOD only finetune

```bash
python train.py --do-train --run-name adv-ood-finetune --adversarial \
    --load-dir save/adversarial-03 \
    --num-epochs 5 --batch-size 32 \
    --visualize-predictions
```

### Baseline + Finetune by Domain

```bash
python train.py --do-train --do-finetune \
    --run-name baseline-finetune \
    --dis_lambda 0.1 \
    --num-epochs 5 \
    --batch-size 32 \
    --visualize-predictions
```

### Adversarial + Finetune by Domain

```bash
python train.py --do-train --do-finetune \
    --run-name adv-finetune \
    --adversarial \
    --dis_lambda 0.1 \
    --num-epochs 10 \
    --batch-size 32 \
    --visualize-predictions
```

## Validation

### Standard (Per category + All categories)

#### [Standard] Using Baseline model

```bash
python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val
```

#### [Standard] Using Categorical model

```bash
python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-cat0-01 --eval-dir datasets/oodomain_val
```

### Single Category

#### [Single Category] Using Baseline model

```bash
python train.py --do-eval --sub-file mtl_submission_val_cat0.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val --category during
```

#### [Single Category] Using Categorical model

```bash
python train.py --do-eval --sub-file mtl_submission_val_cat0.csv --save-dir save/baseline-cat0 --eval-dir datasets/oodomain_val --category during
```

### Ensemble

```bash
python train.py --do-eval --do-ensemble --sub-file mtl_submission_val.csv --run-name ensemble --log-file ensemble --eval-dir datasets/oodomain_val
```

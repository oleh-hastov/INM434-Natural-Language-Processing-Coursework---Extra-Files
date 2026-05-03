# INM434 NLP Coursework: From Bag-of-Words to Transformers

A four-tier comparison of SMS spam detection methods, built for INM434 Natural Language Processing module.

**Author:** Oleh Hastov

This repository describes the the full codebase submited to Moodle as "nlp_spam_cw_final.zip", the data splits, the unit tests, and the instructions for evaluating the two best-trained models on the test set.

## Contents

- [The two best-trained models](#the-two-best-trained-models)
- [Software requirements](#software-requirements)
- [Test set](#test-set)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [Running Model 1: T4A_bert_lr3e-5](#running-model-1-t4a_bert_lr3e-5)
- [Running Model 2: T3 C04](#running-model-2-t3-c04)
- [Reproducing from scratch (optional)](#reproducing-from-scratch-optional)
- [Sanity checks](#sanity-checks)
- [Project structure](#project-structure)
- [Academic integrity](#academic-integrity)

## The two best-trained models

I picked these two by ranking every cell on the joint deployment criterion (clean test macro-F1 combined with ham-side robustness across 5 seeds). Model 1 is the strongest overal under that criterion; Model 2 is the strongest non-BERT model under the same criterion.

### Model 1: T4A_bert_lr3e-5

Tier 4. BERT-base-uncased, full fine-tune on the balanced training split.

| Setting | Value |
| --- | --- |
| Architecture | bert-base-uncased |
| Trainable | all parameters |
| Learning rate | 3e-5 |
| Batch size | 16 |
| Max length | 128 |
| Epochs | 2 |
| Seed | 123 (plus 4 cross-seed siblings) |

| Metric | Single seed (123) | Cross-seed (5 seeds) |
| --- | --- | --- |
| Val macro-F1 | 0.9865 | 0.9840 +/- 0.0038 |
| Test macro-F1 | 0.9657 | 0.9747 +/- 0.0069 |

### Model 2: T3 C04

Tier 3. GPT-2-small (124M parameters), end-to-end fine-tune with the flexible-token classification head.

| Setting | Value |
| --- | --- |
| Architecture | gpt2 (small, 124M) |
| Trainable | all parameters |
| Token position | flexible_token |
| Context length | 120 |
| Learning rate | 5e-5 |
| Batch size | 8 |
| Dropout | 0.1 |
| Seed | 123 (plus 4 cross-seed siblings) |

| Metric | Single seed (123) | Cross-seed (5 seeds) |
| --- | --- | --- |
| Val macro-F1 | 0.9863 | 0.9856 +/- 0.0019 |
| Test macro-F1 | 0.9727 | 0.9758 +/- 0.0073 |

## Software requirements

Tested with Python 3.10 and 3.11 on macOS (MPS), and Google Colab T4 GPU. About 6 GB of free disk space if you want every checkpoint locally.

The full pinned list is in `requirements.txt`. The relevant ones for evaluation are:

- numpy >= 1.26
- pandas >= 2.2
- scikit-learn >= 1.4
- torch >= 2.2, < 3.0
- transformers >= 4.40
- tiktoken >= 0.7
- matplotlib >= 3.8
- pyarrow >= 14

Set up a virtual environment from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate
pip install -r requirements.txt
```

Evaluation runs in a few minutes per model on CPU. A GPU is not required for evaluation, only for re-training.

## Test set

The clean test split (1032 rows, natural ~12.4% spam) is already in this repository at `data/splits/test.csv`. The adversarial sets are in the same directory:

```
data/splits/test_adv_mild.csv
data/splits/test_adv_moderate.csv
data/splits/test_adv_aggressive.csv
data/splits/test_adv_ham_mild.csv
data/splits/test_adv_ham_moderate.csv
data/splits/test_adv_ham_aggressive.csv
```

If for any reason you want to regenerate the splits from raw UCI SMS Spam Collection (the result is byte-identical because of the fixed seed):

```bash
python -m shared.splits
python -m data.make_adversarial_test
```

A copy of the splits directory is also available on Drive if you cannot or do not want to rebuild from UCI:

```
https://drive.google.com/drive/folders/<DRIVE_TEST_DATA_FOLDER_ID>
```

(replace the placeholder if you go this route, otherwise just use the in-repo CSVs).

## Pretrained checkpoints

Tier 1 (sklearn `.joblib` files) and Tier 2 (the LSTM `best_model.pt`) checkpoints are small and ship inside the repository:

```
tier1_classical/reports/checkpoints/
tier2_lstm/reports/best_model.pt
```

Tier 3 GPT-2 checkpoints (~500 MB each) and Tier 4 BERT/DistilBERT checkpoints (~250-500 MB each) are too big for git. They live in my Google Drive backup folder. Please download from:

```
https://drive.google.com/drive/folders/<DRIVE_CHECKPOINTS_FOLDER_ID>
```

The Drive folder mirrors what the training notebooks rsync to during Colab runs:

```
INM434_artifacts/
├── tier3/
│   ├── checkpoints/
│   │   ├── C04_s123.pt           <-- THIS for Model 2
│   │   ├── C04_s456.pt
│   │   ├── C04_s789.pt
│   │   ├── C04_s1011.pt
│   │   ├── C04_s1213.pt
│   │   └── ...other Tier 3 cells
│   ├── predictions/
│   ├── metadata/
│   └── history/
└── tier4/
    ├── checkpoints/
    │   ├── T4A_bert_lr3e-5.pt    <-- THIS for Model 1
    │   ├── T4B_bert_lr3e-5_s456.pt
    │   ├── T4B_bert_lr3e-5_s789.pt
    │   ├── T4B_bert_lr3e-5_s1011.pt
    │   ├── T4B_bert_lr3e-5_s1213.pt
    │   └── ...other Tier 4 cells
    ├── predictions/
    ├── metadata/
    └── history/
```

To run the two best models you only need two files from Drive:

| Model | Drive path | Place at |
| --- | --- | --- |
| Model 1 | `INM434_artifacts/tier4/checkpoints/T4A_bert_lr3e-5.pt` | `tier4_bert/reports/checkpoints/T4A_bert_lr3e-5.pt` |
| Model 2 | `INM434_artifacts/tier3/checkpoints/C04_s123.pt` | `tier3_gpt2/reports/checkpoints/C04_s123.pt` |

## Running Model 1: T4A_bert_lr3e-5

Confirm the checkpoint is in place:

```bash
ls tier4_bert/reports/checkpoints/T4A_bert_lr3e-5.pt
```

Run the Tier 4 evaluation script pointing at this checkpoint:

```bash
python -m tier4_bert.scripts.evaluate \
    --checkpoint tier4_bert/reports/checkpoints/T4A_bert_lr3e-5.pt
```

What the script does:

- Reloads BERT-base and loads the trained weights.
- Tokenises `data/splits/test.csv` with the bert-base-uncased tokenizer.
- Predicts on the clean test set, writes predictions and metrics.
- Also evaluates on all 6 adversarial test sets if they're present.
- Prints a summary table to stdout.
- Writes `tier4_bert/reports/metrics.json` (the headline metrics blok) and `tier4_bert/reports/errors_for_review.csv` (top FPs and FNs by model confidence, used in the cross-tier error analysis).

Expected runtime: 1-3 minutes on CPU, under 30 seconds on GPU.

## Running Model 2: T3 C04

Confirm the checkpoint is in place:

```bash
ls tier3_gpt2/reports/checkpoints/C04_s123.pt
```

Run the Tier 3 evaluation script pointing at this checkpoint:

```bash
python -m tier3_gpt2.scripts.evaluate \
    --checkpoint tier3_gpt2/reports/checkpoints/C04_s123.pt \
    --no-hf-init
```

The `--no-hf-init` flag skips the fresh GPT-2 download from Hugging Face, since the checkpoint already contains the fully-trained weights (saves about 500 MB of bandwidth).

What the script does:

- Rebuilds GPT-2-small with the flexible_token classification head.
- Loads the trained weights from the checkpoint.
- Tokenises `data/splits/test.csv` with the GPT-2 BPE tokenizer.
- Predicts on the clean test set and the 6 adversarial test sets.
- Prints a summary and writes `tier3_gpt2/reports/metrics.json` plus confusion-matrix PNGs to `tier3_gpt2/reports/figures/`.

Expected runtime: 2-4 minutes on CPU, around 1 minute on GPU.

## Reproducing from scratch (optional)

If for any reason the checkpoints can't be downloaded, both models can be retrained from scratch. This takes longer, but the training scripts are deterministic given the seed so the retrained checkpoints reproduce the headline numbers above to within rounding.

Tier 4 BERT-base (Model 1):

```bash
python -m tier4_bert.scripts.train --cells T4A_bert_lr3e-5
```

Expected runtime: 60-120 seconds on a Colab T4 GPU.

Tier 3 GPT-2 small with the C04 config (Model 2):

```bash
python -m tier3_gpt2.scripts.train --cells C04_s123
```

Expected runtime: 8-12 minutes on a Colab T4 GPU.

## Sanity checks

If you want to confirm the project is healthy before running either model, the unit test suite runs without a GPU:

```bash
pytest tests/ -q
pytest tier1_classical/tests/ -q
pytest tier2_lstm/tests/ -q
pytest tier3_gpt2/tests/ -q
pytest tier4_bert/tests/ -q
```

All tests should pass. If pytest reports failures it's usually a dependency version mismatch, try upgrading from `requirements.txt`.

## Project structure

```
data/                  UCI SMS Spam Collection plus the script that builds
                       train / val / test / adversarial CSVs
shared/                cross-tier infrastructure: splits, metrics, grid runner,
                       checkpointing, comparison
tier1_classical/       TF-IDF / BoW / Word2Vec with NB / LR / SVM
tier2_lstm/            LSTM and BiLSTM with random / GloVe / Word2Vec
tier3_gpt2/            GPT-2 implemented from scratch in PyTorch
                       (adapted from Raschka 2024, ch02-ch06)
tier4_bert/            fine-tuned BERT-base / DistilBERT via Hugging Face
tests/                 unit tests for the shared/ infrastructure
colab/                 Colab notebooks used for grid training
```

Each tier has the same internal layout: `config_grid.py` listing the cells in the grid, a `scripts/` folder with `train.py` and `evaluate.py`, and a `reports/` folder with predictions, metadata, history, and checkpoints.

## Academic integrity

Every file in `tier3_gpt2/gpt2_scratch/` that contains code adapted from Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024, carries a header coment naming source chapter and the adaptation mode (REWRITE for small verifiable layers, ADAPT for the math-critical reshape lines). Everything else is original work.

Pretreined weights for Tier 3 (GPT-2 from OpenAI via Hugging Face) and Tier 4 (BERT-base-uncased and DistilBERT-base-uncased via Hugging Face) are downloaded the first time they're used. Both sources are cited in the report.

## Contact

If anything in these instructions doesn't work, or if a checkpoint appears to be missing from the Drive folder, please get in touch through my City email address.

# ── README.md ─────────────────────────────────────────────────
readme = """# HinglishDB: Generalizing Data Retrieval through a Hinglish-to-SQL Framework

[![Paper](https://img.shields.io/badge/Paper-Springer%20Nature-blue)](https://link.springer.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **HinglishDB** is the first fine-tuning-ready text-to-SQL dataset
> for Hinglish — the Roman-script, Hindi-English code-mixed language
> spoken by ~350 million urban Indian speakers.

## Dataset

| Split | Databases | Pairs |
|-------|-----------|-------|
| Train | 45 | 5,232 |
| Dev   | 5  | 528   |
| Test  | 5  | 381   |
| **Total** | **55** | **6,141** |

Splits are at the **database level** — zero schema overlap between
train, dev, and test sets.

## Key Results

| Training data | BLEU | SQAM | TSED |
|---|---|---|---|
| Hinglish only | 20.39 | 0.319 | 0.308 |
| English only (ck-175) | 22.71 | 0.326 | **0.405** |
| **English + Hindi + Hinglish** | **22.88** | **0.329** | 0.374 |

Full results: [`results/FINAL_ALL_RESULTS.json`](results/FINAL_ALL_RESULTS.json)

## Installation

```bash
git clone https://github.com/[username]/HinglishDB.git
cd HinglishDB
pip install -r requirements.txt
```

## Reproducing Experiments

### EX1 — Fine-tune on Hinglish (CREATE TABLE prompts)
```bash
python src/train.py \\
  --train_files data/hinglish_finetuning_ready/train.json \\
  --dev_file data/hinglish_finetuning_ready/dev.json \\
  --output_dir checkpoints/hinglish_createtable \\
  --max_steps 400
```

### EX2 — English baseline
```bash
python src/train.py \\
  --train_files data/hinglish_finetuning_ready/train.json \\
  --dev_file data/hinglish_finetuning_ready/dev.json \\
  --output_dir checkpoints/english_baseline \\
  --max_steps 400
```

### EX3 — Multilingual (English + Hindi + Hinglish)
```bash
python src/train.py \\
  --train_files path/to/english_train.json,path/to/hindi_train.json,data/hinglish_finetuning_ready/train.json \\
  --dev_file data/hinglish_finetuning_ready/dev.json \\
  --output_dir checkpoints/multilingual \\
  --max_steps 400
```

### Evaluate
```bash
python src/evaluate.py \\
  --predictions results/hinglish_predictions.json \\
  --output results/metrics.json
```

### Translate your own data
```bash
export GROQ_API_KEY=your_key_here
python src/translate.py \\
  --input_dir data/splits \\
  --output_dir data/hinglish_splits
```

## Model Checkpoints

Fine-tuned LoRA adapter weights available at:
[HuggingFace Hub — Alex-24816/HinglishDB](https://huggingface.co/Alex-24816/HinglishDB)

## Citation

If you use HinglishDB in your work, please cite:

```bibtex
@article{Vikram2026hinglishdb,
  author  = {Singh, Vikram and Bhagat, Aniruddha},
  title   = {Generalizing Data Retrieval through a Hinglish-to-SQL Framework},
  journal = {Language Resources and Evaluation},
  publisher = {Springer Nature},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

## Acknowledgements

Built on the [BIRD benchmark](https://bird-bench.github.io/).
Translation via [Llama-3.1](https://groq.com) (Groq API).
Fine-tuning with [QLoRA](https://github.com/artidoro/qlora).
"""

with open(f'{repo_dir}/README.md', 'w') as f:
    f.write(readme)
print("✅ README.md written")

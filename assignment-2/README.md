# Assignment 2: Sentiment Classification with Neural Language Models

## Model summary

**Approach:** Fine-tuned `distilroberta-base` (pretrained Transformer encoder, 6
layers, 768 hidden size, ~82M parameters) with a sequence classification head,
using class-weighted cross-entropy loss to handle the 180/60 training imbalance.

The notebook contains two fine-tuning approaches, run back to back:

1. **Whole-training-set fine-tuning** — trains on all 240 examples for 5 epochs.
   Achieved **~69% accuracy** on the public test set but overfit to the majority
   (positive) class.
2. **K-fold cross-validation with early stopping** — 5-fold stratified CV, up to
   10 epochs per fold, early stopping with patience 2 on validation loss. Achieved
   **~78% accuracy** on the public test set with less overfitting and better
   generalization.

Both approaches save their resulting model to `model_checkpoint/`, evaluate on
`public_test.csv`, and write `public_test_predictions.csv`. Running the notebook
top to bottom means the second (K-fold) run's checkpoint and predictions are what
end up on disk, since it executes last and overwrites the first.

## Handling the small, imbalanced training set

- **Class imbalance (180 positive / 60 negative):** addressed with
  `sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", ...)`,
  fed into `nn.CrossEntropyLoss(weight=class_weights_tensor)` so errors on the
  minority (negative) class are penalized more heavily.
- **Small dataset / overfitting:** addressed by using a pretrained encoder
  (rather than training from scratch) so the model starts with strong general
  language features, plus 5-fold stratified cross-validation with early stopping
  (patience 2 on validation loss) to avoid memorizing the training set.

## Key training techniques

- **Optimizer:** AdamW, chosen for decoupled weight decay — important
  regularization given only 240 training examples.
- **Learning rate:** `1e-5` (reduced from an initial `2e-5`, alongside reducing
  batch size, to fit available CPU memory and improve generalization).
- **Batch size:** `4` (reduced from an initial `16` after running into CPU RAM
  limits).
- **Weight decay:** `0.01`.
- **Scheduler:** linear warmup schedule (`get_linear_schedule_with_warmup`,
  0 warmup steps) for the single full-training-set run; the K-fold run does not
  use a scheduler.
- **Epochs:** 5 for the full-training-set run; up to 10 per fold for the K-fold
  run, cut short by early stopping.
- **Gradient clipping:** max norm 1.0 (full-training-set run).
- **Seed:** 42, set for `torch` and `numpy` for reproducibility.

## Evaluation

Accuracy, a classification report, and a confusion matrix are printed/plotted
for each approach's run against `public_test.csv`:

- Approach 1 (whole training set): ~69% accuracy, overfit toward positive class.
- Approach 2 (K-fold + early stopping): ~78% accuracy, better balanced.

See the notebook's Model Structure markdown cell for the embedded confusion
matrix images from both runs.

## Repository contents

```
stage1_notebook.ipynb      # full pipeline: both fine-tuning approaches, evaluation, prediction
README.md
requirements.txt
model_checkpoint/          # saved distilroberta-base model + tokenizer (from the K-fold run)
public_test_predictions.csv
```

## Setup

1. Create a virtual environment (Python 3.10+ recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `train.csv` and `public_test.csv` in this same directory.

## How to run

```bash
jupyter notebook stage1_notebook.ipynb
```

Run all cells top to bottom. This will:
- Load and inspect `train.csv` / `public_test.csv`
- Fine-tune `distilroberta-base` on the full training set (approach 1)
- Fine-tune `distilroberta-base` with 5-fold CV and early stopping (approach 2)
- Evaluate each approach on `public_test.csv` (accuracy, classification report,
  confusion matrix)
- Save the final model and tokenizer to `model_checkpoint/`
- Write predictions to `public_test_predictions.csv`

**Note:** CPU fine-tuning of a transformer, even distilled, is slow relative to
the lightweight-classifier alternative — the K-fold run in particular can take a
while on a laptop CPU. 

### Reload the saved checkpoint for inference only (no retraining)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("model_checkpoint")
model = AutoModelForSequenceClassification.from_pretrained("model_checkpoint")
model.eval()

def predict(texts, max_len=256):
    enc = tokenizer(texts, truncation=True, padding="max_length",
                     max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.argmax(logits, dim=1).tolist()
```

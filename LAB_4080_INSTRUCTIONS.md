# style_gen 4080 Run Instructions

Use these steps on a lab machine with an RTX 4080.

## 1. Open the project

```bash
cd /path/to/style_gen
source style/bin/activate
```

If the virtualenv does not exist on that machine, create it first and install deps:

```bash
python -m venv style
source style/bin/activate
pip install -r requirements.txt
```

## 2. Train on one 4080

Start with this command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./style/bin/python training/train.py \
  --device cuda:0 \
  --epochs 10 \
  --batch-size 4 \
  --val-batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 4
```

If memory is clearly fine, try a larger batch:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./style/bin/python training/train.py \
  --device cuda:0 \
  --epochs 10 \
  --batch-size 8 \
  --val-batch-size 8 \
  --grad-accum-steps 1 \
  --num-workers 4
```

Notes:

- Current code uses a single GPU.
- `grad-accum-steps 1` is the normal lab setting.
- Gradient checkpointing and attention slicing are still enabled in code. They mainly save memory and do not meaningfully reduce model quality.

## 3. Check that training produced checkpoints

Expected files:

```text
training/checkpoints/style_gen_best.pt
training/checkpoints/style_gen_final.pt
```

## 4. Evaluate the checkpoint

```bash
./style/bin/python evaluation/evaluate.py \
  --checkpoint training/checkpoints/style_gen_best.pt \
  --split validation \
  --batch-size 4 \
  --num-batches 10 \
  --sampler ddim \
  --steps 50
```

This should create:

```text
outputs/eval_preview.png
outputs/eval_metrics.json
```

## 5. Test your own handwriting sample in the UI

Start the UI:

```bash
bash run_ui.sh
```

Open:

```text
http://127.0.0.1:7860
```

In the UI:

- checkpoint path: `training/checkpoints/style_gen_best.pt`
- upload a handwriting sample image
- type the text to render
- click `Generate`

## 6. Fast CLI inference instead of UI

```bash
./style/bin/python inference.py \
  --checkpoint training/checkpoints/style_gen_best.pt \
  --style-image path/to/your_sample.png \
  --text "Hello world" "This is my handwriting style" \
  --output outputs/test.png
```

## 7. What to submit

At minimum keep these:

- `training/checkpoints/style_gen_best.pt`
- `outputs/eval_preview.png`
- `outputs/eval_metrics.json`
- one or more generated images from the UI or CLI

## 8. If training fails

First retry with smaller batch:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./style/bin/python training/train.py \
  --device cuda:0 \
  --epochs 10 \
  --batch-size 2 \
  --val-batch-size 2 \
  --grad-accum-steps 1 \
  --num-workers 4
```

If that still fails, reduce to batch size 1 and increase gradient accumulation:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./style/bin/python training/train.py \
  --device cuda:0 \
  --epochs 10 \
  --batch-size 1 \
  --val-batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 2
```

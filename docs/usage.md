# Usage

Install in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
bash scripts/test.sh
```

Run real-data training/evaluation/inference:

```bash
MAX_STEPS=1 bash scripts/train.sh
CHECKPOINT=checkpoints/reddit/hdt_multi_objective_1steps.pt MAX_BATCHES=1 bash scripts/eval.sh
CHECKPOINT=checkpoints/reddit/hdt_multi_objective_1steps.pt bash scripts/infer.sh
```

For a full Reddit run, omit `MAX_STEPS` and `MAX_BATCHES`:

```bash
bash scripts/train.sh
CHECKPOINT=checkpoints/reddit/hdt_multi_objective_<N>steps.pt bash scripts/eval.sh
```

Direct Python entry points:

```bash
PYTHONPATH=src python -m hdt.training.train --config configs/experiments/reddit_main.yaml
PYTHONPATH=src python -m hdt.evaluation.evaluate --config configs/eval.yaml --checkpoint checkpoints/reddit/hdt_multi_objective_1steps.pt --max-batches 1
PYTHONPATH=src python -m hdt.inference.infer --config configs/train.yaml --checkpoint checkpoints/reddit/hdt_multi_objective_1steps.pt --input "1,2,3"
```

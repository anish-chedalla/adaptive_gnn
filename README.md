# Astra-Stim

Simulate quantum error correction codes with realistic biased noise. This project builds CSS stabilizer codes and samples measurement syndromes with Z-heavy noise (where Z errors are more likely than X errors).

## Setup

```bash
pip install stim numpy
```

## Try It Out

**Test that everything works:**
```bash
python test_imports.py
```

**Sample syndromes with no noise:**
```bash
python sample_syndromes.py --d 6 --noise none --shots 10
```

**Add realistic noise (1% error rate, 20× bias towards Z):**
```bash
python sample_syndromes.py --d 6 --noise biased_circuit --p 0.01 --eta 20 --shots 10
```

**Only apply noise to data qubits:**
```bash
python sample_syndromes.py --d 6 --noise biased_data --p 0.01 --eta 20 --shots 10
```

**Add time-varying noise (error rate drifts over time):**
```bash
python sample_syndromes.py --d 6 --noise biased_circuit --p 0.01 --eta 20 \
  --drift sine --drift_target p --drift_amp 0.3 --drift_period_ticks 50 --shots 10
```

**Save results to a file:**
```bash
python sample_syndromes.py --d 6 --noise biased_circuit --p 0.01 --eta 20 --shots 100 --out results.npz
```

## Parameters

- `--d`: Code distance (6, 10, 12, 18, 24, or 34)
- `--rounds`: Number of measurement rounds (default: same as d)
- `--basis`: Measure data qubits in x or z basis (default: z)
- `--noise`: Noise type (none, biased_data, or biased_circuit)
- `--p`: Physical error rate (0 to 1)
- `--eta`: Bias ratio pZ/pX (default: 20)
- `--shots`: Number of syndrome samples to collect
- `--print_n`: Print first N samples (default: 10)
- `--out`: Save results to .npz file (optional)
- `--drift`: Drift mode (none or sine)
- `--drift_target`: Which parameter drifts (p or eta)
- `--drift_amp`: Drift amplitude (e.g., 0.3 for ±30%)
- `--drift_period_ticks`: Period of drift in measurement rounds


## Anish Runs: 


```bash
# INITIALIZATION
python --version

python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

pip install -e .
python -m tests.test_imports


```bash 
 
Generate code-capacity datasets (static + multiple drift models)

mkdir data -ErrorAction SilentlyContinue

# Static (no drift)
python -m gnn_pipeline.generate_codecap `
  --code 72_12_6 `
  --p 0.03 `
  --eta 20 `
  --shots 50000 `
  --drift_model none `
  --seed 42 `
  --out .\data\codecap_static_72_p003.npz

# Sine drift
python -m gnn_pipeline.generate_codecap `
  --code 72_12_6 `
  --p 0.03 `
  --eta 20 `
  --shots 50000 `
  --drift_model sine `
  --drift_amp 0.5 `
  --drift_period 1000 `
  --seed 42 `
  --out .\data\codecap_drift_sine_72.npz

# Ornstein–Uhlenbeck drift
python -m gnn_pipeline.generate_codecap `
  --code 72_12_6 `
  --p 0.03 `
  --eta 20 `
  --shots 50000 `
  --drift_model ou `
  --drift_amp 0.5 `
  --drift_period 1000 `
  --ou_theta 0.05 `
  --ou_sigma 0.02 `
  --seed 42 `
  --out .\data\codecap_drift_ou_72.npz

# Random telegraph noise drift
python -m gnn_pipeline.generate_codecap `
  --code 72_12_6 `
  --p 0.03 `
  --eta 20 `
  --shots 50000 `
  --drift_model rtn `
  --drift_amp 0.5 `
  --drift_period 1000 `
  --rtn_delta 0.01 `
  --rtn_switch 0.001 `
  --seed 42 `
  --out .\data\codecap_drift_rtn_72.npz

3) Build PyG datasets (selfsup + supervised)
# Self-supervised dataset
python -m gnn_pipeline.build_dataset `
  --in_glob ".\data\codecap_drift_sine_72.npz" `
  --mode selfsup `
  --W 4 `
  --out ".\data\graph_selfsup_sine_W4.pt"

# Supervised dataset
python -m gnn_pipeline.build_dataset `
  --in_glob ".\data\codecap_drift_sine_72.npz" `
  --mode supervised `
  --W 4 `
  --out ".\data\graph_supervised_sine_W4.pt"

# Verify it loads and has splits
python -c "import torch; obj=torch.load(r'.\data\graph_selfsup_sine_W4.pt', weights_only=False); print(obj.keys()); print('train',len(obj['train']),'val',len(obj['val']),'test',len(obj['test']))"

4) Train self-supervised (checkpoint must appear)
mkdir runs -ErrorAction SilentlyContinue

python -m gnn_pipeline.train_selfsupervised `
  --in_glob ".\data\codecap_drift_sine_72.npz" `
  --W 4 `
  --epochs 20 `
  --batch_size 128 `
  --out_dir ".\runs\selfsup_sine_W4_gpu"

dir .\runs\selfsup_sine_W4_gpu
# Must include: best_model.pt

5) Train supervised (uses BP-in-the-loop) with correct pretrained path
python -m gnn_pipeline.train_supervised `
  --in_glob ".\data\codecap_drift_sine_72.npz" `
  --W 4 `
  --epochs 10 `
  --batch_size 64 `
  --out_dir ".\runs\sup_sine_W4_gpu" `
  --pretrained ".\runs\selfsup_sine_W4_gpu\best_model.pt"

6) Evaluate: BP vs GNN-BP on multiple test sets (static + drift variants)
# Static test
python -m gnn_pipeline.evaluate `
  --test_npz ".\data\codecap_static_72_p003.npz" `
  --out_dir ".\runs\eval_static_bp"

python -m gnn_pipeline.evaluate `
  --test_npz ".\data\codecap_static_72_p003.npz" `
  --gnn_model ".\runs\sup_sine_W4_gpu\best_model.pt" `
  --out_dir ".\runs\eval_static_gnn"

# Sine drift test
python -m gnn_pipeline.evaluate `
  --test_npz ".\data\codecap_drift_sine_72.npz" `
  --out_dir ".\runs\eval_sine_bp"

python -m gnn_pipeline.evaluate `
  --test_npz ".\data\codecap_drift_sine_72.npz" `
  --gnn_model ".\runs\sup_sine_W4_gpu\best_model.pt" `
  --out_dir ".\runs\eval_sine_gnn"

# OU drift test (generalization)
python -m gnn_pipeline.evaluate `
  --test_npz ".\data\codecap_drift_ou_72.npz" `
  --gnn_model ".\runs\sup_sine_W4_gpu\best_model.pt" `
  --out_dir ".\runs\eval_ou_gnn"

# RTN drift test (generalization)
python -m gnn_pipeline.evaluate `
  --test_npz ".\data\codecap_drift_rtn_72.npz" `
  --gnn_model ".\runs\sup_sine_W4_gpu\best_model.pt" `
  --out_dir ".\runs\eval_rtn_gnn"


What “success” looks like (minimum):

All tests pass.

Training produces best_model.pt.

Evaluation produces JSON outputs and does not crash.

GNN-BP doesn’t tank convergence vs BP.

Ideally: GNN-BP improves LER on drift, and doesn’t overfit only to sine.
```
## Full Pipeline Smoke Test (PowerShell)

Run the following from the **repo root**. This is an **intensive** end-to-end check (data generation → dataset → training → sweeps/ablations).

> Notes:
> - These commands assume `python -m ...` entrypoints exist exactly as shown.
> - Large shot counts (50k) are deliberate for research-grade signal; reduce if you only want a quick sanity check.
> - `threshold_sweep` / `ablation` require your `gnn_pipeline.evaluate` API to be consistent.

---


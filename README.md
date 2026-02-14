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


## Full Pipeline Smoke Test (PowerShell)

Run the following from the **repo root**. This is an **intensive** end-to-end check (data generation → dataset → training → sweeps/ablations).

> Notes:
> - These commands assume `python -m ...` entrypoints exist exactly as shown.
> - Large shot counts (50k) are deliberate for research-grade signal; reduce if you only want a quick sanity check.
> - `threshold_sweep` / `ablation` require your `gnn_pipeline.evaluate` API to be consistent.

---

### 0) Environment sanity

```powershell
python --version
python -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
pip install -e .
python -m tests.test_imports

# If pytest is used in this repo:
pytest -q
mkdir data -ErrorAction SilentlyContinue

# Biased circuit-level (faults on gates/idle/meas/reset)
python -m astra_stim.sample_syndromes `
  --d 6 --rounds 10 --noise biased_circuit --p 0.01 --eta 20 `
  --shots 2000 --seed 42 --out .\data\biased_circuit_d6_r10.npz

# Drifting circuit-level (sine schedule p(t), eta(t))
python -m astra_stim.sample_syndromes `
  --d 6 --rounds 10 --noise drifting_biased_circuit --p 0.01 --eta 20 `
  --drift_p 0.002 --drift_eta 2.0 --period 100 `
  --shots 2000 --seed 42 --out .\data\drift_circuit_d6_r10.npz
# Static biased code-capacity
python -m gnn_pipeline.generate_codecap `
  --d 6 --shots 50000 --p 0.03 --eta 20 --mode static `
  --out .\data\codecap_static_p003.npz

# Drifting (sine) biased code-capacity
python -m gnn_pipeline.generate_codecap `
  --d 6 --shots 50000 --p 0.03 --eta 20 --mode drifting `
  --drift sine --drift_amp 0.5 --drift_period 1000 `
  --out .\data\codecap_drift_sine.npz

# Drifting (OU) biased code-capacity
python -m gnn_pipeline.generate_codecap `
  --d 6 --shots 50000 --p 0.03 --eta 20 --mode drifting `
  --drift ou --drift_amp 0.5 --drift_period 1000 `
  --out .\data\codecap_drift_ou.npz

# Drifting (RTN) biased code-capacity
python -m gnn_pipeline.generate_codecap `
  --d 6 --shots 50000 --p 0.03 --eta 20 --mode drifting `
  --drift rtn --drift_amp 0.5 --drift_period 1000 `
  --out .\data\codecap_drift_rtn.npz
# Self-supervised dataset (window W)
python -m gnn_pipeline.build_dataset `
  --in_glob ".\data\codecap_drift_sine.npz" --mode selfsup --W 4 `
  --out ".\data\graph_selfsup_sine_W4.pt"

# Supervised dataset (requires x_errors/z_errors in NPZ; provided by generate_codecap)
python -m gnn_pipeline.build_dataset `
  --in_glob ".\data\codecap_drift_sine.npz" --mode supervised --W 4 `
  --out ".\data\graph_supervised_sine_W4.pt"

# Quick integrity check
python -c "import torch; obj=torch.load(r'.\data\graph_selfsup_sine_W4.pt', weights_only=False); print(obj.keys()); print('train:', len(obj['train']), 'val:', len(obj['val']), 'test:', len(obj['test']))"
mkdir runs -ErrorAction SilentlyContinue

# Self-supervised pretraining
python -m gnn_pipeline.train_selfsupervised `
  --in_glob ".\data\codecap_drift_sine.npz" --W 4 `
  --epochs 20 --batch_size 64 `
  --out_dir ".\runs\selfsup_sine_W4"

# Supervised fine-tuning (BP-in-the-loop)
python -m gnn_pipeline.train_supervised `
  --in_glob ".\data\codecap_drift_sine.npz" --W 4 `
  --epochs 10 --batch_size 32 `
  --out_dir ".\runs\sup_sine_W4" `
  --pretrained ".\runs\selfsup_sine_W4\model_best.pt"
# Threshold sweep (LER vs p curve)
python -m gnn_pipeline.threshold_sweep `
  --code bb_d6 --noise codecap --eta 20 `
  --ps 0.01 0.02 0.03 0.04 0.05 `
  --shots 50000 `
  --model ".\runs\sup_sine_W4\model_best.pt" `
  --out ".\runs\sweep_codecap.csv"

# Ablation
python -m gnn_pipeline.ablation `
  --data ".\data\graph_supervised_sine_W4.pt" `
  --model ".\runs\sup_sine_W4\model_best.pt" `
  --out_dir ".\runs\ablations"




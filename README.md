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
# =========================
# 0) Enter repo root + sanity
# =========================
cd "C:\path\to\QLDPC_Pipeline_2"   # <-- CHANGE THIS

python --version
git status

# GPU visibility (must show a GPU)
nvidia-smi

# =========================
# 1) Clean venv
# =========================
if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools

# =========================
# 2) Install PyTorch WITH CUDA
# Pick ONE of these. Prefer cu121 unless driver is ancient.
# =========================

# Option A: CUDA 12.1 wheels (recommended)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
```bash 
 
1️⃣ Generate Static Code-Capacity Data


mkdir data -ErrorAction SilentlyContinue

python -m gnn_pipeline.generate_codecap `
  --code 72_12_6 `
  --p 0.03 `
  --eta 20 `
  --shots 20000 `
  --drift_model none `
  --seed 42 `
  --out .\data\codecap_static_72_p003.npz

2️⃣ Generate Drifting (Sine) Code-Capacity

python -m gnn_pipeline.generate_codecap `
  --code 72_12_6 `
  --p 0.03 `
  --eta 20 `
  --shots 20000 `
  --drift_model sine `
  --drift_amp 0.5 `
  --drift_period 1000 `
  --seed 42 `
  --out .\data\codecap_drift_sine_72.npz

3️⃣ Generate OU Drift

python -m gnn_pipeline.generate_codecap `
  --code 72_12_6 `
  --p 0.03 `
  --eta 20 `
  --shots 20000 `
  --drift_model ou `
  --drift_amp 0.5 `
  --drift_period 1000 `
  --ou_theta 0.01 `
  --ou_sigma 0.005 `
  --seed 42 `
  --out .\data\codecap_drift_ou_72.npz

4️⃣ Generate Random Telegraph Noise (RTN)

python -m gnn_pipeline.generate_codecap `
  --code 72_12_6 `
  --p 0.03 `
  --eta 20 `
  --shots 20000 `
  --drift_model rtn `
  --drift_amp 0.5 `
  --drift_period 1000 `
  --rtn_delta 0.01 `
  --rtn_switch 0.001 `
  --seed 42 `
  --out .\data\codecap_drift_rtn_72.npz


Build Dataset

python -m gnn_pipeline.build_dataset `
  --in_glob ".\data\codecap_drift_sine_72.npz" `
  --mode selfsup `
  --W 4 `
  --out ".\data\graph_selfsup_sine_W4.pt"


Supervised:

python -m gnn_pipeline.build_dataset `
  --in_glob ".\data\codecap_drift_sine_72.npz" `
  --mode supervised `
  --W 4 `
  --out ".\data\graph_supervised_sine_W4.pt"

Train (GPU)

mkdir runs -ErrorAction SilentlyContinue
python -m gnn_pipeline.train_selfsupervised `
  --in_glob ".\data\codecap_drift_sine_72.npz" `
  --W 4 `
  --epochs 5 `
  --batch_size 128 `
  --out_dir ".\runs\selfsup_gpu"


Then:


python -m gnn_pipeline.train_supervised `
  --in_glob ".\data\codecap_drift_sine_72.npz" `
  --W 4 `
  --epochs 3 `
  --batch_size 64 `
  --out_dir ".\runs\sup_gpu" `
  --pretrained ".\runs\selfsup_gpu\model_best.pt"



Evaluate


Baseline:


python -m gnn_pipeline.evaluate `
  --test_npz ".\data\codecap_drift_sine_72.npz" `
  --out_dir ".\runs\eval_bp"

GNN-assisted:

python -m gnn_pipeline.evaluate `
  --test_npz ".\data\codecap_drift_sine_72.npz" `
  --gnn_model ".\runs\sup_gpu\model_best.pt" `
  --out_dir ".\runs\eval_gnn"

```
## Full Pipeline Smoke Test (PowerShell)

Run the following from the **repo root**. This is an **intensive** end-to-end check (data generation → dataset → training → sweeps/ablations).

> Notes:
> - These commands assume `python -m ...` entrypoints exist exactly as shown.
> - Large shot counts (50k) are deliberate for research-grade signal; reduce if you only want a quick sanity check.
> - `threshold_sweep` / `ablation` require your `gnn_pipeline.evaluate` API to be consistent.

---


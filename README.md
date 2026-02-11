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

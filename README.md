# Astra-Stim

Simulate quantum error correction codes with realistic biased noise. This project builds CSS stabilizer codes and samples measurement syndromes with Z-heavy noise (where Z errors are more likely than X errors).

## Setup

```bash
pip install numpy scipy torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
pip install ldpc pymatching stim
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"
bash run_full_pipeline.sh
```




> Notes:
> - These commands assume `python -m ...` entrypoints exist exactly as shown.
> - Large shot counts (50k) are deliberate for research-grade signal; reduce if you only want a quick sanity check.
> - `threshold_sweep` / `ablation` require your `gnn_pipeline.evaluate` API to be consistent.

---

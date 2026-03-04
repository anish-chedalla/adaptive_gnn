# Astra-Stim

Simulate quantum error correction codes with realistic biased noise. This project builds CSS stabilizer codes and samples measurement syndromes with Z-heavy noise (where Z errors are more likely than X errors).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
pip install ldpc pymatching stim
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"

```
## FULL RUN FOR PROF.RAVEENDRAN

```bash
echo "phase 1 - generating test data only (training uses online regeneration)"
python -m gnn_pipeline.generate_codecap --code 72_12_6 --p 0.05 --eta 20 --shots 10000 --out data/big72_test_p05.npz
python -m gnn_pipeline.generate_codecap --code 144_12_12 --p 0.05 --eta 20 --shots 10000 --out data/big144_test_p05.npz
python -m gnn_pipeline.generate_codecap --code 288_12_18 --p 0.05 --eta 20 --shots 10000 --out data/big288_test_p05.npz

echo "phase 2 - training base models with online data regeneration"
echo "fresh random data every epoch, p sampled uniformly from 0.01 to 0.10"
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big72_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --hidden_dim 64 --num_mp_layers 3 --bp_iters 10 --loss focal --neural_bp --use_film --correction_mode additive --epochs 15 --batch_size 16 --lr 1e-4 --scheduler cosine --amp --out_dir runs/big72_film
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big144_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --hidden_dim 64 --num_mp_layers 3 --bp_iters 10 --loss focal --neural_bp --use_film --correction_mode additive --epochs 15 --batch_size 16 --lr 1e-4 --scheduler cosine --amp --out_dir runs/big144_film
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big288_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --hidden_dim 64 --num_mp_layers 3 --bp_iters 10 --loss focal --neural_bp --use_film --correction_mode additive --epochs 15 --batch_size 16 --lr 1e-4 --scheduler cosine --amp --out_dir runs/big288_film

echo "phase 3 - full decoder comparison on all 3 codes"
echo "runs BP, GNN-BP, BP-OSD, GNN+BP-OSD, BP-LSD, GNN+BP-LSD, BeliefFind, MWPM"
python -m gnn_pipeline.evaluate --test_npz data/big72_test_p05.npz --gnn_model runs/big72_film/best_model.pt --bposd --bplsd --belieffind --mwpm --use_syndrome_p --out_dir runs/eval_big72
python -m gnn_pipeline.evaluate --test_npz data/big144_test_p05.npz --gnn_model runs/big144_film/best_model.pt --bposd --bplsd --belieffind --mwpm --use_syndrome_p --out_dir runs/eval_big144
python -m gnn_pipeline.evaluate --test_npz data/big288_test_p05.npz --gnn_model runs/big288_film/best_model.pt --bposd --bplsd --belieffind --mwpm --use_syndrome_p --out_dir runs/eval_big288

echo "phase 4 - threshold sweep p=0.005 to 0.10 with sine, OU, and RTN drift"
python -m gnn_pipeline.threshold_sweep --code 72_12_6 --p_min 0.005 --p_max 0.10 --num_points 15 --shots 10000 --eta 20 --drift_models sine,ou,rtn --drift_amp 0.015 --gnn_model runs/big72_film/best_model.pt --bposd --bplsd --mwpm --out_dir runs/sweep_big72
python -m gnn_pipeline.threshold_sweep --code 144_12_12 --p_min 0.005 --p_max 0.10 --num_points 15 --shots 10000 --eta 20 --drift_models sine,ou,rtn --drift_amp 0.015 --gnn_model runs/big144_film/best_model.pt --bposd --bplsd --mwpm --out_dir runs/sweep_big144
python -m gnn_pipeline.threshold_sweep --code 288_12_18 --p_min 0.005 --p_max 0.10 --num_points 15 --shots 10000 --eta 20 --drift_models sine,ou,rtn --drift_amp 0.015 --gnn_model runs/big288_film/best_model.pt --bposd --bplsd --mwpm --out_dir runs/sweep_big288

echo "phase 5 - transfer learning from 72 qubit model to 144 and 288"
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big144_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --pretrained runs/big72_film/best_model.pt --freeze_backbone --hidden_dim 64 --num_mp_layers 3 --bp_iters 10 --loss focal --neural_bp --use_film --correction_mode additive --epochs 10 --batch_size 16 --lr 5e-4 --scheduler cosine --amp --out_dir runs/transfer_72to144_frozen
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big144_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --pretrained runs/big72_film/best_model.pt --hidden_dim 64 --num_mp_layers 3 --bp_iters 10 --loss focal --neural_bp --use_film --correction_mode additive --epochs 10 --batch_size 16 --lr 1e-5 --scheduler cosine --amp --out_dir runs/transfer_72to144_finetune
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big288_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --pretrained runs/big72_film/best_model.pt --freeze_backbone --hidden_dim 64 --num_mp_layers 3 --bp_iters 10 --loss focal --neural_bp --use_film --correction_mode additive --epochs 10 --batch_size 16 --lr 5e-4 --scheduler cosine --amp --out_dir runs/transfer_72to288_frozen
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big288_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --pretrained runs/big72_film/best_model.pt --hidden_dim 64 --num_mp_layers 3 --bp_iters 10 --loss focal --neural_bp --use_film --correction_mode additive --epochs 10 --batch_size 16 --lr 1e-5 --scheduler cosine --amp --out_dir runs/transfer_72to288_finetune
python -m gnn_pipeline.evaluate --test_npz data/big144_test_p05.npz --gnn_model runs/transfer_72to144_frozen/best_model.pt --bposd --bplsd --out_dir runs/eval_transfer_144_frozen
python -m gnn_pipeline.evaluate --test_npz data/big144_test_p05.npz --gnn_model runs/transfer_72to144_finetune/best_model.pt --bposd --bplsd --out_dir runs/eval_transfer_144_finetune
python -m gnn_pipeline.evaluate --test_npz data/big288_test_p05.npz --gnn_model runs/transfer_72to288_frozen/best_model.pt --bposd --bplsd --out_dir runs/eval_transfer_288_frozen
python -m gnn_pipeline.evaluate --test_npz data/big288_test_p05.npz --gnn_model runs/transfer_72to288_finetune/best_model.pt --bposd --bplsd --out_dir runs/eval_transfer_288_finetune

echo "phase 6 - component ablation on 288 qubit code"
python -m gnn_pipeline.ablation --grid_json configs/component_ablation.json --train_data "data/big288_test_p05.npz" --test_data data/big288_test_p05.npz --bposd --bplsd --epochs 15 --out_dir runs/ablation_components

echo "phase 7 - interleaved GNN-BP on all 3 codes"
echo "GNN corrects mid-BP using intermediate beliefs as 5th feature"
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big72_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --hidden_dim 64 --num_mp_layers 3 --bp_iters 20 --loss focal --neural_bp --use_film --correction_mode additive --node_feat_dim 5 --interleaved_train --stage1_iters 10 --stage2_iters 10 --epochs 15 --batch_size 16 --lr 1e-4 --scheduler cosine --amp --out_dir runs/big72_interleaved
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big144_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --hidden_dim 64 --num_mp_layers 3 --bp_iters 20 --loss focal --neural_bp --use_film --correction_mode additive --node_feat_dim 5 --interleaved_train --stage1_iters 10 --stage2_iters 10 --epochs 15 --batch_size 16 --lr 1e-4 --scheduler cosine --amp --out_dir runs/big144_interleaved
python -m gnn_pipeline.train_unified --mode code_capacity --online_regen --code_npz data/big288_test_p05.npz --p_range 0.01 0.10 --p_base 0.05 --eta 20 --samples_per_epoch 10000 --hidden_dim 64 --num_mp_layers 3 --bp_iters 20 --loss focal --neural_bp --use_film --correction_mode additive --node_feat_dim 5 --interleaved_train --stage1_iters 10 --stage2_iters 10 --epochs 15 --batch_size 16 --lr 1e-4 --scheduler cosine --amp --out_dir runs/big288_interleaved
python -m gnn_pipeline.evaluate --test_npz data/big72_test_p05.npz --gnn_model runs/big72_interleaved/best_model.pt --bposd --bplsd --use_syndrome_p --out_dir runs/eval_big72_interleaved
python -m gnn_pipeline.evaluate --test_npz data/big144_test_p05.npz --gnn_model runs/big144_interleaved/best_model.pt --bposd --bplsd --use_syndrome_p --out_dir runs/eval_big144_interleaved
python -m gnn_pipeline.evaluate --test_npz data/big288_test_p05.npz --gnn_model runs/big288_interleaved/best_model.pt --bposd --bplsd --use_syndrome_p --out_dir runs/eval_big288_interleaved

echo "phase 8 - generating all figures"
python -m gnn_pipeline.make_figures --results_dir runs/eval_big72 --out_dir figures/big72
python -m gnn_pipeline.make_figures --results_dir runs/eval_big144 --out_dir figures/big144
python -m gnn_pipeline.make_figures --results_dir runs/eval_big288 --out_dir figures/big288
python -m gnn_pipeline.make_figures --results_dir runs/eval_big72_interleaved --out_dir figures/big72_interleaved
python -m gnn_pipeline.make_figures --results_dir runs/eval_big144_interleaved --out_dir figures/big144_interleaved
python -m gnn_pipeline.make_figures --results_dir runs/eval_big288_interleaved --out_dir figures/big288_interleaved

echo "all done - results in runs/ and figures/"


```



> Notes:
> - These commands assume `python -m ...` entrypoints exist exactly as shown.
> - Large shot counts (50k) are deliberate for research-grade signal; reduce if you only want a quick sanity check.
> - `threshold_sweep` / `ablation` require your `gnn_pipeline.evaluate` API to be consistent.

---

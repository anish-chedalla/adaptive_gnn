#!/usr/bin/env bash
# =============================================================================
#  QLDPC GNN-BP Full Pipeline — NVIDIA GPU (Dell XPS)
#  Generates all data, trains models, evaluates, and produces figures.
#  Estimated runtime: ~6-10 hours on RTX 3060/4060 class GPU
# =============================================================================
set -euo pipefail

# =============================================================================
#  SETUP — Install dependencies (uncomment on first run)
# =============================================================================
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
# pip install -r requirements.txt
# pip install -e .

# Confirm GPU is visible
echo "=== GPU check ==="
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU FOUND'; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

# Verify critical dependencies
python -c "import ldpc; import pymatching; import stim; import torch_geometric; print('All dependencies OK')"

# Create output dirs
mkdir -p data runs configs figures

# =============================================================================
#  PHASE 1 — Generate test data (training uses online regeneration via FiLM)
#  One test set per code at p=0.05 for decoder comparison,
#  plus additional p-values for static-noise multi-point evaluation.
# =============================================================================
echo ""
echo "=========================================="
echo " PHASE 1: Generating test data"
echo "=========================================="

python -m gnn_pipeline.generate_codecap --code 72_12_6   --p 0.05 --eta 20 --shots 10000 --out data/big72_test_p05.npz
python -m gnn_pipeline.generate_codecap --code 144_12_12 --p 0.05 --eta 20 --shots 10000 --out data/big144_test_p05.npz
python -m gnn_pipeline.generate_codecap --code 288_12_18 --p 0.05 --eta 20 --shots 10000 --out data/big288_test_p05.npz

echo "Phase 1 done: 3 test datasets generated."

# =============================================================================
#  PHASE 2 — Train base FiLM models (one per code size)
#  Online regeneration: fresh data every epoch, p ~ Uniform[0.01, 0.10]
#  FiLM adapts to noise rate via syndrome weight — single model handles all p.
# =============================================================================
echo ""
echo "=========================================="
echo " PHASE 2: Training base FiLM models"
echo "=========================================="

python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big72_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 10 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive \
    --loss focal --syn_weight 0.1 \
    --epochs 15 --batch_size 16 --lr 1e-4 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 2 \
    --amp --patience 10 \
    --out_dir runs/big72_film

python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big144_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 10 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive \
    --loss focal --syn_weight 0.1 \
    --epochs 15 --batch_size 16 --lr 1e-4 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 2 \
    --amp --patience 10 \
    --out_dir runs/big144_film

python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big288_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 10 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive \
    --loss focal --syn_weight 0.1 \
    --epochs 15 --batch_size 16 --lr 1e-4 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 2 \
    --amp --patience 10 \
    --out_dir runs/big288_film

echo "Phase 2 done: 3 base FiLM models trained."

# =============================================================================
#  PHASE 3 — Full decoder comparison on all 3 codes
#  Runs: Plain BP, GNN-BP, BP-OSD, GNN+BP-OSD, BP-LSD, GNN+BP-LSD,
#        BeliefFind, MWPM (9 configs per code)
#  --use_syndrome_p lets FiLM estimate p from syndrome weight at eval time.
# =============================================================================
echo ""
echo "=========================================="
echo " PHASE 3: Full decoder comparison (9 decoders x 3 codes)"
echo "=========================================="

python -m gnn_pipeline.evaluate \
    --test_npz data/big72_test_p05.npz \
    --gnn_model runs/big72_film/best_model.pt \
    --bposd --bplsd --belieffind --mwpm \
    --use_syndrome_p \
    --out_dir runs/eval_big72

python -m gnn_pipeline.evaluate \
    --test_npz data/big144_test_p05.npz \
    --gnn_model runs/big144_film/best_model.pt \
    --bposd --bplsd --belieffind --mwpm \
    --use_syndrome_p \
    --out_dir runs/eval_big144

python -m gnn_pipeline.evaluate \
    --test_npz data/big288_test_p05.npz \
    --gnn_model runs/big288_film/best_model.pt \
    --bposd --bplsd --belieffind --mwpm \
    --use_syndrome_p \
    --out_dir runs/eval_big288

echo "Phase 3 done: decoder comparison complete for all 3 codes."

# =============================================================================
#  PHASE 4 — Threshold sweeps with drift
#  p from 0.005 to 0.10, 15 points, 10k shots per point.
#  Tests static + 3 drift models (sine, OU, RTN).
#  FiLM model handles all conditions — no retraining needed.
# =============================================================================
echo ""
echo "=========================================="
echo " PHASE 4: Threshold sweeps (15 p-points x 4 noise conditions x 3 codes)"
echo "=========================================="

python -m gnn_pipeline.threshold_sweep \
    --code 72_12_6 \
    --p_min 0.005 --p_max 0.10 --num_points 15 \
    --shots 10000 --eta 20 \
    --drift_models sine,ou,rtn --drift_amp 0.015 \
    --gnn_model runs/big72_film/best_model.pt \
    --bposd --bplsd --mwpm \
    --out_dir runs/sweep_big72

python -m gnn_pipeline.threshold_sweep \
    --code 144_12_12 \
    --p_min 0.005 --p_max 0.10 --num_points 15 \
    --shots 10000 --eta 20 \
    --drift_models sine,ou,rtn --drift_amp 0.015 \
    --gnn_model runs/big144_film/best_model.pt \
    --bposd --bplsd --mwpm \
    --out_dir runs/sweep_big144

python -m gnn_pipeline.threshold_sweep \
    --code 288_12_18 \
    --p_min 0.005 --p_max 0.10 --num_points 15 \
    --shots 10000 --eta 20 \
    --drift_models sine,ou,rtn --drift_amp 0.015 \
    --gnn_model runs/big288_film/best_model.pt \
    --bposd --bplsd --mwpm \
    --out_dir runs/sweep_big288

echo "Phase 4 done: threshold sweeps with drift complete."

# =============================================================================
#  PHASE 5 — Transfer learning: 72-qubit model -> 144 and 288
#  Two strategies per target: frozen backbone (only retrain readout) and
#  full fine-tune (low lr). Evaluates both to populate transfer learning table.
# =============================================================================
echo ""
echo "=========================================="
echo " PHASE 5: Transfer learning (72 -> 144, 72 -> 288)"
echo "=========================================="

# --- 72 -> 144: frozen backbone ---
python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big144_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --pretrained runs/big72_film/best_model.pt --freeze_backbone \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 10 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive \
    --loss focal --syn_weight 0.1 \
    --epochs 10 --batch_size 16 --lr 5e-4 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 1 \
    --amp --patience 10 \
    --out_dir runs/transfer_72to144_frozen

# --- 72 -> 144: full fine-tune ---
python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big144_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --pretrained runs/big72_film/best_model.pt \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 10 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive \
    --loss focal --syn_weight 0.1 \
    --epochs 10 --batch_size 16 --lr 1e-5 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 1 \
    --amp --patience 10 \
    --out_dir runs/transfer_72to144_finetune

# --- 72 -> 288: frozen backbone ---
python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big288_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --pretrained runs/big72_film/best_model.pt --freeze_backbone \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 10 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive \
    --loss focal --syn_weight 0.1 \
    --epochs 10 --batch_size 16 --lr 5e-4 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 1 \
    --amp --patience 10 \
    --out_dir runs/transfer_72to288_frozen

# --- 72 -> 288: full fine-tune ---
python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big288_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --pretrained runs/big72_film/best_model.pt \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 10 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive \
    --loss focal --syn_weight 0.1 \
    --epochs 10 --batch_size 16 --lr 1e-5 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 1 \
    --amp --patience 10 \
    --out_dir runs/transfer_72to288_finetune

# --- Evaluate all transfer models ---
python -m gnn_pipeline.evaluate \
    --test_npz data/big144_test_p05.npz \
    --gnn_model runs/transfer_72to144_frozen/best_model.pt \
    --bposd --bplsd --use_syndrome_p \
    --out_dir runs/eval_transfer_144_frozen

python -m gnn_pipeline.evaluate \
    --test_npz data/big144_test_p05.npz \
    --gnn_model runs/transfer_72to144_finetune/best_model.pt \
    --bposd --bplsd --use_syndrome_p \
    --out_dir runs/eval_transfer_144_finetune

python -m gnn_pipeline.evaluate \
    --test_npz data/big288_test_p05.npz \
    --gnn_model runs/transfer_72to288_frozen/best_model.pt \
    --bposd --bplsd --use_syndrome_p \
    --out_dir runs/eval_transfer_288_frozen

python -m gnn_pipeline.evaluate \
    --test_npz data/big288_test_p05.npz \
    --gnn_model runs/transfer_72to288_finetune/best_model.pt \
    --bposd --bplsd --use_syndrome_p \
    --out_dir runs/eval_transfer_288_finetune

echo "Phase 5 done: transfer learning trained and evaluated."

# =============================================================================
#  PHASE 6 — Component ablation on 288-qubit code
#  Uses configs/component_ablation.json: cumulative component addition
#  Plain BP -> +Neural BP -> +GNN -> +FiLM -> +Both mode -> +Interleaved
# =============================================================================
echo ""
echo "=========================================="
echo " PHASE 6: Component ablation (288-qubit code)"
echo "=========================================="

python -m gnn_pipeline.ablation \
    --grid_json configs/component_ablation.json \
    --train_data "data/big288_test_p05.npz" \
    --test_data data/big288_test_p05.npz \
    --bposd --bplsd \
    --epochs 15 \
    --out_dir runs/ablation_components

echo "Phase 6 done: component ablation complete."

# =============================================================================
#  PHASE 7 — Interleaved GNN-BP training and evaluation
#  GNN corrects mid-BP using intermediate marginals as 5th node feature.
#  Two GNN passes: before BP and between BP Stage 1 / Stage 2.
# =============================================================================
echo ""
echo "=========================================="
echo " PHASE 7: Interleaved GNN-BP (3 codes)"
echo "=========================================="

python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big72_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 20 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive --node_feat_dim 5 \
    --interleaved_train --stage1_iters 10 --stage2_iters 10 \
    --loss focal --syn_weight 0.1 \
    --epochs 15 --batch_size 16 --lr 1e-4 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 2 \
    --amp --patience 10 \
    --out_dir runs/big72_interleaved

python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big144_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 20 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive --node_feat_dim 5 \
    --interleaved_train --stage1_iters 10 --stage2_iters 10 \
    --loss focal --syn_weight 0.1 \
    --epochs 15 --batch_size 16 --lr 1e-4 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 2 \
    --amp --patience 10 \
    --out_dir runs/big144_interleaved

python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --online_regen --code_npz data/big288_test_p05.npz \
    --p_range 0.01 0.10 --p_base 0.05 --eta 20 \
    --samples_per_epoch 10000 \
    --hidden_dim 64 --num_mp_layers 3 --dropout 0.1 \
    --bp_iters 20 --neural_bp --use_film --noise_feat_dim 1 \
    --correction_mode additive --node_feat_dim 5 \
    --interleaved_train --stage1_iters 10 --stage2_iters 10 \
    --loss focal --syn_weight 0.1 \
    --epochs 15 --batch_size 16 --lr 1e-4 \
    --weight_decay 1e-4 --scheduler cosine --warmup_epochs 2 \
    --amp --patience 10 \
    --out_dir runs/big288_interleaved

# --- Evaluate interleaved models ---
python -m gnn_pipeline.evaluate \
    --test_npz data/big72_test_p05.npz \
    --gnn_model runs/big72_interleaved/best_model.pt \
    --bposd --bplsd --use_syndrome_p \
    --out_dir runs/eval_big72_interleaved

python -m gnn_pipeline.evaluate \
    --test_npz data/big144_test_p05.npz \
    --gnn_model runs/big144_interleaved/best_model.pt \
    --bposd --bplsd --use_syndrome_p \
    --out_dir runs/eval_big144_interleaved

python -m gnn_pipeline.evaluate \
    --test_npz data/big288_test_p05.npz \
    --gnn_model runs/big288_interleaved/best_model.pt \
    --bposd --bplsd --use_syndrome_p \
    --out_dir runs/eval_big288_interleaved

echo "Phase 7 done: interleaved models trained and evaluated."

# =============================================================================
#  PHASE 8 — Generate all figures
#  make_figures auto-detects eval results.json and sweep results.json
#  in subdirectories. We point it at the parent dirs to collect everything.
# =============================================================================
echo ""
echo "=========================================="
echo " PHASE 8: Generating all figures"
echo "=========================================="

# Per-code figures (decoder comparison bars, convergence, etc.)
python -m gnn_pipeline.make_figures --results_dir runs/eval_big72              --out_dir figures/big72             --format pdf
python -m gnn_pipeline.make_figures --results_dir runs/eval_big144             --out_dir figures/big144            --format pdf
python -m gnn_pipeline.make_figures --results_dir runs/eval_big288             --out_dir figures/big288            --format pdf

# Interleaved comparison figures
python -m gnn_pipeline.make_figures --results_dir runs/eval_big72_interleaved  --out_dir figures/big72_interleaved --format pdf
python -m gnn_pipeline.make_figures --results_dir runs/eval_big144_interleaved --out_dir figures/big144_interleaved --format pdf
python -m gnn_pipeline.make_figures --results_dir runs/eval_big288_interleaved --out_dir figures/big288_interleaved --format pdf

# Threshold sweep figures (LER vs p curves, drift comparison)
python -m gnn_pipeline.make_figures --results_dir runs/sweep_big72             --out_dir figures/sweep72           --format pdf
python -m gnn_pipeline.make_figures --results_dir runs/sweep_big144            --out_dir figures/sweep144          --format pdf
python -m gnn_pipeline.make_figures --results_dir runs/sweep_big288            --out_dir figures/sweep288          --format pdf

# Ablation figures
python -m gnn_pipeline.make_figures --results_dir runs/ablation_components     --out_dir figures/ablation          --format pdf

# Transfer learning figures
python -m gnn_pipeline.make_figures --results_dir runs/eval_transfer_144_frozen    --out_dir figures/transfer144_frozen    --format pdf
python -m gnn_pipeline.make_figures --results_dir runs/eval_transfer_144_finetune  --out_dir figures/transfer144_finetune  --format pdf
python -m gnn_pipeline.make_figures --results_dir runs/eval_transfer_288_frozen    --out_dir figures/transfer288_frozen    --format pdf
python -m gnn_pipeline.make_figures --results_dir runs/eval_transfer_288_finetune  --out_dir figures/transfer288_finetune  --format pdf

# Combined multi-code figure (all codes on one plot) — point at parent runs/ dir
python -m gnn_pipeline.make_figures --results_dir runs                         --out_dir figures/combined          --format pdf

echo ""
echo "=========================================="
echo " ALL PHASES COMPLETE"
echo "=========================================="
echo ""
echo "Results:  runs/"
echo "Figures:  figures/"
echo ""
echo "Key outputs for paper:"
echo "  figures/combined/fig1_*   - LER vs code size scaling (Fig 1)"
echo "  figures/big72/fig2_*      - Decoder comparison bars (Fig 2)"
echo "  figures/sweep72/fig3_*    - Threshold curves (Fig 3)"
echo "  figures/sweep72/fig4_*    - Static vs drift comparison (Fig 4)"
echo "  figures/*/fig5_*          - Training dynamics (Fig 5)"
echo "  figures/combined/fig6_*   - Improvement heatmap (Fig 6)"
echo "  figures/combined/fig7_*   - Multi-code threshold overlay (Fig 7)"
echo "  figures/ablation/fig8_*   - Component ablation (Fig 8)"
echo "  figures/combined/fig9_*   - Transfer learning comparison (Fig 9)"
echo "  figures/combined/fig10_*  - Timing / latency bars (Fig 10)"
echo "  figures/sweep72/fig11_*   - BP convergence rate vs p (Fig 11)"
echo "  figures/combined/fig12_*  - FiLM noise estimation accuracy (Fig 12)"
echo "  runs/eval_big*/           - McNemar statistical tests (JSON)"
echo ""
echo "Copy runs/ and figures/ to your local machine for paper integration."

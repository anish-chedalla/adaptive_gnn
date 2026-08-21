#!/usr/bin/env python3
"""Confound 4 – Capacity test.

Train GNN for 50 epochs on exactly 1000 FIXED syndromes.
Evaluate on THOSE SAME 1000 syndromes.
Verdict: if GNN cannot improve on its own training set, architecture is broken.
"""
from __future__ import annotations
import sys, time, json, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, '/home/user/adaptive_gnn')
from gnn_pipeline.tanner_graph import build_tanner_graph
from gnn_pipeline.gnn_model import TannerGNN
from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.loss_functions import focal_loss

torch.manual_seed(42); np.random.seed(42)

# ── Load 1000 fixed syndromes ─────────────────────────────────────────────────
DATA_FILE = 'data/big72_test_p04.npz'
N = 1000
print(f"Loading {N} fixed syndromes from {DATA_FILE}...")
raw = np.load(DATA_FILE, allow_pickle=True)
hx = raw['hx'].astype(np.uint8); hz = raw['hz'].astype(np.uint8)
lx = raw['lx'].astype(np.float32); lz = raw['lz'].astype(np.float32)
n = hx.shape[1]; mx = hx.shape[0]; mz = hz.shape[0]
lx_t = torch.from_numpy(lx); lz_t = torch.from_numpy(lz)
num_nodes = n + mx + mz

# Collapse multi-round syndromes
syn_raw   = raw['syndromes'][:N].astype(np.float32)
total_chk = mx + mz
num_rnds  = syn_raw.shape[1] // total_chk
syn3d     = syn_raw.reshape(N, num_rnds, total_chk)
collapsed = (syn3d.sum(axis=1) % 2).astype(np.float32)   # (N, mx+mz)
z_err_np  = raw['z_errors'][:N].astype(np.float32)
x_err_np  = raw['x_errors'][:N].astype(np.float32)
p_vals    = raw['p_values'][:N].astype(np.float32)
llr_vals  = np.log((1-p_vals)/(p_vals+1e-12)).astype(np.float32)  # (N,)

x_syn_t = torch.from_numpy(collapsed[:, :mx])
z_syn_t = torch.from_numpy(collapsed[:, mx:])
llr_t   = torch.from_numpy(np.outer(llr_vals, np.ones(n)).astype(np.float32))
z_err_t = torch.from_numpy(z_err_np)
x_err_t = torch.from_numpy(x_err_np)

# ── Tanner graph topology ─────────────────────────────────────────────────────
node_type_np, edge_index_np, edge_type_np = build_tanner_graph(hx, hz)
nt_t  = torch.from_numpy(node_type_np)
ei_t  = torch.from_numpy(edge_index_np)
ety_t = torch.from_numpy(edge_type_np)
print(f"Tanner graph: {num_nodes} nodes, {ei_t.shape[1]} edges")

def make_data(i):
    x = torch.zeros(num_nodes, 4)
    x[:n, 0] = float(llr_vals[i]); x[:n, 1] = 1.0
    x[n:n+mx, 0] = torch.from_numpy(collapsed[i, :mx]); x[n:n+mx, 2] = 1.0
    x[n+mx:,  0] = torch.from_numpy(collapsed[i, mx:]); x[n+mx:,  3] = 1.0
    return Data(x=x, edge_index=ei_t, edge_type=ety_t, node_type=nt_t,
                y=torch.from_numpy(z_err_np[i]))

print(f"Building {N} Tanner graph Data objects...")
t0 = time.time()
graphs = [make_data(i) for i in range(N)]
print(f"  Done in {time.time()-t0:.1f}s")
loader = DataLoader(graphs, batch_size=32, shuffle=True)
all_loader = DataLoader(graphs, batch_size=N, shuffle=False)

# ── Decoders ──────────────────────────────────────────────────────────────────
dec_z = MinSumBPDecoder(hx, max_iter=10, alpha=0.8)
dec_x = MinSumBPDecoder(hz, max_iter=10, alpha=0.8)

# ── Helpers ────────────────────────────────────────────────────────────────────
def logical_errors(hz_dec, hx_dec, z_gt, x_gt):
    z_net = (hz_dec.float() + z_gt) % 2
    x_net = (hx_dec.float() + x_gt) % 2
    return ((z_net @ lx_t.t() + x_net @ lz_t.t()) % 2).sum(1).gt(0).numpy().astype(bool)

def mcnemar(a, b):
    n01 = int(((~a)&b).sum()); n10 = int((a&(~b)).sum()); d=n01+n10
    if d<1: return dict(n01=n01,n10=n10,chi2=0.,p=1.,disc=d)
    chi2=(abs(n01-n10)-1)**2/d
    return dict(n01=n01,n10=n10,chi2=chi2,p=math.erfc(math.sqrt(chi2/2)),disc=d)

def gnn_eval(gnn):
    gnn.eval()
    with torch.no_grad():
        b = next(iter(all_loader))
        corr = gnn(b).view(N, n)   # (N, n) additive corrections
        _, hz_d, _ = dec_z(x_syn_t, llr_t + corr)
        _, hx_d, _ = dec_x(z_syn_t, llr_t + corr)
    return logical_errors(hz_d, hx_d, z_err_t, x_err_t)

# ── Baseline: pure flooding BP ─────────────────────────────────────────────────
print("\n=== BASELINE: Flooding BP (10 iter, 1000 shots) ===")
with torch.no_grad():
    _, hz_bp, _ = dec_z(x_syn_t, llr_t)
    _, hx_bp, _ = dec_x(z_syn_t, llr_t)
bp_out = logical_errors(hz_bp, hx_bp, z_err_t, x_err_t)
bp_ler = float(bp_out.mean()); bp_ev = int(bp_out.sum())
print(f"  LER={bp_ler:.4f} ({bp_ev}/{N} events)")

# ── GNN ────────────────────────────────────────────────────────────────────────
gnn = TannerGNN(node_feat_dim=4, hidden_dim=32, num_mp_layers=2,
                dropout=0.0, use_film=False, use_attention=False,
                use_residual=True, use_layer_norm=True)
print(f"GNN params: {sum(p.numel() for p in gnn.parameters()):,}")
opt = AdamW(gnn.parameters(), lr=3e-3, weight_decay=1e-4)

# ── Train 50 epochs, evaluate on SAME 1000 syndromes ─────────────────────────
print("\n=== TRAINING (50 epochs, fixed 1000 syndromes) ===")
epoch_results = []; best_ler = bp_ler; best_ep = 0

for ep in range(1, 51):
    t0 = time.time(); total_loss=0.; nb=0
    gnn.train()
    for batch in loader:
        opt.zero_grad()
        out = gnn(batch)                           # (B*n,)
        is_d = batch.node_type == 0
        avg_llr = batch.x[is_d, 0]                # (B*n,)
        pred = torch.sigmoid(-(avg_llr + out.view(-1)))
        targets = batch.y.view(-1).clamp(0, 1)
        loss = focal_loss(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
        opt.step(); total_loss += loss.item(); nb += 1

    avg_loss = total_loss / max(nb, 1)

    if ep % 5 == 0 or ep == 1 or ep == 50:
        gnn_out = gnn_eval(gnn)
        gnn_ler = float(gnn_out.mean()); gnn_ev = int(gnn_out.sum())
        mc = mcnemar(bp_out, gnn_out)
        elapsed = time.time()-t0
        rec = dict(epoch=ep, loss=avg_loss, ler=gnn_ler, events=gnn_ev, mcnemar=mc)
        epoch_results.append(rec)
        print(f"  Ep {ep:2d}: loss={avg_loss:.4f} | GNN LER={gnn_ler:.4f}({gnn_ev}) "
              f"BP={bp_ler:.4f} | n10↑={mc['n10']} n01↓={mc['n01']} p={mc['p']:.4f} | {elapsed:.1f}s")
        if gnn_ler < best_ler:
            best_ler = gnn_ler; best_ep = ep
            torch.save(gnn.state_dict(), '/tmp/audit_c4_best.pt')
    else:
        print(f"  Ep {ep:2d}: loss={avg_loss:.4f}")

verdict = 'GNN_CAN_OVERFIT' if best_ler < bp_ler * 0.90 else 'GNN_CANNOT_OVERFIT'
results = dict(confound=4, code='[[72,12,6]]', data_file=DATA_FILE, n_shots=N,
               bp_ler=bp_ler, bp_events=bp_ev, best_gnn_ler=best_ler, best_epoch=best_ep,
               epoch_results=epoch_results, verdict=verdict)
with open('audit_c4_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n=== CONFOUND 4 VERDICT ===")
print(f"  Best GNN LER: {best_ler:.4f} (ep {best_ep}) vs BP: {bp_ler:.4f}")
print(f"  Verdict: {verdict}")

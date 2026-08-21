#!/usr/bin/env python3
"""Confound 1 – Training/evaluation schedule mismatch.

Test A: GNN vs FLOODING BP (the decoder it was trained against)
Test B: GNN vs BP-OSD (the paper's evaluation baseline, if ldpc available)

[[72,12,6]], 8k training shots (p∈[0.02,0.03]), 8k test shots (p=0.04).
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_collapsed(path, N):
    raw = np.load(path, allow_pickle=True)
    hx = raw['hx'].astype(np.uint8); hz = raw['hz'].astype(np.uint8)
    lx = raw['lx'].astype(np.float32); lz = raw['lz'].astype(np.float32)
    n=hx.shape[1]; mx=hx.shape[0]; mz=hz.shape[0]; total=mx+mz
    syn = raw['syndromes'][:N].astype(np.float32)
    nr  = syn.shape[1]//total
    col = (syn.reshape(N,nr,total).sum(1) % 2).astype(np.float32)
    pv  = raw['p_values'][:N].astype(np.float32)
    llr = np.log((1-pv)/(pv+1e-12)).astype(np.float32)
    ze  = raw['z_errors'][:N].astype(np.float32)
    xe  = raw['x_errors'][:N].astype(np.float32)
    return dict(hx=hx,hz=hz,lx=lx,lz=lz,n=n,mx=mx,mz=mz,
                col=col,llr=llr,ze=ze,xe=xe,pv=pv)

TRAIN_FILES = ['data/big72_train_p02.npz', 'data/big72_train_p03.npz']
TEST_FILE   = 'data/big72_test_p04.npz'
N_TRAIN=8000; N_TEST=8000
HALF=N_TRAIN//2

print("Loading data...")
d1 = load_collapsed(TRAIN_FILES[0], HALF)
d2 = load_collapsed(TRAIN_FILES[1], HALF)
dt = load_collapsed(TEST_FILE, N_TEST)

hx=dt['hx']; hz=dt['hz']; lx=dt['lx']; lz=dt['lz']
n=dt['n']; mx=dt['mx']; mz=dt['mz']
lx_t = torch.from_numpy(lx); lz_t = torch.from_numpy(lz)
num_nodes = n+mx+mz

col_tr = np.concatenate([d1['col'], d2['col']])
llr_tr = np.concatenate([d1['llr'], d2['llr']])
ze_tr  = np.concatenate([d1['ze'],  d2['ze']])
xe_tr  = np.concatenate([d1['xe'],  d2['xe']])
print(f"  Train: {N_TRAIN} shots, p=[{min(d1['pv'].min(),d2['pv'].min()):.3f},{max(d1['pv'].max(),d2['pv'].max()):.3f}]")
print(f"  Test:  {N_TEST} shots, p={dt['pv'].mean():.3f}")

# ── Tanner graph topology ─────────────────────────────────────────────────────
node_type_np, edge_index_np, edge_type_np = build_tanner_graph(hx, hz)
nt_t  = torch.from_numpy(node_type_np)
ei_t  = torch.from_numpy(edge_index_np)
ety_t = torch.from_numpy(edge_type_np)

def make_data(col, llr_val, ze, ye=None):
    x = torch.zeros(num_nodes, 4)
    x[:n, 0] = float(llr_val); x[:n, 1] = 1.0
    x[n:n+mx, 0] = torch.from_numpy(col[:mx]); x[n:n+mx, 2] = 1.0
    x[n+mx:,  0] = torch.from_numpy(col[mx:]); x[n+mx:,  3] = 1.0
    return Data(x=x, edge_index=ei_t, edge_type=ety_t, node_type=nt_t,
                y=torch.from_numpy(ze))

print("Building Tanner graph datasets...")
t0=time.time()
train_graphs = [make_data(col_tr[i], llr_tr[i], ze_tr[i]) for i in range(N_TRAIN)]
test_graphs  = [make_data(dt['col'][i], dt['llr'][i], dt['ze'][i]) for i in range(N_TEST)]
print(f"  Done in {time.time()-t0:.1f}s")

train_loader  = DataLoader(train_graphs, batch_size=64, shuffle=True)
test_all_ld   = DataLoader(test_graphs,  batch_size=512, shuffle=False)

# Tensors for BP decoding
x_syn_te = torch.from_numpy(dt['col'][:, :mx])
z_syn_te = torch.from_numpy(dt['col'][:, mx:])
llr_te   = torch.from_numpy(np.outer(dt['llr'], np.ones(n)).astype(np.float32))
ze_te_t  = torch.from_numpy(dt['ze']); xe_te_t = torch.from_numpy(dt['xe'])

# ── Decoders ──────────────────────────────────────────────────────────────────
dec_z = MinSumBPDecoder(hx, max_iter=10, alpha=0.8)
dec_x = MinSumBPDecoder(hz, max_iter=10, alpha=0.8)

def logical_errors(hz_d, hx_d, ze_t, xe_t):
    z_net = (hz_d.float()+ze_t) % 2; x_net = (hx_d.float()+xe_t) % 2
    return ((z_net@lx_t.t() + x_net@lz_t.t()) % 2).sum(1).gt(0).numpy().astype(bool)

def mcnemar(a,b):
    n01=int(((~a)&b).sum()); n10=int((a&(~b)).sum()); d=n01+n10
    if d<1: return dict(n01=n01,n10=n10,chi2=0.,p=1.,disc=d)
    chi2=(abs(n01-n10)-1)**2/d
    return dict(n01=n01,n10=n10,chi2=chi2,p=math.erfc(math.sqrt(chi2/2)),disc=d)

def gnn_decode(gnn):
    gnn.eval()
    all_hz=[]; all_hx=[]; start=0
    with torch.no_grad():
        for batch in test_all_ld:
            B=batch.num_graphs
            corr=gnn(batch).view(B,n)
            ll=llr_te[start:start+B]; xs=x_syn_te[start:start+B]; zs=z_syn_te[start:start+B]
            _,hz_d,_=dec_z(xs,ll+corr); _,hx_d,_=dec_x(zs,ll+corr)
            all_hz.append(hz_d); all_hx.append(hx_d); start+=B
    return torch.cat(all_hz), torch.cat(all_hx)

# ── Baseline: Flooding BP ──────────────────────────────────────────────────────
print("\n=== BASELINE: Flooding BP (10 iter) ===")
with torch.no_grad():
    _,hz_bp,_=dec_z(x_syn_te,llr_te); _,hx_bp,_=dec_x(z_syn_te,llr_te)
bp_out=logical_errors(hz_bp,hx_bp,ze_te_t,xe_te_t)
bp_ler=float(bp_out.mean()); bp_ev=int(bp_out.sum())
print(f"  LER={bp_ler:.4f} ({bp_ev}/{N_TEST})")

# ── BP-OSD baseline ────────────────────────────────────────────────────────────
HAVE_BPOSD=False; bposd_ler=None; bposd_out=None; bposd_ev=None
try:
    from ldpc import BpOsdDecoder
    HAVE_BPOSD=True
    print("\n=== BASELINE: BP-OSD (flooding, 100 iter, osd_cs-10) ===")
    pm=float(dt['pv'].mean())
    dzO=BpOsdDecoder(hx,error_rate=pm,max_iter=100,bp_method='ms',
                     ms_scaling_factor=0.625,schedule='flooding',osd_method='osd_cs',osd_order=10)
    dxO=BpOsdDecoder(hz,error_rate=pm,max_iter=100,bp_method='ms',
                     ms_scaling_factor=0.625,schedule='flooding',osd_method='osd_cs',osd_order=10)
    syn_u8=(dt['col']>0.5).astype(np.uint8)
    zds=np.zeros((N_TEST,n),np.uint8); xds=np.zeros((N_TEST,n),np.uint8)
    for i in range(N_TEST):
        zds[i]=dzO.decode(syn_u8[i,:mx]); xds[i]=dxO.decode(syn_u8[i,mx:])
        if i%2000==0: print(f"  OSD {i}/{N_TEST}")
    bposd_out=logical_errors(torch.from_numpy(zds.astype(np.float32)),
                             torch.from_numpy(xds.astype(np.float32)),ze_te_t,xe_te_t)
    bposd_ler=float(bposd_out.mean()); bposd_ev=int(bposd_out.sum())
    print(f"  BP-OSD LER={bposd_ler:.4f} ({bposd_ev}/{N_TEST})")
except Exception as e:
    print(f"  BP-OSD skipped: {e}")

# ── Train GNN (trained against flooding BP — what the paper did) ───────────────
print(f"\n=== TRAINING GNN ({N_TRAIN} shots, 20 epochs) ===")
gnn=TannerGNN(node_feat_dim=4,hidden_dim=32,num_mp_layers=3,
              dropout=0.0,use_film=False,use_attention=False,
              use_residual=True,use_layer_norm=True)
print(f"GNN params: {sum(p.numel() for p in gnn.parameters()):,}")
opt=AdamW(gnn.parameters(),lr=2e-3,weight_decay=1e-4)

train_history=[]; best_val=1.0; best_ep=0

for ep in range(1,21):
    t0=time.time(); total_loss=0.; nb=0
    gnn.train()
    for batch in train_loader:
        opt.zero_grad()
        out=gnn(batch)
        is_d=batch.node_type==0
        avg=batch.x[is_d,0]; d=out.view(-1)
        pred=torch.sigmoid(-(avg+d))
        tgt=batch.y.view(-1).clamp(0,1)
        loss=focal_loss(pred,tgt)
        loss.backward(); torch.nn.utils.clip_grad_norm_(gnn.parameters(),1.); opt.step()
        total_loss+=loss.item(); nb+=1

    avg_loss=total_loss/max(nb,1)

    if ep%5==0 or ep==1 or ep==20:
        hz_g,hx_g=gnn_decode(gnn)
        gout=logical_errors(hz_g,hx_g,ze_te_t,xe_te_t)
        gl=float(gout.mean()); ge=int(gout.sum())
        mc_a=mcnemar(bp_out,gout)
        elapsed=time.time()-t0
        rec=dict(epoch=ep,loss=avg_loss,gnn_ler=gl,gnn_events=ge,mcnemar_a=mc_a)
        train_history.append(rec)
        print(f"  Ep {ep:2d}: loss={avg_loss:.4f} | GNN={gl:.4f}({ge}) "
              f"BP={bp_ler:.4f} | n10↑={mc_a['n10']} n01↓={mc_a['n01']} "
              f"p={mc_a['p']:.4f} | {elapsed:.1f}s")
        if gl<best_val:
            best_val=gl; best_ep=ep
            torch.save(gnn.state_dict(),'/tmp/audit_c1_best.pt')
    else:
        print(f"  Ep {ep:2d}: loss={avg_loss:.4f}")

# ── Final eval with best model ─────────────────────────────────────────────────
print(f"\n=== FINAL (best ep={best_ep}) ===")
gnn.load_state_dict(torch.load('/tmp/audit_c1_best.pt',map_location='cpu'))
hz_g,hx_g=gnn_decode(gnn)
fo=logical_errors(hz_g,hx_g,ze_te_t,xe_te_t)
fl=float(fo.mean()); fe=int(fo.sum())
mc_af=mcnemar(bp_out,fo)
print(f"  TEST A — GNN+flooding vs flooding alone:")
print(f"    BP  LER={bp_ler:.4f} ({bp_ev})")
print(f"    GNN LER={fl:.4f} ({fe})")
print(f"    McNemar: n10(GNN better)={mc_af['n10']}, n01(GNN worse)={mc_af['n01']}, "
      f"chi2={mc_af['chi2']:.3f}, p={mc_af['p']:.4f}")
if HAVE_BPOSD and bposd_out is not None:
    mc_bpo=mcnemar(bp_out,bposd_out)
    print(f"\n  Reference: BP vs BP-OSD: p={mc_bpo['p']:.4f} "
          f"n10(OSD better)={mc_bpo['n10']} n01(OSD worse)={mc_bpo['n01']}")

verdict_a=('GNN_IMPROVES_FLOODING' if fl<bp_ler*0.90 and mc_af['p']<0.05
           else 'GNN_WORSENS_FLOODING' if fl>bp_ler*1.10 and mc_af['p']<0.05
           else 'GNN_NO_EFFECT_ON_FLOODING')
mc_bpo_final=mcnemar(bp_out,bposd_out) if (HAVE_BPOSD and bposd_out is not None) else None
results=dict(confound=1,code='[[72,12,6]]',n_train=N_TRAIN,n_test=N_TEST,
             bp_ler=bp_ler,bp_events=bp_ev,bposd_ler=bposd_ler,bposd_events=bposd_ev,
             gnn_ler=fl,gnn_events=fe,mcnemar_test_a=mc_af,
             mcnemar_bp_vs_bposd=mc_bpo_final,
             train_history=train_history,best_epoch=best_ep,
             test_a_verdict=verdict_a)
with open('audit_c1_results.json','w') as f:
    json.dump(results,f,indent=2)
print(f"\nTest A verdict: {verdict_a}")
print("Results saved to audit_c1_results.json")

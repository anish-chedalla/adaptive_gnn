#!/usr/bin/env python3
"""Confound 3 – Gradient flow through GNN message passing.

Measures gradient magnitude ratio: GNN input layer vs readout layer.
If ratio < 0.01, gradients vanish through the MP stack.
Reports mean |delta_i| at init vs trained.
"""
from __future__ import annotations
import sys, json, math, pathlib
import numpy as np
import torch
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, '/home/user/adaptive_gnn')
from gnn_pipeline.tanner_graph import build_tanner_graph
from gnn_pipeline.gnn_model import TannerGNN
from gnn_pipeline.loss_functions import focal_loss

torch.manual_seed(42); np.random.seed(42)

DATA_FILE='data/big72_test_p04.npz'; N=256
raw=np.load(DATA_FILE,allow_pickle=True)
hx=raw['hx'].astype(np.uint8); hz=raw['hz'].astype(np.uint8)
n=hx.shape[1]; mx=hx.shape[0]; mz=hz.shape[0]; total=mx+mz; num_nodes=n+mx+mz

syn=(raw['syndromes'][:N]).astype(np.float32)
nr=syn.shape[1]//total; col=(syn.reshape(N,nr,total).sum(1)%2).astype(np.float32)
pv=raw['p_values'][:N].astype(np.float32)
llr=np.log((1-pv)/(pv+1e-12)).astype(np.float32)
ze=raw['z_errors'][:N].astype(np.float32)

node_type_np,edge_index_np,edge_type_np=build_tanner_graph(hx,hz)
nt_t=torch.from_numpy(node_type_np)
ei_t=torch.from_numpy(edge_index_np)
ety_t=torch.from_numpy(edge_type_np)

def mk(i):
    x=torch.zeros(num_nodes,4)
    x[:n,0]=float(llr[i]); x[:n,1]=1.
    x[n:n+mx,0]=torch.from_numpy(col[i,:mx]); x[n:n+mx,2]=1.
    x[n+mx:, 0]=torch.from_numpy(col[i,mx:]); x[n+mx:, 3]=1.
    return Data(x=x,edge_index=ei_t,edge_type=ety_t,node_type=nt_t,
                y=torch.from_numpy(ze[i]))

graphs=[mk(i) for i in range(N)]
loader=DataLoader(graphs,batch_size=N,shuffle=False)
batch=next(iter(loader))

def build_gnn():
    return TannerGNN(node_feat_dim=4,hidden_dim=32,num_mp_layers=3,
                     dropout=0.,use_film=False,use_attention=False,
                     use_residual=True,use_layer_norm=True)

def measure(gnn,batch,desc):
    gnn.train()
    for p in gnn.parameters():
        if p.grad is not None: p.grad.zero_()
    out=gnn(batch)
    is_d=batch.node_type==0
    avg=batch.x[is_d,0]; d=out.view(-1)
    pred=torch.sigmoid(-(avg+d))
    tgt=batch.y.view(-1).clamp(0,1)
    loss=focal_loss(pred,tgt)
    loss.backward()

    grads={}
    for nm,p in gnn.named_parameters():
        if p.grad is not None:
            g=p.grad.detach().abs()
            grads[nm]=dict(mean=float(g.mean()),max=float(g.max()),norm=float(g.norm()))

    in_ms  =[grads[k]['mean'] for k in grads if 'input_proj' in k]
    out_ms  =[grads[k]['mean'] for k in grads if 'readout'    in k]
    mp_ms   =[grads[k]['mean'] for k in grads if 'mp_layers'  in k]

    mi=float(np.mean(in_ms))  if in_ms  else 0.
    mo=float(np.mean(out_ms)) if out_ms else 0.
    mm=float(np.mean(mp_ms))  if mp_ms  else 0.
    ratio=mi/(mo+1e-12)
    mean_d=float(d.detach().abs().mean())

    flag=('VANISHING' if ratio<0.01 else 'EXPLODING' if ratio>100 else 'HEALTHY')
    print(f"\n  {desc}")
    print(f"    input proj grad: {mi:.3e} | mp layers: {mm:.3e} | readout: {mo:.3e}")
    print(f"    ratio input/readout: {ratio:.4f} → {flag}")
    print(f"    mean |delta_i|: {mean_d:.6f}  loss={loss.item():.4f}")
    return dict(desc=desc,mi=mi,mm=mm,mo=mo,ratio=ratio,mean_delta=mean_d,
                loss=float(loss.item()),vanishing=ratio<0.01,exploding=ratio>100,flag=flag)

print("=== CONFOUND 3: Gradient flow ===")
gnn_init=build_gnn()
r0=measure(gnn_init,batch,"Epoch 0 (untrained)")

ckpts=['/tmp/audit_c1_best.pt','/tmp/audit_c4_best.pt','/tmp/audit_c3_trained.pt']
ck_used=next((c for c in ckpts if pathlib.Path(c).exists()),None)
if ck_used:
    print(f"\nLoading checkpoint: {ck_used}")
else:
    print("\nNo checkpoint — training 10 epochs...")
    gnn_tr=build_gnn(); tr_ld=DataLoader(graphs,batch_size=64,shuffle=True)
    opt=AdamW(gnn_tr.parameters(),lr=3e-3)
    for ep in range(10):
        gnn_tr.train()
        for b in tr_ld:
            opt.zero_grad(); o=gnn_tr(b)
            id_=b.node_type==0; av=b.x[id_,0]; dd=o.view(-1)
            pr=torch.sigmoid(-(av+dd)); tg=b.y.view(-1).clamp(0,1)
            fl=focal_loss(pr,tg); fl.backward()
            torch.nn.utils.clip_grad_norm_(gnn_tr.parameters(),1.); opt.step()
        print(f"  ep {ep+1}/10")
    torch.save(gnn_tr.state_dict(),'/tmp/audit_c3_trained.pt'); ck_used='/tmp/audit_c3_trained.pt'

gnn_tr=build_gnn()
gnn_tr.load_state_dict(torch.load(ck_used,map_location='cpu'))
r1=measure(gnn_tr,batch,f"Trained ({ck_used})")

verdict=('VANISHING_GRADIENTS' if r1['vanishing'] else
         'EXPLODING_GRADIENTS' if r1['exploding'] else
         'NEAR_ZERO_CORRECTIONS' if r1['mean_delta']<0.01 else 'HEALTHY')
print(f"\n=== CONFOUND 3 VERDICT ===")
print(f"  Untrained: ratio={r0['ratio']:.4f} delta={r0['mean_delta']:.6f}")
print(f"  Trained:   ratio={r1['ratio']:.4f} delta={r1['mean_delta']:.6f}")
print(f"  Verdict: {verdict}")

results=dict(confound=3,epoch0=r0,trained=r1,checkpoint=ck_used,verdict=verdict)
with open('audit_c3_results.json','w') as f: json.dump(results,f,indent=2)
print("Results saved to audit_c3_results.json")

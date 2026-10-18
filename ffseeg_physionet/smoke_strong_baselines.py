#!/usr/bin/env python3
"""Data-free API/shape smoke test; NEVER a scientific EEG model evaluation."""
import numpy as np
import torch
from three_run_motor_imagery import (
    EEGNet, GNet, MLP, norm_adj, make_adjacency,
    feature_bands, csp_lda, fb_csp_lda, riemann_mdm,
    train_net, predict_net
)
np.random.seed(42)
torch.set_num_threads(1)
n=48
X=(np.random.randn(n,14,320)*10.).astype(np.float32)
y=np.tile([0,1],n//2)
tr,te=X[:40],X[40:]
yt,ye=y[:40],y[40:]
adj=make_adjacency()
features=feature_bands(X)
assert features.shape==(48,14,4), features.shape
assert np.isfinite(features).all()
for label,fn in (
    ("CSP-LDA14",lambda:csp_lda(tr,yt,te)),
    ("FBCSP-LDA14",lambda:fb_csp_lda(tr,yt,te)),
    ("RiemannMDM14",lambda:riemann_mdm(tr,yt,te)),
):
    p=fn()
    assert p.shape==(8,) and np.isfinite(p).all(),(label,p)
    print("API-SMOKE-SUCCESS",label,"shape",p.shape,flush=True)
for name in ("FullGraph14","MI-InputGraph8","LearnedTopK8","MLP56","EEGNet14"):
    if name=="EEGNet14":
        xtr=(tr/20.).astype(np.float32);xv=(te/20.).astype(np.float32)
    else:
        xtr=features[:40];xv=features[40:]
    model,meta=train_net(name,xtr,yt,xv,ye,adj,
                         list(range(8)),seed=42,epochs=5)
    out_in=xv[:, :8, :] if name=="MI-InputGraph8" else xv
    p=predict_net(model,out_in)
    assert p.shape==(8,) and np.isfinite(p).all(),(name,p)
    print("API-SMOKE-SUCCESS",name,"params",meta["n_params"],flush=True)
print("ALL_DATA_FREE_SMOKE_TESTS_PASSED",flush=True)

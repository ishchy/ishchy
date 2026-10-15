#!/usr/bin/env python3
"""Exploratory held-out-subject sparse residual EEG GNN experiment.

Use only outer-fold train to fit feature ranking and scalers, validation to
select checkpoints, and held-out test for reporting. Reusing the historical
benchmark to design this model means significance is exploratory.
"""
import os, argparse, json, random, time
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from scipy.stats import wilcoxon

CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
EDGES = [("AF3","F7"),("AF3","F3"),("AF3","AF4"),("F7","F3"),("F7","FC5"),("F7","T7"),
         ("F3","FC5"),("F3","F4"),("FC5","T7"),("FC5","P7"),("FC5","FC6"),("T7","P7"),
         ("T7","T8"),("P7","O1"),("P7","P8"),("O1","O2"),("O2","P8"),("P8","T8"),
         ("T8","FC6"),("T8","F8"),("FC6","F4"),("FC6","F8"),("F4","F8"),
         ("F4","AF4"),("F8","AF4")]

def adjacency():
    mat = np.eye(14, dtype=np.float32)
    indices = {name: i for i, name in enumerate(CHANNELS)}
    for a, b in EDGES:
        mat[indices[a], indices[b]] = mat[indices[b], indices[a]] = 1.
    inv = mat.sum(1)**-.5
    return torch.tensor(inv[:,None]*mat*inv[None,:], dtype=torch.float32)

class FFSEEGResidual(nn.Module):
    def __init__(self, k=8, dropout=.2, mi_rank=None, residual=True):
        super().__init__()
        self.k, self.residual = k, residual
        self.register_buffer("adj", adjacency())
        self.gate = nn.Parameter(torch.zeros(14))
        if mi_rank is not None:
            with torch.no_grad():
                self.gate[mi_rank[:k]] = 1.
        self.g1 = nn.Linear(2,24)
        self.g2 = nn.Linear(48,16)
        self.fc1 = nn.Linear(14*16 + (14 if residual else 0),48)
        self.fc2 = nn.Linear(48,2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        soft = torch.sigmoid(self.gate)
        hard = torch.zeros_like(soft).scatter_(0,torch.topk(self.gate,self.k).indices,1.)
        mask = (hard-soft).detach()+soft
        z = x*mask
        h = z.unsqueeze(-1)
        h = F.relu(self.g1(torch.cat([h,torch.matmul(self.adj,h)],dim=-1)))
        h = self.dropout(h)
        h = F.relu(self.g2(torch.cat([h,torch.matmul(self.adj,h)],dim=-1)))
        flat = h.flatten(1)
        if self.residual:
            flat = torch.cat((flat,z),dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(flat))))

def fit(xtr,ytr,xva,yva,seed,*,residual=True,k=8,epochs=100,dropout=.2,lr=.002,weight_decay=.002):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ranking = np.argsort(-mutual_info_classif(xtr,ytr,random_state=seed),kind="stable")
    model = FFSEEGResidual(k=k,dropout=dropout,mi_rank=ranking,residual=residual)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    x,y,xv = torch.from_numpy(xtr),torch.from_numpy(ytr),torch.from_numpy(xva)
    best=-1.; bad=0; best_state=None; best_epoch=0
    for epoch in range(1,epochs+1):
        model.train()
        for ids in torch.randperm(len(y)).split(256):
            logits=model(x[ids])
            loss=F.cross_entropy(logits,y[ids])
            optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if epoch%5==0:
            model.eval()
            with torch.no_grad():
                prob=model(xv).softmax(1)[:,1].numpy()
            score=balanced_accuracy_score(yva,(prob>=.5).astype(int))
            if score>best+1e-5:
                best=score;best_epoch=epoch
                best_state={name:value.detach().clone() for name,value in model.state_dict().items()}
                bad=0
            else:
                bad+=1
            if bad>=12:
                break
    model.load_state_dict(best_state)
    selected=",".join(CHANNELS[i] for i in torch.topk(model.gate,k).indices.numpy())
    return model, dict(val_ba=best,best_epoch=best_epoch,selected=selected)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--splits",required=True)
    p.add_argument("--baseline",default="")
    p.add_argument("--out",required=True)
    p.add_argument("--folds",type=int,default=10)
    p.add_argument("--k",type=int,default=8)
    p.add_argument("--dropout",type=float,default=.2)
    p.add_argument("--lr",type=float,default=.002)
    p.add_argument("--seed",type=int,default=24701)
    p.add_argument("--variant",choices=["residual","noresidual"],default="residual")
    a=p.parse_args()
    torch.set_num_threads(1)
    df=pd.read_csv(a.dataset)
    sp=pd.read_csv(a.splits)
    X=df[CHANNELS].to_numpy(np.float32)
    y=df.label.to_numpy(np.int64)
    rows=[];foldrows=[]
    os.makedirs(a.out,exist_ok=True)
    for fold,s in sp.head(a.folds).iterrows():
        getmask=lambda col: df.subject.isin([int(q) for q in s[col].split(",")]).to_numpy()
        tr,va,te=(getmask(col) for col in ("train_subjects","validation_subjects","test_subjects"))
        assert not np.any((tr&va)|(tr&te)|(va&te)), "Subject leakage across partitions"
        scaler=StandardScaler().fit(X[tr])
        xtr=scaler.transform(X[tr]).astype(np.float32)
        xva=scaler.transform(X[va]).astype(np.float32)
        xte=scaler.transform(X[te]).astype(np.float32)
        t0=time.monotonic()
        model,info=fit(xtr,y[tr],xva,y[va],a.seed+fold,
                       residual=(a.variant=="residual"),k=a.k,dropout=a.dropout,lr=a.lr)
        model.eval()
        with torch.no_grad():
            probs=model(torch.from_numpy(xte)).softmax(1)[:,1].numpy()
        pred=(probs>=.5).astype(int)
        ba=balanced_accuracy_score(y[te],pred)
        f1=f1_score(y[te],pred)
        auc=roc_auc_score(y[te],probs)
        foldrows.append(dict(fold=fold+1,ba=ba,f1=f1,auc=auc,**info))
        for j,i in enumerate(np.flatnonzero(te)):
            rows.append(dict(fold=fold+1,model="FFSEEG-Residual",subject=int(df.subject.iloc[i]),
                             run=int(df.run.iloc[i]),window=int(df.window.iloc[i]),y_true=int(y[i]),
                             y_pred=int(pred[j]),prob_closed=float(probs[j])))
        print("fold",fold+1,"BA",round(ba,4),"F1",round(f1,4),"AUC",round(auc,4),
              "seconds",round(time.monotonic()-t0,1),flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out,"predictions_partial.csv"),index=False)
    pd.DataFrame(foldrows).to_csv(os.path.join(a.out,"fold_metrics.csv"),index=False)
    if a.folds!=len(sp):
        return
    pred=pd.DataFrame(rows)
    assert len(pred)==len(df) and not pred.duplicated(["subject","run","window"]).any()
    pred.to_csv(os.path.join(a.out,"predictions.csv"),index=False)
    subjects=[]
    for subject,g in pred.groupby("subject"):
        subjects.append(dict(subject=int(subject),
                             ba=balanced_accuracy_score(g.y_true,g.y_pred),
                             f1=f1_score(g.y_true,g.y_pred),
                             auc=roc_auc_score(g.y_true,g.prob_closed)))
    subject_metrics=pd.DataFrame(subjects)
    subject_metrics.to_csv(os.path.join(a.out,"subject_metrics.csv"),index=False)
    result=dict(model="FFSEEG-Residual",n_subjects=len(subject_metrics),n_folds=a.folds,
                ba_fold_mean=float(np.mean([r["ba"] for r in foldrows])),
                ba_subject_mean=float(subject_metrics.ba.mean()),
                f1_fold_mean=float(np.mean([r["f1"] for r in foldrows])),
                auc_fold_mean=float(np.mean([r["auc"] for r in foldrows])))
    if a.baseline:
        baseline=pd.read_csv(a.baseline)
        baseline=baseline[baseline.model.isin(["FFSEEG-GNN","FullGraphConv"])]
        for name,g in baseline.groupby("model"):
            old=g.groupby("subject").apply(
                lambda z: balanced_accuracy_score(z.y_true,z.y_pred),include_groups=False)
            differences=subject_metrics.set_index("subject").ba-old
            d=differences.to_numpy()
            rng=np.random.default_rng(2048)
            boot=d[rng.integers(0,len(d),size=(5000,len(d)))].mean(1)
            result["comparison_"+name]=dict(
                mean_difference_pp=float(d.mean()*100),
                ci95_pp=(100*np.quantile(boot,[.025,.975])).tolist(),
                wilcoxon_p=float(wilcoxon(d).pvalue))
    with open(os.path.join(a.out,"metrics_summary.json"),"w") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    main()

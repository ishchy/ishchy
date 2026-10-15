#!/usr/bin/env python3
"""Audit whether EEG global Top-K gate actually LEARNS a different subset.

Controlled same-architecture variants with training-only feature rankings:
  mi_fixed, mi_noisy_fixed, mi_learned,
  random_fixed, random_noisy_fixed, random_learned, neutral_learned.

The noisy fixed-vs-learned contrast holds Gumbel exploration constant, isolating
the optimization of selector logits. All choices are locked before the test.
This is exploratory: both prior benchmarks informed the experimental design.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from scipy.stats import wilcoxon
import physionet_sparse_residual as sparse
import uci_forward_time_validation as uci

VARIANTS = ["mi_fixed", "mi_noisy_fixed", "mi_learned",
            "random_fixed", "random_noisy_fixed", "random_learned", "neutral_learned"]
FEATURES = sparse.CHANNELS

class AuditedTopK(sparse.FFSEEGResidual):
    def __init__(self, variant, mi_rank, seed, k=8):
        super().__init__(k=k, residual=True, dropout=.20, mi_rank=None)
        self.variant = variant
        self.noisy = variant not in ("mi_fixed", "random_fixed")
        self.learnable = variant in ("mi_learned", "random_learned", "neutral_learned")
        self.tau = .8
        self.noise_scale = .25
        rng = np.random.default_rng(seed + 31977)
        if variant.startswith("mi_"):
            ordered = np.asarray(mi_rank)
        elif variant.startswith("random_"):
            ordered = rng.permutation(14)
        else:
            ordered = np.argsort(-rng.normal(size=14), kind="stable")
        initial = ordered[:k].tolist()
        with torch.no_grad():
            self.gate.zero_()
            if variant == "neutral_learned":
                self.gate.copy_(torch.tensor(rng.normal(0,.025,size=14),dtype=torch.float32))
            else:
                self.gate[initial] = .35
        self.gate.requires_grad_(self.learnable)
        self.initial_indices = sorted(initial)
        self.mi_indices = sorted(np.asarray(mi_rank)[:k].tolist())

    def selected_indices(self):
        return sorted(torch.topk(self.gate.detach(),self.k).indices.cpu().tolist())

    def forward(self, x):
        logits=self.gate
        if self.training and self.noisy:
            u=torch.rand_like(logits).clamp_(1e-6,1-1e-6)
            gumbel=-torch.log(-torch.log(u))
            logits=logits + self.noise_scale*gumbel
        hard=torch.zeros_like(logits).scatter_(0,torch.topk(logits,self.k).indices,1.)
        if self.learnable:
            soft=torch.softmax(self.gate / max(self.tau,.2),dim=0)*self.k
            mask=hard + soft - soft.detach()
        else:
            mask=hard
        z=x*mask
        h=z.unsqueeze(-1)
        h=F.relu(self.g1(torch.cat((h,torch.matmul(self.adj,h)),dim=-1)))
        h=self.dropout(h)
        h=F.relu(self.g2(torch.cat((h,torch.matmul(self.adj,h)),dim=-1)))
        v=torch.cat((h.flatten(1),z),dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(v))))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def learn(xtr,ytr,xva,yva,variant,mi_rank,seed,epochs):
    set_seed(seed)
    model=AuditedTopK(variant,mi_rank,seed,k=8)
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),
                          lr=.002,weight_decay=.002)
    xtr=torch.from_numpy(xtr);ytr=torch.from_numpy(ytr)
    xva=torch.from_numpy(xva)
    best=-np.inf;bad=0;state=None;best_ep=0
    for ep in range(1,epochs+1):
        model.tau=max(.3,.8*(.97**(ep-1)))
        model.noise_scale=max(.04,.25*(1-ep/epochs)) if model.noisy else 0.
        model.train()
        for indexes in torch.randperm(len(ytr)).split(256):
            score=model(xtr[indexes])
            loss=F.cross_entropy(score,ytr[indexes])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
        if ep%5==0:
            model.eval()
            with torch.inference_mode():
                probs=torch.cat([model(batch).softmax(1)[:,1] for batch in xva.split(512)]).numpy()
            ba=balanced_accuracy_score(yva,probs>=.5)
            if ba>best+1e-7:
                best=ba;best_ep=ep;bad=0
                state={name:p.detach().clone() for name,p in model.state_dict().items()}
            else:
                bad+=1
            if bad>=8:break
    if state is None:
        raise RuntimeError("No validation checkpoint")
    model.load_state_dict(state)
    model.eval()
    ending=model.selected_indices()
    initial=set(model.initial_indices); final=set(ending)
    mi=set(model.mi_indices)
    return model,dict(initial_channels=",".join(FEATURES[i] for i in model.initial_indices),
        final_channels=",".join(FEATURES[i] for i in ending),
        mi_channels=",".join(FEATURES[i] for i in sorted(mi)),
        moved_nodes=int(len(initial-final)),
        jaccard_initial_final=float(len(initial&final)/len(initial|final)),
        overlap_mi=int(len(mi&final)),
        validation_ba=float(best),best_epoch=best_ep,
        n_params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)))

def get_physionet(csv,splits):
    df=pd.read_csv(csv)
    sp=pd.read_csv(splits)
    if df.subject.nunique()!=109:raise ValueError("Expected 109 subjects")
    if len(sp)!=10:raise ValueError("Expected 10 original folds")
    fold_items=[]
    for _,s in sp.iterrows():
        masks=[]
        for col in ("train_subjects","validation_subjects","test_subjects"):
            subs=[int(a) for a in s[col].split(",")]
            masks.append(np.flatnonzero(df.subject.isin(subs).to_numpy()))
        tr,va,te=masks
        if len(set(tr)&set(va)) or len(set(tr)&set(te)) or len(set(va)&set(te)):
            raise ValueError("Overlapping subject groups")
        fold_items.append((tr,va,te))
    return df, df[FEATURES].to_numpy(np.float32), df.label.to_numpy(np.int64), fold_items

def get_uci(zipfile):
    df,_=uci.load_data(Path(zipfile))
    return df,df[FEATURES].to_numpy(np.float32),df.eyeDetection.to_numpy(np.int64),\
        uci.make_forward_folds(len(df),256,1250)

def metrics(y,p):
    pred=(p>=.5).astype(int)
    return dict(balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                f1=float(f1_score(y,pred,zero_division=0)),
                auc=float(roc_auc_score(y,p))),pred

def paired_subject_stats(pred,comparisons=(("mi_learned","mi_noisy_fixed"),
                                            ("random_learned","random_noisy_fixed"),
                                            ("mi_learned","mi_fixed"),
                                            ("random_learned","random_fixed"),
                                            ("neutral_learned","mi_fixed"))):
    rows=[]
    by=pred.groupby(["seed","variant","subject"])
    subj=by.apply(lambda g:balanced_accuracy_score(g.y_true,g.y_pred),include_groups=False)
    subj=subj.rename("ba").reset_index()
    pivot=subj.pivot(index=["seed","subject"],columns="variant",values="ba")
    # Average over seeds FIRST, then resample 109 subjects, never pseudo-replicate seeds.
    bysubj=pivot.groupby(level="subject").mean()
    for a,b in comparisons:
        d=(bysubj[a]-bysubj[b]).dropna().to_numpy()
        rng=np.random.default_rng(1206)
        samples=rng.integers(0,len(d),size=(5000,len(d)))
        ci=np.quantile(d[samples].mean(1)*100,[.025,.975])
        p=float(wilcoxon(d).pvalue) if not np.allclose(d,0) else 1.
        rows.append(dict(a=a,b=b,n_subjects=len(d),mean_delta_pp=float(d.mean()*100),
                         ci95_low_pp=float(ci[0]),ci95_high_pp=float(ci[1]),p_wilcoxon=p))
    res=pd.DataFrame(rows)
    # Holm familywise correction on all prespecified selector contrasts.
    order=np.argsort(res.p_wilcoxon.to_numpy())
    adj=np.empty(len(res));m=len(res);running=0.
    for pos,i in enumerate(order):
        running=max(running,min(1.,(m-pos)*res.p_wilcoxon.iloc[i]))
        adj[i]=running
    res["holm_p"]=adj
    return res,subj

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",choices=["physionet","uci"],required=True)
    parser.add_argument("--features",help="PhysioNet feature CSV")
    parser.add_argument("--splits",help="PhysioNet original 10-fold subjects CSV")
    parser.add_argument("--uci-zip",help="Official UCI ZIP")
    parser.add_argument("--out",required=True)
    parser.add_argument("--epochs",type=int,default=65)
    parser.add_argument("--seeds",nargs="+",type=int,default=[24701,24702])
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.data=="physionet":df,X,y,folds=get_physionet(args.features,args.splits)
    else:df,X,y,folds=get_uci(args.uci_zip)
    test_ids=np.concatenate([te for _,_,te in folds])
    assert len(np.unique(test_ids))==len(test_ids)
    counts=[]
    for i,(tr,va,te) in enumerate(folds,1):
        assert max(tr)<min(va) and max(va)<min(te) if args.data=="uci" else True
        assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
        for partition,ids in (("train",tr),("val",va),("test",te)):
            cnt=np.bincount(y[ids],minlength=2)
            if min(cnt)==0:raise ValueError(f"fold {i} {partition}: only one class")
        counts.append(dict(fold=i,n_train=len(tr),n_validation=len(va),n_test=len(te),
                           test_pos_rate=float(y[te].mean())))
    pd.DataFrame(counts).to_csv(out/"split_checks.csv",index=False)
    if args.smoke:
        print("VALIDATED-FOLDS",len(folds),"TOTAL-TEST",len(test_ids),flush=True)
        return
    allfold=[];allselected=[];allpred=[]
    for i,(tr,va,te) in enumerate(folds,1):
        scale=StandardScaler().fit(X[tr])
        Xtr=scale.transform(X[tr]).astype(np.float32)
        Xva=scale.transform(X[va]).astype(np.float32)
        Xte=scale.transform(X[te]).astype(np.float32)
        for seed in args.seeds:
            actual_seed=seed+i*100
            mi=np.argsort(-mutual_info_classif(Xtr,y[tr],random_state=actual_seed),
                          kind="stable")
            for variant in VARIANTS:
                start=time.monotonic()
                model,select=learn(Xtr,y[tr],Xva,y[va],variant,mi,actual_seed,args.epochs)
                with torch.inference_mode():
                    p=torch.cat([model(b).softmax(1)[:,1] for b in torch.from_numpy(Xte).split(512)]).numpy()
                foldmetrics,preds=metrics(y[te],p)
                row=dict(data=args.data,fold=i,seed=seed,variant=variant,
                         n_test=len(te),elapsed_s=time.monotonic()-start,**foldmetrics)
                allfold.append(row)
                allselected.append(dict(data=args.data,fold=i,seed=seed,variant=variant,**select))
                for j,ix in enumerate(te):
                    rec=dict(data=args.data,fold=i,seed=seed,variant=variant,
                             sample_index=int(ix),y_true=int(y[ix]),
                             y_pred=int(preds[j]),p_closed=float(p[j]))
                    if args.data=="physionet":rec["subject"]=int(df.subject.iloc[ix])
                    else:rec["original_sample_index"]=int(df.original_index.iloc[ix])
                    allpred.append(rec)
                print("RESULT",args.data,"fold",i,"seed",seed,variant,
                      "BA",round(foldmetrics["balanced_accuracy"],4),
                      "moved",select["moved_nodes"],"time_s",round(row["elapsed_s"],1),flush=True)
                pd.DataFrame(allfold).to_csv(out/"fold_metrics_partial.csv",index=False)
    f=pd.DataFrame(allfold);p=pd.DataFrame(allpred);s=pd.DataFrame(allselected)
    expected=len(test_ids)*len(args.seeds)*len(VARIANTS)
    if len(p)!=expected:raise RuntimeError(f"Missing predictions {len(p)} / {expected}")
    if p.duplicated(["variant","seed","sample_index"]).any():
        raise RuntimeError("Duplicate out-of-fold predictions")
    f.to_csv(out/"fold_metrics.csv",index=False)
    p.to_csv(out/"predictions.csv",index=False)
    s.to_csv(out/"selection_audit.csv",index=False)
    f.groupby("variant")[["balanced_accuracy","f1","auc"]].agg(["mean","std"]).to_csv(out/"performance_summary.csv")
    s.groupby("variant")[["moved_nodes","jaccard_initial_final","overlap_mi"]].agg(["mean","std"]).to_csv(out/"selector_summary.csv")
    if args.data=="physionet":
        stats,subjects=paired_subject_stats(p)
        stats.to_csv(out/"subject_paired_tests.csv",index=False)
        subjects.to_csv(out/"subject_metrics.csv",index=False)
    metadata=dict(data=args.data,fold_count=len(folds),n_subjects=109 if args.data=="physionet" else 1,
                  n_obs=len(df),n_test_per_variant_seed=len(test_ids),seeds=args.seeds,
                  k=8,variants=VARIANTS,max_epochs=args.epochs,
                  gates="Gumbel stochastic train-only exploration in noisy variants; deterministic test Top-K",
                  main_comparison="MI learned vs MI noisy-fixed; random learned vs random noisy-fixed",
                  limitation="Exploratory model development informed by prior tests; not independent confirmatory study.",
                  uci_note="Single session; time folds are correlated; no subject-level statistical significance." if args.data=="uci" else
                           "R01/R02 label perfectly confounded with recording run.")
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (out/"RUN_COMPLETE.txt").write_text("REAL DATA SELECTOR AUDIT\n",encoding="utf-8")
    print("AUDIT-COMPLETE",args.data,"n_predictions",len(p),flush=True)
    print(f.groupby("variant")[["balanced_accuracy","f1","auc"]].mean().to_string(),flush=True)
    print(s.groupby("variant")[["moved_nodes","jaccard_initial_final","overlap_mi"]].mean().to_string(),flush=True)
if __name__=="__main__":
    main()

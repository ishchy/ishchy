#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict forward-in-time 3-fold UCI EEG Eye State evaluation (exploratory).

Only one continuous recording/participant exists, so no cross-subject claim.
For each fold: chronological training -> purge -> validation -> purge -> test.
Test intervals cover the final half of the recording with no overlap.
Train-only scaler/MI; validation-only early stopping and SVM C selection.
All methods share exactly the same input time partitions, no test optimization.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import ffseeg_repro_experiments as base
import physionet_sparse_residual as sparse

MODELS = ["FullGraphConv", "FFSEEG-GNN", "FFSEEG-Residual", "FFSEEG-NoResidual",
          "MLP", "MI-SVM-8", "RBF-SVM-14"]
DEFAULT_PURGE = 256
DEFAULT_VAL = 1250

def load_data(archive: Path):
    raw_bytes = archive.read_bytes()
    data_hash = hashlib.sha256(raw_bytes).hexdigest()
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        names = [s for s in z.namelist() if s.lower().endswith(".arff")]
        if len(names) != 1:
            raise RuntimeError(f"Expected exactly one ARFF in UCI ZIP; got {names}")
        arff_bytes = z.read(names[0])
    raw, _ = arff.loadarff(io.BytesIO(arff_bytes))
    frame = pd.DataFrame(raw)
    for name in frame.columns:
        if frame[name].dtype == object:
            frame[name] = frame[name].map(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    frame = base._normalize_columns(frame)
    if len(frame) != 14980:
        raise ValueError(f"Expected 14980 chronological samples, observed {len(frame)}")
    if set(frame.eyeDetection.unique()) != {0, 1}:
        raise ValueError("Expected two target classes 0 and 1")
    frame["original_index"] = np.arange(len(frame), dtype=int)
    removed = [q-1 for q in base.OUTLIER_ROWS_1BASED]
    frame = frame.drop(index=removed).reset_index(drop=True)
    x = frame[base.CHANNELS].to_numpy(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("EEG inputs contain non-finite values")
    return frame, {
        "dataset": "UCI EEG Eye State, dataset ID 264",
        "dataset_doi": "10.24432/C57G7J",
        "source_zip_sha256": data_hash,
        "arff_member": names[0],
        "original_rows": 14980,
        "removed_original_row_numbers_1_based": base.OUTLIER_ROWS_1BASED,
        "cleaned_rows": len(frame),
        "one_continuous_subject": True,
    }

def make_forward_folds(n: int, purge: int, val_size: int):
    cut = [n//2, (2*n)//3, (5*n)//6, n]
    out = []
    for j in range(3):
        test_start, test_end = cut[j:j+2]
        val_end = test_start - purge
        val_start = val_end - val_size
        train_end = val_start - purge
        if train_end <= 1000:
            raise ValueError("Training split too small")
        tr = np.arange(0, train_end, dtype=int)
        va = np.arange(val_start, val_end, dtype=int)
        te = np.arange(test_start, test_end, dtype=int)
        assert tr[-1] + purge < va[0] and va[-1] + purge < te[0]
        assert len(set(tr)&set(va)) == len(set(tr)&set(te)) == len(set(va)&set(te)) == 0
        out.append((tr, va, te))
    assert sum(len(te) for _,_,te in out)==n-n//2
    assert len(set(np.concatenate([te for _,_,te in out]))) == n - n//2
    return out

def summarize(y, prob):
    pred=(np.asarray(prob)>=.5).astype(int)
    return dict(accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                f1=float(f1_score(y,pred,zero_division=0)),
                auc=float(roc_auc_score(y,prob))),pred

def model_predict(name, split, A_norm, cfg, seed):
    xtr,ytr,xva,yva,xte = split
    device = torch.device("cpu")
    base.seed_everything(seed)
    t0=time.monotonic()
    selection=""
    parameters=0
    valid_score=None
    if name == "FullGraphConv":
        model=base.FullGraphConv(A_norm,dropout=.10)
        model=base._train_generic(model,xtr,ytr,xva,yva,cfg,device,lr=cfg.lr_fullgcn)
    elif name == "FFSEEG-GNN":
        model, indices, extra=base.train_ffseeg(A_norm,xtr,ytr,xva,yva,cfg,device)
        selection=",".join(base.CHANNELS[i] for i in indices)
        valid_score=float(extra["soft_best_val_f1"])
    elif name in ("FFSEEG-Residual","FFSEEG-NoResidual"):
        model, info=sparse.fit(xtr,ytr,xva,yva,seed,
                             residual=(name=="FFSEEG-Residual"),k=8,
                             epochs=cfg.max_epochs,dropout=.2,lr=.002,
                             weight_decay=.002)
        selection=info["selected"]
        valid_score=float(info["val_ba"])
    elif name == "MLP":
        model=base.MLP(dropout=.10)
        model=base._train_generic(model,xtr,ytr,xva,yva,cfg,device,lr=cfg.lr_default)
    elif name in ("MI-SVM-8","RBF-SVM-14"):
        if name=="MI-SVM-8":
            mi=mutual_info_classif(xtr,ytr,random_state=seed)
            cols=np.argsort(-mi,kind="stable")[:8]
            selection=",".join(base.CHANNELS[i] for i in cols)
        else:
            cols=np.arange(len(base.CHANNELS))
        best=None
        for c in (.1,1.,10.):
            svc=SVC(C=c,kernel="rbf",gamma="scale")
            svc.fit(xtr[:,cols],ytr)
            val_prob = 1/(1+np.exp(-np.clip(svc.decision_function(xva[:,cols]),-30,30)))
            val_ba=balanced_accuracy_score(yva,val_prob>=.5)
            if best is None or val_ba > best[0]+1e-12:
                best=(val_ba,c,svc)
        valid_score=float(best[0])
        scores=best[2].decision_function(xte[:,cols])
        prob=1/(1+np.exp(-np.clip(scores,-30,30)))
        return prob, dict(train_seconds=round(time.monotonic()-t0,2),selected=selection,
                           n_params=None,valid_score=valid_score,svm_C=best[1])
    else:
        raise ValueError(name)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.inference_mode():
        if name in ("FFSEEG-Residual", "FFSEEG-NoResidual"):
            prob=torch.cat([model(torch.from_numpy(b).float()).softmax(1)[:,1]
                            for b in np.array_split(xte,max(1,(len(xte)+255)//256))]).numpy()
        else:
            logits=base.infer_logits(model,xte,256,device)
            prob=torch.softmax(torch.from_numpy(logits),dim=1).numpy()[:,1]
    return prob, dict(train_seconds=round(time.monotonic()-t0,2),selected=selection,
                       n_params=parameters,valid_score=valid_score,svm_C=None)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-zip",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--seed",type=int,default=24701)
    p.add_argument("--purge",type=int,default=DEFAULT_PURGE)
    p.add_argument("--val-size",type=int,default=DEFAULT_VAL)
    p.add_argument("--smoke",action="store_true",
                   help="Check data and folds only; creates no empirical model predictions")
    args=p.parse_args()
    torch.set_num_threads(1)
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    df,meta=load_data(Path(args.dataset_zip))
    folds=make_forward_folds(len(df),args.purge,args.val_size)
    y=df.eyeDetection.to_numpy(np.int64)
    X=df[base.CHANNELS].to_numpy(np.float32)
    splits=[]
    for fold,(tr,va,te) in enumerate(folds,1):
        counts={}
        for key,ids in (("train",tr),("val",va),("test",te)):
            n0,n1=np.bincount(y[ids],minlength=2)
            if min(n0,n1)==0:
                raise ValueError(f"fold {fold} {key}: missing class, {n0=} {n1=}")
            counts[key+"_n"]=len(ids);counts[key+"_class0"]=int(n0);counts[key+"_class1"]=int(n1)
            counts[key+"_first_original_index"]=int(df.original_index.iloc[ids[0]])
            counts[key+"_last_original_index"]=int(df.original_index.iloc[ids[-1]])
        splits.append(dict(fold=fold,**counts))
    meta.update({"fold_count":3,"split_strategy":"forward chaining: train < validation < test",
                 "test_portion":"chronological last 50% in three nonoverlapping blocks",
                 "purge_cleaned_rows_each_boundary":args.purge,
                 "validation_cleaned_rows_per_fold":args.val_size,
                 "seed":args.seed,"max_epochs":args.epochs,"models":MODELS,
                 "caveats":["Only one person / session, no cross-subject inference.",
                            "Temporal concept drift can affect performance.",
                            "Blocks/folds from the same recording are not independent study participants.",
                            "Previously seen PhysioNet metrics motivated this candidate architecture."]})
    (dest/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(splits).to_csv(dest/"split_audit.csv",index=False)
    if args.smoke:
        print("DATA-AND-SPLIT-SMOKE-OK",len(df),pd.DataFrame(splits).to_string(index=False),flush=True)
        return
    A,A_norm,_ = base.build_adjacency(torch.device("cpu"))
    cfg=base.TrainConfig(max_epochs=args.epochs,k=8,batch_size=256,check_every=5,patience=10)
    metrics=[];predictions=[]
    for fold,(tr,va,te) in enumerate(folds,1):
        scaler=StandardScaler().fit(X[tr])
        xtr=scaler.transform(X[tr]).astype(np.float32)
        xva=scaler.transform(X[va]).astype(np.float32)
        xte=scaler.transform(X[te]).astype(np.float32)
        split=(xtr,y[tr],xva,y[va],xte)
        for m in MODELS:
            prob,details=model_predict(m,split,A_norm,cfg,args.seed+fold*100)
            row,pred=summarize(y[te],prob)
            row.update(dict(model=m,fold=fold,train_n=len(tr),validation_n=len(va),
                            test_n=len(te),test_class1_rate=float(y[te].mean()),**details))
            metrics.append(row)
            for j,i in enumerate(te):
                predictions.append(dict(model=m,fold=fold,original_sample_index=int(df.original_index.iloc[i]),
                                        y_true=int(y[i]),y_pred=int(pred[j]),prob_closed=float(prob[j])))
            pd.DataFrame(metrics).to_csv(dest/"fold_metrics_partial.csv",index=False)
            print(f"fold={fold} model={m} BA={row['balanced_accuracy']:.4f} "
                  f"F1={row['f1']:.4f} AUC={row['auc']:.4f} "
                  f"train_seconds={details['train_seconds']}",flush=True)
    pred=pd.DataFrame(predictions)
    expected_n=sum(len(te) for _,_,te in folds)
    assert len(pred)==len(MODELS)*expected_n
    for m,g in pred.groupby("model"):
        assert len(g)==expected_n and not g.original_sample_index.duplicated().any()
    pred.to_csv(dest/"predictions.csv",index=False)
    pd.DataFrame(metrics).to_csv(dest/"fold_metrics.csv",index=False)
    summary=pd.DataFrame(metrics).groupby("model")[["balanced_accuracy","f1","auc","train_seconds"]].agg(["mean","std"])
    summary.to_csv(dest/"summary.csv")
    pivot=pd.DataFrame(metrics).pivot(index="fold",columns="model",values="balanced_accuracy")
    diffs=pivot.sub(pivot["FFSEEG-Residual"],axis=0)*-100.
    diffs.to_csv(dest/"paired_fold_differences_pp.csv")
    (dest/"RUN_COMPLETE.txt").write_text("REAL UCI EEG / CHRONOLOGICAL 3-FOLD TEST\n",encoding="utf-8")
    print("FINISHED UCI CHRONOLOGICAL REAL DATA",summary.to_string(),flush=True)

if __name__=="__main__":
    main()

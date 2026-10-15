#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-registered pilot: within-run left-versus-right motor imagery.

Independent classification TASK from the earlier R01/R02 eye-state experiment:
PhysioNet EEGMMIDB RUN R04. The cue annotations T1 and T2 identify left-
and right-fist imagery inside the SAME run for each subject. T0 rest is excluded.
Event-locked 0.5-2.5 second epochs, 14 motor-area electrodes, log relative
mu and beta spectral power (2 per node), four disjoint-subject outer folds.

Preprocessing, channel ranking and all model training use TRAIN participants;
checkpoint/C selection uses VALIDATION participants; outer-test participants
are untouched until predictions. No test-informed adjustments. Exploratory
pilot: cannot claim independent full-cohort confirmation or robustness.

Models: FullGraph14, MI-InputGraph8, LearnedTopK8, RBF-SVM28, MLP28.
The learned selector still encodes 14 graph nodes, so no computed FLOP savings
are claimed. MI-InputGraph8 genuinely runs an 8-node induced subgraph.

Scientific safety: no fabricated results, labels or test edits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PHYSIONET="https://physionet.org/files/eegmmidb/1.0.0"
CHANNELS=["FC5","FC3","FC1","FC2","FC4","FC6",
          "C5","C3","C1","C2","C4","C6","CP3","CP4"]
RUN=4
MODELS=["FullGraph14","MI-InputGraph8","LearnedTopK8","RBF-SVM28","MLP28"]
MU=(8.,13.)
BETA=(13.,30.)
TOTAL=(4.,40.)

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def fetch_edf(subject, cache, tries=7):
    s=f"S{subject:03d}"
    name=f"{s}R{RUN:02d}.edf"
    dest=cache/s/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size>100000:
        return dest
    url=f"{PHYSIONET}/{s}/{name}"
    for i in range(tries):
        try:
            request=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 (EEG academic experiment)"})
            temp=dest.with_suffix(".edf.tmp")
            with urllib.request.urlopen(request,timeout=90) as remote,open(temp,"wb") as f:
                while True:
                    chunk=remote.read(2**20)
                    if not chunk:break
                    f.write(chunk)
            if temp.stat().st_size < 100000:
                raise OSError("EDF downloaded unexpectedly small")
            temp.replace(dest)
            return dest
        except Exception as ex:
            if "temp" in locals():temp.unlink(missing_ok=True)
            print("DOWNLOAD-RETRY",subject,i+1,repr(ex),flush=True)
            if i==tries-1:raise
            time.sleep(min(20,2*(i+1)))
    raise RuntimeError("Unreachable")

def normalize_label(s):
    return str(s).upper().replace(".","").replace(" ","").replace("-REF","")

def epoch_features(epochs, sfreq):
    # [epoch,channel,time] -> [epoch,node,mu/beta].
    n=epochs.shape[-1]
    data=epochs.astype(np.float64)
    data=data-data.mean(-1,keepdims=True)
    fft=np.fft.rfft(data*np.hanning(n)[None,None,:],axis=-1)
    power=np.abs(fft)**2
    hz=np.fft.rfftfreq(n,1.0/sfreq)
    total=power[:,:,(hz>=TOTAL[0]) & (hz<TOTAL[1])].sum(axis=-1)
    eps=1e-18
    out=[]
    for lo,hi in (MU,BETA):
        band=power[:,:,(hz>=lo) & (hz<hi)].sum(axis=-1)
        out.append(np.log((band+eps)/(total+eps)))
    X=np.stack(out,axis=-1).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite power features")
    return X

def extract_data(cache,subjects):
    import mne
    rows=[]
    audit=[]
    feature_columns=[f"{ch}_{band}" for ch in CHANNELS for band in ("mu","beta")]
    for subject in subjects:
        path=fetch_edf(subject,cache)
        raw=mne.io.read_raw_edf(str(path),preload=False,verbose="ERROR")
        fs=float(raw.info["sfreq"])
        if not np.isclose(fs,160.,atol=1e-3):
            raise ValueError(f"Unexpected sampling rate for S{subject}: {fs}")
        mapping={normalize_label(c):c for c in raw.ch_names}
        absent=[s for s in CHANNELS if s not in mapping]
        if absent:
            raise ValueError(f"Missing sensorimotor EEG channels for S{subject}: {absent}")
        events,event_id=mne.events_from_annotations(raw,
            event_id={"T1":1,"T2":2},verbose="ERROR")
        if set(events[:,2])!={1,2}:
            raise ValueError(f"Subject S{subject} missing T1 or T2, ids={event_id}")
        picks=[mapping[s] for s in CHANNELS]
        epoch=mne.Epochs(raw,events,event_id={"T1":1,"T2":2},
                         tmin=.5,tmax=2.5,baseline=None,detrend=1,
                         picks=picks,preload=True,reject_by_annotation=True,
                         verbose="ERROR")
        trials=epoch.get_data()
        labels=(epoch.events[:,2]==2).astype(int)
        if (trials.ndim!=3 or trials.shape[1]!=14 or
                len(labels)!=len(trials)):
            raise RuntimeError("Epoch extraction shape mismatch")
        if min(np.bincount(labels,minlength=2))<3:
            raise ValueError(f"Subject S{subject} too few T1/T2 events: {np.bincount(labels,minlength=2)}")
        X=epoch_features(trials,fs)
        for j,(feat,label) in enumerate(zip(X,labels)):
            row={"subject":int(subject),"run":RUN,"event_index":j,
                 "event_sample":int(epoch.events[j,0]),"label":int(label)}
            row.update({col:float(x) for col,x in zip(feature_columns,feat.reshape(-1))})
            rows.append(row)
        audit.append({"subject":int(subject),"run":RUN,"n_epoch":len(trials),
                      "n_T1":int((labels==0).sum()),"n_T2":int((labels==1).sum()),
                      "fs":fs,"edf_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "event_labels_T1_T2_in_same_run":True})
        print("EXTRACTED",subject,"left",int((labels==0).sum()),
              "right",int((labels==1).sum()),flush=True)
    df=pd.DataFrame(rows)
    if df.duplicated(["subject","run","event_index"]).any():raise ValueError("Duplicate events")
    if df.subject.nunique()!=len(subjects):raise ValueError("Missing subjects")
    return df,pd.DataFrame(audit),feature_columns

def geometry_adjacency():
    import mne
    positions=mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    pts=np.stack([positions[c] for c in CHANNELS])
    dist=np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2)
    A=np.eye(len(CHANNELS),dtype=np.float32)
    for i in range(len(CHANNELS)):
        for j in np.argsort(dist[i])[1:4]:
            A[i,j]=A[j,i]=1.
    return A

def normalized_adj(A):
    d=A.sum(1).clip(1)
    return torch.tensor(A/(np.sqrt(d[:,None]*d[None,:])),dtype=torch.float32)

def pick_mi_electrodes(X,y,seed,k=8):
    scores=mutual_info_classif(X,y,random_state=seed)
    scores=scores.reshape(14,2).mean(axis=1)
    return np.argsort(-scores,kind="stable")[:k].tolist()

class GraphLayer(nn.Module):
    def __init__(self,din,dout,adj):
        super().__init__()
        self.self_lin=nn.Linear(din,dout,bias=False)
        self.neigh_lin=nn.Linear(din,dout,bias=False)
        self.bias=nn.Parameter(torch.zeros(dout))
        self.register_buffer("adj",adj)
    def forward(self,x):
        return self.self_lin(x)+self.neigh_lin(torch.einsum("ij,bjd->bid",self.adj,x))+self.bias

class NodeGraph(nn.Module):
    def __init__(self,adj,n_nodes,dropout=.2,learn_selector=False,k=8):
        super().__init__()
        self.learn_selector=learn_selector
        self.k=k
        self.g1=GraphLayer(2,24,adj)
        self.g2=GraphLayer(24,16,adj)
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Sequential(nn.Linear(n_nodes*16,48),nn.ReLU(),
                                nn.Dropout(dropout),nn.Linear(48,2))
        if learn_selector:
            self.score=nn.Parameter(torch.randn(n_nodes)*.025)
            self.tau=.8
    def forward(self,x):
        if self.learn_selector:
            logits=self.score
            if self.training:
                random_noise=torch.rand_like(logits).clamp(1e-6,1-1e-6)
                logits=logits+.15*(-torch.log(-torch.log(random_noise)))
            discrete=torch.zeros_like(logits).scatter_(0,torch.topk(logits,self.k).indices,1.)
            continuous=self.k*torch.softmax(self.score/max(.3,self.tau),dim=0)
            mask=discrete+continuous-continuous.detach()
            x=x*mask[None,:,None]
        h=self.dropout(F.relu(self.g1(x)))
        h=F.relu(self.g2(h))
        return self.head(h.flatten(1))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=nn.Sequential(nn.Flatten(),nn.Linear(28,64),nn.ReLU(),
                               nn.Dropout(.2),nn.Linear(64,32),nn.ReLU(),
                               nn.Linear(32,2))
    def forward(self,x):return self.seq(x)

def infer(model,X):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(part).softmax(1)[:,1] for part in
                          torch.from_numpy(X).float().split(256)]).numpy()

def fit_network(name,xtr,ytr,xva,yva,fullA,select,seed,epochs):
    seed_everything(seed)
    if name=="MLP28":
        model=MLP()
    elif name=="MI-InputGraph8":
        idx=np.asarray(select)
        sub=fullA[np.ix_(idx,idx)]
        model=NodeGraph(normalized_adj(sub),len(idx),dropout=.2)
        xtr=xtr[:,idx,:].copy()
        xva=xva[:,idx,:].copy()
    else:
        model=NodeGraph(normalized_adj(fullA),14,dropout=.2,
                        learn_selector=(name=="LearnedTopK8"))
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    yt=torch.from_numpy(ytr.astype(np.int64))
    X=torch.from_numpy(xtr).float()
    best_val=-np.inf;best_state=None;best_epoch=0;bad=0
    for epoch in range(1,epochs+1):
        model.train()
        if name=="LearnedTopK8":model.tau=max(.3,.8*.985**(epoch-1))
        for ids in torch.randperm(len(X)).split(64):
            logits=model(X[ids])
            loss=F.cross_entropy(logits,yt[ids])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optim.step()
        if epoch%5==0:
            metric=balanced_accuracy_score(yva,infer(model,xva)>=.5)
            if metric>best_val+1e-8:
                best_val=metric;best_epoch=epoch;bad=0
                best_state={name:value.detach().clone() for name,value in model.state_dict().items()}
            else:
                bad+=1
            if bad>=10:break
    if best_state is None:raise ValueError("No validation checkpoint")
    model.load_state_dict(best_state)
    return model,dict(val_ba=float(best_val),best_epoch=best_epoch,
                      n_params=sum(p.numel() for p in model.parameters() if p.requires_grad))

def model_predict(name,xtr,ytr,xva,yva,xte,adj,mi_select,seed,epochs):
    if name=="RBF-SVM28":
        train=xtr.reshape(len(xtr),-1)
        val=xva.reshape(len(xva),-1)
        test=xte.reshape(len(xte),-1)
        candidate=None
        for C in (.1,1.,10.):
            svc=SVC(C=C,gamma="scale",kernel="rbf",class_weight="balanced")
            svc.fit(train,ytr)
            score=svc.decision_function(val)
            ba=balanced_accuracy_score(yva,score>=0)
            if candidate is None or ba>candidate[0]:
                candidate=(ba,C,svc)
        svc=candidate[2]
        prob=1/(1+np.exp(-np.clip(svc.decision_function(test),-30,30)))
        return prob,dict(val_ba=float(candidate[0]),best_epoch=None,n_params=None,
                         svm_C=float(candidate[1]),selected="")
    net,meta=fit_network(name,xtr,ytr,xva,yva,adj,mi_select,seed,epochs)
    if name=="MI-InputGraph8":
        xte=xte[:,mi_select,:]
        meta["selected"]=",".join(CHANNELS[i] for i in mi_select)
    elif name=="LearnedTopK8":
        selected=torch.topk(net.score.detach(),8).indices.tolist()
        meta["selected"]=",".join(CHANNELS[i] for i in selected)
        meta["changed_vs_train_mi"]=int(len(set(selected)-set(mi_select)))
    else:
        meta["selected"]=""
    prob=infer(net,xte)
    return prob,meta

def evaluate(y,p):
    pred=(p>=.5).astype(int)
    return dict(ba=float(balanced_accuracy_score(y,pred)),
                f1=float(f1_score(y,pred,zero_division=0)),
                auc=float(roc_auc_score(y,p)),
                accuracy=float(accuracy_score(y,pred))),pred

def run(df,features,out,n_folds=4,epochs=65,seed=34179,smoke=False):
    torch.set_num_threads(1)
    X=df[features].to_numpy(np.float32).reshape(len(df),14,2)
    y=df.label.to_numpy(np.int64)
    subjects=np.sort(df.subject.unique())
    if len(subjects)<8:raise ValueError("At least 8 subjects needed")
    folds=KFold(n_splits=n_folds,shuffle=True,random_state=seed)
    graph=geometry_adjacency()
    partition_audit=[];fold_results=[];predictions=[]
    for fold,(dev,test) in enumerate(folds.split(subjects),1):
        development=subjects[dev].copy()
        tests=subjects[test]
        rng=np.random.default_rng(seed+fold)
        rng.shuffle(development)
        nval=max(2,round(.20*len(development)))
        validations=development[:nval]
        training=development[nval:]
        if set(training)&set(validations) or set(training)&set(tests) or set(validations)&set(tests):
            raise ValueError("Overlapping subject splits")
        tr=df.subject.isin(training).to_numpy()
        va=df.subject.isin(validations).to_numpy()
        te=df.subject.isin(tests).to_numpy()
        for part,mask in (("train",tr),("validation",va),("test",te)):
            if min(np.bincount(y[mask],minlength=2))<2:
                raise ValueError(f"Fold {fold} {part} lacks either class")
        partition_audit.append(dict(fold=fold,n_train=int(tr.sum()),n_validation=int(va.sum()),
                                    n_test=int(te.sum()),train_subjects=",".join(map(str,sorted(training))),
                                    validation_subjects=",".join(map(str,sorted(validations))),
                                    test_subjects=",".join(map(str,sorted(tests))),
                                    train_pos_rate=float(y[tr].mean()),
                                    validation_pos_rate=float(y[va].mean()),
                                    test_pos_rate=float(y[te].mean())))
        if smoke:continue
        scaler=StandardScaler().fit(X[tr].reshape(int(tr.sum()),-1))
        def standard(mask):
            return scaler.transform(X[mask].reshape(int(mask.sum()),-1)).astype(np.float32).reshape(-1,14,2)
        xtr,xva,xte=(standard(mask) for mask in (tr,va,te))
        select=pick_mi_electrodes(xtr.reshape(len(xtr),-1),y[tr],seed+fold)
        print("FOLD",fold,"ntrain",tr.sum(),"nval",va.sum(),"ntest",te.sum(),flush=True)
        for name in MODELS:
            start=time.monotonic()
            prob,meta=model_predict(name,xtr,y[tr],xva,y[va],xte,graph,select,
                                    seed=seed+fold*100,epochs=epochs)
            scores,pred=evaluate(y[te],prob)
            record=dict(fold=fold,model=name,n_test=int(te.sum()),n_train=int(tr.sum()),
                        duration_s=time.monotonic()-start,**scores,**meta)
            fold_results.append(record)
            for j,i in enumerate(np.flatnonzero(te)):
                predictions.append(dict(fold=fold,model=name,subject=int(df.subject.iloc[i]),
                                        run=RUN,event_index=int(df.event_index.iloc[i]),
                                        y_true=int(y[i]),y_pred=int(pred[j]),
                                        p_right=float(prob[j])))
            print("MODEL",name,"fold",fold,"BA",round(scores["ba"],4),
                  "F1",round(scores["f1"],4),"AUC",round(scores["auc"],4),flush=True)
        pd.DataFrame(fold_results).to_csv(out/"PARTIAL_fold_metrics.csv",index=False)
    audit=pd.DataFrame(partition_audit)
    audit.to_csv(out/"split_audit.csv",index=False)
    if smoke:
        print("PILOT-PREFLIGHT-OK",audit.to_string(index=False),flush=True)
        return
    folds_df=pd.DataFrame(fold_results)
    pred=pd.DataFrame(predictions)
    if len(pred)!=len(df)*len(MODELS):raise ValueError("Missing outer-test predictions")
    if pred.duplicated(["model","subject","run","event_index"]).any():
        raise ValueError("Duplicate outer-test prediction")
    folds_df.to_csv(out/"fold_metrics.csv",index=False)
    pred.to_csv(out/"predictions.csv",index=False)
    folds_df.groupby("model")[["ba","f1","auc","accuracy"]].agg(["mean","std"]).to_csv(out/"fold_summary.csv")
    subj=[]
    for (name,s),group in pred.groupby(["model","subject"]):
        if set(group.y_true.unique())!={0,1}:
            raise ValueError(f"{s} doesn't have two task labels")
        metric,_=evaluate(group.y_true.to_numpy(),group.p_right.to_numpy())
        subj.append(dict(model=name,subject=int(s),**metric))
    pd.DataFrame(subj).to_csv(out/"subject_metrics.csv",index=False)
    # Pilot exploratory paired subject tests; never treat epochs as independent people.
    piv=pd.DataFrame(subj).pivot(index="subject",columns="model",values="ba")
    rows=[];rng=np.random.default_rng(20261015)
    for other in MODELS:
        if other=="LearnedTopK8":continue
        delta=(piv["LearnedTopK8"]-piv[other]).to_numpy()
        samples=rng.integers(0,len(delta),size=(5000,len(delta)))
        ci=np.quantile(delta[samples].mean(1)*100,[.025,.975])
        try:p=float(wilcoxon(delta).pvalue)
        except Exception:p=1.
        rows.append(dict(reference="LearnedTopK8",comparison=other,
                         n_subjects=len(delta),mean_delta_pp=float(delta.mean()*100),
                         ci_low_pp=float(ci[0]),ci_high_pp=float(ci[1]),unadjusted_p=p))
    stats=pd.DataFrame(rows)
    sorted_idx=np.argsort(stats.unadjusted_p.to_numpy())
    adjusted=np.empty(len(stats))
    running=0.
    for order,i in enumerate(sorted_idx):
        running=max(running,(len(stats)-order)*float(stats.unadjusted_p.iloc[i]))
        adjusted[i]=min(1.,running)
    stats["holm_p"]=adjusted
    stats.to_csv(out/"paired_subject_exploratory.csv",index=False)
    (out/"PARTIAL_fold_metrics.csv").unlink(missing_ok=True)
    (out/"RUN_COMPLETE.txt").write_text("REAL EDF WITHIN-RUN MOTOR IMAGERY TASK; PILOT\n")
    print("REAL MOTOR IMAGERY PILOT FINISHED",flush=True)
    print(folds_df.groupby("model")[["ba","f1","auc"]].mean().to_string(),flush=True)
    print(stats.to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--subject-end",type=int,default=24)
    p.add_argument("--folds",type=int,default=4)
    p.add_argument("--epochs",type=int,default=65)
    p.add_argument("--cache-dir",default="cache/eegmmidb")
    p.add_argument("--out",default="outputs/motor_imagery")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if not 8<=args.subject_end<=109:raise ValueError("subject-end must be 8..109")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df,audit,features=extract_data(Path(args.cache_dir),range(1,args.subject_end+1))
    df.to_csv(out/"motor_imagery_real_features.csv",index=False)
    audit.to_csv(out/"edf_event_audit.csv",index=False)
    metadata=dict(task="left_vs_right_fist_motor_imagery_T1_T2_same_run",
        physionet_url="https://physionet.org/content/eegmmidb/1.0.0/",
        run=RUN,n_subjects=int(df.subject.nunique()),n_epochs=len(df),
        n_channels=14,channel_order=CHANNELS,feature_bands=dict(mu=MU,beta=BETA,total=TOTAL),
        event_epoch_seconds=[.5,2.5],sampling_rate=160,n_splits=args.folds,
        models=MODELS,k_sparse=8,random_seed=34179,
        leakage_controls="subject-disjoint train/validation/test, train-only scaler and MI, validation-only SVM C and checkpoints",
        prior_data="earlier R01/R02 eye-state task differs in both EEG task and label semantics",
        scientific_status="pre-registered exploratory PILOT; NOT a final independent multi-cohort confirmation",
        efficiency_note="LearnedTopK8 masks 14 nodes; it does not reduce GCN node count; MI-InputGraph8 is genuine 8-node input")
    (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    run(df,features,out,args.folds,args.epochs,smoke=args.smoke)

if __name__=="__main__":main()

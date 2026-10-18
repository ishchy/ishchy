#!/usr/bin/env python3
"""Real EEGMMIDB R04/R08/R12 motor imagery, subject-held-out exploratory study.

Labels T1/T2 always denote LEFT/RIGHT imagined fist WITHIN EACH run.
Subject-disjoint train/validation/test splits; no window leakage.
All preprocessing, MI selection, spatial filters, covariance templates and
network weights are fitted on TRAIN participants only.
Predeclared models:
  FullGraph14, MI-InputGraph8, LearnedTopK8, MLP56,
  CSP-LDA14, FBCSP-LDA14, RiemannMDM14, EEGNet14.
T0 ignored. These 24-subject pilots are NOT independent confirmatory evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F

mne.set_log_level("ERROR")
BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
RUNS = (4, 8, 12)
CHANNELS = ("FC5","FC3","FC1","FC2","FC4","FC6",
            "C5","C3","C1","C2","C4","C6","CP3","CP4")
FEATURE_BANDS = ((4.,8.),(8.,13.),(13.,20.),(20.,30.))
CSP_BANDS = ((8.,12.),(12.,16.),(16.,22.),(22.,30.))
MODELS = ("FullGraph14","MI-InputGraph8","LearnedTopK8","MLP56",
          "CSP-LDA14","FBCSP-LDA14","RiemannMDM14","EEGNet14")
SFREQ = 160.
SAMPLES = 320
SEED = 34179

def fix_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# Prior successful subject-24 R04 EDF bytes from run 38740937352 are
# pinned as a provenance check. A mismatch is a source integrity error.
KNOWN_R04_HASHES=json.loads(
    Path(__file__).with_name("verified_r04_edf_hashes.json").read_text()
)["known_sha256"]

def download_edf(cache: Path, subject: int, run: int):
    code=f"S{subject:03d}"
    name=f"{code}R{run:02d}.edf"
    path=cache/code/name
    path.parent.mkdir(parents=True,exist_ok=True)
    expected=KNOWN_R04_HASHES.get(name)
    def valid_download(p):
        if not (p.is_file() and p.stat().st_size>100_000):
            return False
        if expected:
            observed=hashlib.sha256(p.read_bytes()).hexdigest()
            if observed!=expected:
                print("SOURCE_HASH_MISMATCH",name,"expected",expected,
                      "observed",observed,flush=True)
                return False
        return True
    if valid_download(path):
        return path
    path.unlink(missing_ok=True)
    url=f"{BASE_URL}/{code}/{name}"
    for attempt in range(6):
        try:
            request=urllib.request.Request(url, headers={
                "User-Agent":"Academic-EEG-CrossSubject-Pilot/1.0 (EEG analysis)"})
            tmp=path.with_suffix(".edf.partial")
            with urllib.request.urlopen(request,timeout=90) as response, tmp.open("wb") as out:
                if response.status != 200:raise IOError(f"HTTP {response.status}")
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    out.write(chunk)
            if not valid_download(tmp):
                raise IOError(f"EDF invalid or disagrees with prior verified SHA256: {name}")
            tmp.replace(path)
            print("DOWNLOAD_VERIFIED",name,path.stat().st_size,flush=True)
            return path
        except Exception as ex:
            tmp.unlink(missing_ok=True)
            print("DOWNLOAD-RETRY",subject,run,attempt+1,str(ex)[:200],flush=True)
            if attempt==5:raise
            time.sleep(2*(attempt+1))
    raise RuntimeError("No valid EDF")

def normalized(ch):
    return ch.upper().replace(".","").replace(" ","").replace("-REF","")

def signal_band(data,low,high):
    sos=butter(4,[low,high],btype="bandpass",fs=SFREQ,output="sos")
    return sosfiltfilt(sos,data,axis=-1).astype(np.float32)

def feature_bands(X):
    # X = EEG epochs (N,14,320) in microvolts after 4-40 Hz filtering.
    window=np.hanning(X.shape[-1]).astype(np.float32)
    spectrum=np.abs(np.fft.rfft(X*window[None,None,:],axis=-1))**2
    freq=np.fft.rfftfreq(X.shape[-1],1/SFREQ)
    total=spectrum[:,:, (freq>=4.)&(freq<40.)].sum(-1)
    feat=[]
    for low,high in FEATURE_BANDS:
        power=spectrum[:,:,(freq>=low)&(freq<high)].sum(-1)
        feat.append(np.log((power+1e-8)/(total+1e-8)))
    result=np.stack(feat,axis=-1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError("non-finite band-power feature")
    return result

def one_record(entry):
    subject,run,cache=entry
    edf=download_edf(cache,subject,run)
    raw=mne.io.read_raw_edf(str(edf),preload=False,verbose="ERROR")
    if not np.isclose(raw.info["sfreq"],SFREQ,atol=.001):
        raise ValueError(f"Unexpected sample rate {subject}/{run}: {raw.info['sfreq']}")
    available={normalized(name):name for name in raw.ch_names}
    missing=[ch for ch in CHANNELS if ch not in available]
    if missing:raise ValueError(f"Missing electrodes {subject}/{run}: {missing}")
    picks=[available[ch] for ch in CHANNELS]
    annotation_counts=dict(zip(*np.unique(np.asarray(raw.annotations.description,dtype=str),
                                         return_counts=True)))
    events,_=mne.events_from_annotations(raw,event_id={"T1":1,"T2":2},verbose="ERROR")
    observed=dict(zip(*np.unique(events[:,2],return_counts=True))) if len(events) else {}
    print("EVENT_AUDIT",subject,run,"annotations",annotation_counts,
          "T1_T2",observed,"n",len(events),flush=True)
    # Real EEGMMIDB records need not all contain 15 imagery cues. For example,
    # S013/R12 has four genuine T1/T2 events (2+2) in its original EDF.
    # Retain ALL verified recordings and ALL available event labels. No
    # resampling or padding of the short run is permitted.
    if len(events)<2 or set(events[:,2])!={1,2}:
        raise ValueError(
            f"EDF has invalid T1/T2 imagery annotations S{subject}/R{run}; "
            f"original_annotation_counts={annotation_counts}, mapped_counts={observed}, "
            f"file_sha256={hashlib.sha256(edf.read_bytes()).hexdigest()}")
    # Filtering within each recording only, never across subjects.
    signal_uv=raw.get_data(picks=picks).astype(np.float32)*1e6
    filtered=signal_band(signal_uv,4.,40.)
    raw.close()
    records=[]; labels=[]; starts=[]; dropped=0
    for ix,(sample,_,code) in enumerate(events):
        start=int(sample)+round(.5*SFREQ)
        stop=start+SAMPLES
        if stop>filtered.shape[-1]:
            dropped+=1
            continue
        labels.append(int(code)-1)
        records.append(filtered[:,start:stop].copy())
        starts.append(int(sample))
    y=np.asarray(labels,dtype=np.int64)
    if not records or np.min(np.bincount(y,minlength=2))<1:
        raise ValueError(f"Insufficient left/right events S{subject}/R{run}")
    waves=np.stack(records).astype(np.float32)
    if waves.shape[1:]!=(14,SAMPLES) or not np.isfinite(waves).all():
        raise ValueError(f"Invalid waveform shape S{subject}/R{run}")
    row=dict(subject=int(subject),run=int(run),n_events=int(len(y)),
             n_events_annotated=int(len(events)),
             original_T0_events=int(annotation_counts.get("T0",0)),
             original_T1_events=int(annotation_counts.get("T1",0)),
             original_T2_events=int(annotation_counts.get("T2",0)),
             is_short_run=bool(len(events)<15),
             n_left=int((y==0).sum()),n_right=int((y==1).sum()),
             events_out_of_bounds=int(dropped),
             edf_sha256=hashlib.sha256(edf.read_bytes()).hexdigest(),
             label_semantics="T1=left_mental_fist,T2=right_mental_fist_in_same_run")
    return subject,run,waves,y,np.asarray(starts),row

def acquire_dataset(subjects,cache,workers=4):
    input_items=[(int(s),int(run),cache) for s in subjects for run in RUNS]
    out=[]
    # Parallel HTTP transfers are fine, but EDF event parsing and MNE reads
    # are deliberately SERIAL to avoid concurrent-library state/annotation issues.
    # No subject or run may be silently skipped, even on source abnormalities.
    with ThreadPoolExecutor(max_workers=min(workers,4)) as pool:
        futures=[pool.submit(download_edf,cache,subject,run)
                 for subject,run,_ in input_items]
        for future in as_completed(futures):
            future.result()
    for entry in input_items:
        res=one_record(entry)
        out.append(res)
        print("EDF-OK",res[0],res[1],len(res[3]),
              "short_run",bool(res[-1]["is_short_run"]),flush=True)
    out.sort(key=lambda it:(it[0],it[1]))
    waves=np.concatenate([r[2] for r in out],axis=0)
    y=np.concatenate([r[3] for r in out],axis=0)
    frames=[]
    for s,run,wave,labs,onsets,audit in out:
        for idx,(lab,onset) in enumerate(zip(labs,onsets)):
            frames.append(dict(subject=s,run=run,event_index=idx,
                               event_sample=int(onset),label=int(lab)))
    table=pd.DataFrame(frames)
    checks=pd.DataFrame([r[-1] for r in out])
    if len(waves)!=len(table) or table.duplicated(["subject","run","event_index"]).any():
        raise RuntimeError("Epoch extraction row alignment invalid")
    if not (table.groupby(["subject","run"]).label.nunique()==2).all():
        raise RuntimeError("T1/T2 not both observed in every recording")
    if not (table.groupby("subject").run.nunique()==3).all():
        raise RuntimeError("Missing one of three runs for some subjects")
    return waves,y,table,checks

def make_adjacency():
    pts=mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    coords=np.stack([pts[ch] for ch in CHANNELS])
    distances=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=-1)
    A=np.eye(len(CHANNELS),dtype=np.float32)
    for i in range(len(CHANNELS)):
        for j in np.argsort(distances[i])[1:4]:
            A[i,j]=A[j,i]=1.
    return A

def norm_adj(A):
    degree=np.maximum(A.sum(1),1.)
    return torch.tensor(A/np.sqrt(degree[:,None]*degree[None,:]),dtype=torch.float32)

class GLayer(nn.Module):
    def __init__(self,features,out,adj):
        super().__init__()
        self.s=nn.Linear(features,out,bias=False)
        self.n=nn.Linear(features,out,bias=False)
        self.bias=nn.Parameter(torch.zeros(out))
        self.register_buffer("adj",adj)
    def forward(self,x):
        return self.s(x)+self.n(torch.einsum("ij,bjd->bid",self.adj,x))+self.bias

class GNet(nn.Module):
    def __init__(self,adj,n_nodes,learnable=False,rank=None,k=8):
        super().__init__()
        self.learnable=learnable;self.k=k
        self.g1=GLayer(4,24,adj)
        self.g2=GLayer(24,16,adj)
        self.dropout=nn.Dropout(.25)
        self.fc1=nn.Linear(n_nodes*16,48)
        self.fc2=nn.Linear(48,2)
        if learnable:
            self.score=nn.Parameter(torch.zeros(n_nodes))
            if rank is not None:
                with torch.no_grad():self.score[rank[:k]]=.25
            self.tau=.8
    def forward(self,x):
        if self.learnable:
            logits=self.score
            if self.training:
                u=torch.rand_like(logits).clamp(1e-6,1-1e-6)
                logits=logits+.20*(-torch.log(-torch.log(u)))
            hard=torch.zeros_like(logits).scatter_(0,logits.topk(self.k).indices,1.)
            soft=self.k*torch.softmax(self.score/max(.3,self.tau),dim=0)
            x=x*(hard+soft-soft.detach())[None,:,None]
        h=self.dropout(F.relu(self.g1(x)))
        h=F.relu(self.g2(h))
        return self.fc2(self.dropout(F.relu(self.fc1(h.flatten(1)))))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(56,64),nn.ReLU(),
                               nn.Dropout(.25),nn.Linear(64,32),nn.ReLU(),
                               nn.Linear(32,2))
    def forward(self,x):return self.net(x)

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal=nn.Sequential(nn.Conv2d(1,8,(1,63),padding=(0,31),bias=False),
                                    nn.BatchNorm2d(8))
        self.spatial=nn.Sequential(nn.Conv2d(8,16,(14,1),groups=8,bias=False),
                                   nn.BatchNorm2d(16),nn.ELU(),
                                   nn.AvgPool2d((1,4)),nn.Dropout(.35))
        self.separable=nn.Sequential(nn.Conv2d(16,16,(1,15),padding=(0,7),
                                                groups=16,bias=False),
                                     nn.Conv2d(16,16,1,bias=False),
                                     nn.BatchNorm2d(16),nn.ELU(),
                                     nn.AvgPool2d((1,8)),nn.Dropout(.35))
        self.classifier=nn.Linear(16*10,2)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(14,320):
            raise ValueError("EEGNet requires (batch,14,320)")
        x=self.temporal(x[:,None,:,:])
        x=self.spatial(x)
        x=self.separable(x)
        return self.classifier(x.flatten(1))

def predict_net(model,X):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(v).softmax(1)[:,1] for v in
                          torch.from_numpy(X).float().split(128)]).numpy()

def train_net(name,train,labels,valid,val_labels,adj,mi_select,seed,epochs):
    fix_seed(seed)
    if name=="EEGNet14":model=EEGNet()
    elif name=="MLP56":model=MLP()
    elif name=="MI-InputGraph8":
        indices=np.asarray(mi_select)
        model=GNet(norm_adj(adj[np.ix_(indices,indices)]),8)
        train=train[:,indices,:]
        valid=valid[:,indices,:]
    elif name=="FullGraph14":
        model=GNet(norm_adj(adj),14)
    else:
        mirank=list(mi_select)+[x for x in range(14) if x not in mi_select]
        model=GNet(norm_adj(adj),14,learnable=True,rank=mirank)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.001)
    X=torch.from_numpy(train).float()
    y=torch.from_numpy(labels).long()
    best=-np.inf;best_state=None;bad=0;best_epoch=0
    for epoch in range(1,epochs+1):
        model.train()
        if name=="LearnedTopK8":model.tau=max(.3,.8*(.985**(epoch-1)))
        for ids in torch.randperm(len(y)).split(64):
            pred=model(X[ids])
            loss=F.cross_entropy(pred,y[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if epoch%5==0:
            score=balanced_accuracy_score(val_labels,predict_net(model,valid)>=.5)
            if score>best+1e-8:
                best=score;bad=0;best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:bad+=1
            if bad>=10:break
    if best_state is None:raise RuntimeError("Missing neural checkpoint")
    model.load_state_dict(best_state)
    meta=dict(best_val_ba=float(best),best_epoch=best_epoch,
              n_params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if name=="MI-InputGraph8":meta["selected_channels"]=",".join(CHANNELS[i] for i in mi_select)
    if name=="LearnedTopK8":
        ids=model.score.detach().topk(8).indices.cpu().tolist()
        meta["selected_channels"]=",".join(CHANNELS[i] for i in ids)
        meta["different_from_MI"]=int(len(set(ids)-set(mi_select)))
    return model,meta

def csp_lda(train,labels,test):
    from mne.decoding import CSP
    estimator=CSP(n_components=4,reg="ledoit_wolf",log=True,
                  cov_est="concat",transform_into="average_power")
    Ztrain=estimator.fit_transform(train.astype(np.float64),labels)
    model=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
    model.fit(Ztrain,labels)
    ztest=estimator.transform(test.astype(np.float64))
    return model.predict_proba(ztest)[:,1]

def fb_csp_lda(train,labels,test):
    from mne.decoding import CSP
    X_train=[];X_test=[]
    for low,high in CSP_BANDS:
        tr=signal_band(train,low,high).astype(np.float64)
        te=signal_band(test,low,high).astype(np.float64)
        estimator=CSP(n_components=2,reg="ledoit_wolf",log=True,
                      cov_est="concat",transform_into="average_power")
        X_train.append(estimator.fit_transform(tr,labels))
        X_test.append(estimator.transform(te))
    model=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
    model.fit(np.concatenate(X_train,axis=1),labels)
    return model.predict_proba(np.concatenate(X_test,axis=1))[:,1]

def riemann_mdm(train,labels,test):
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    cov=Covariances(estimator="oas")
    tr=cov.fit_transform(signal_band(train,8.,30.).astype(np.float64))
    te=cov.transform(signal_band(test,8.,30.).astype(np.float64))
    clf=MDM(metric="riemann")
    clf.fit(tr,labels)
    return clf.predict_proba(te)[:,1]

def measure(labels,prob):
    pred=(prob>=.5).astype(np.int64)
    return dict(ba=float(balanced_accuracy_score(labels,pred)),
                f1=float(f1_score(labels,pred,zero_division=0)),
                auc=float(roc_auc_score(labels,prob)),
                accuracy=float(accuracy_score(labels,pred))),pred

def run(wave,y,rows,out,epochs,folds=4,smoke=False,seed=SEED):
    torch.set_num_threads(1)
    subject_ids=np.sort(rows.subject.unique())
    split=KFold(n_splits=folds,shuffle=True,random_state=seed)
    adj=make_adjacency()
    feat=feature_bands(wave)
    performance=[];predictions=[];partitions=[]
    for fold,(trpool,testidx) in enumerate(split.split(subject_ids),1):
        source=subject_ids[trpool].copy()
        test_subjects=subject_ids[testidx]
        rng=np.random.default_rng(seed+fold)
        rng.shuffle(source)
        nval=max(2,round(.2*len(source)))
        val_subjects=source[:nval];train_subjects=source[nval:]
        if set(train_subjects)&set(val_subjects) or set(train_subjects)&set(test_subjects) or set(val_subjects)&set(test_subjects):
            raise RuntimeError("Subject leakage")
        tr=rows.subject.isin(train_subjects).to_numpy()
        va=rows.subject.isin(val_subjects).to_numpy()
        te=rows.subject.isin(test_subjects).to_numpy()
        for phase,mask in (("train",tr),("val",va),("test",te)):
            if min(np.bincount(y[mask],minlength=2))<3:
                raise RuntimeError(f"Fold {fold} missing {phase} class")
        partitions.append(dict(fold=fold,n_train=int(tr.sum()),n_val=int(va.sum()),
                               n_test=int(te.sum()),n_training_subjects=len(train_subjects),
                               n_val_subjects=len(val_subjects),
                               n_test_subjects=len(test_subjects),
                               train_subjects=",".join(map(str,sorted(train_subjects))),
                               val_subjects=",".join(map(str,sorted(val_subjects))),
                               test_subjects=",".join(map(str,sorted(test_subjects))),
                               test_class1_rate=float(y[te].mean())))
        if smoke:continue
        scaler=StandardScaler().fit(feat[tr].reshape(int(tr.sum()),-1))
        def normalized(mask):
            return scaler.transform(feat[mask].reshape(int(mask.sum()),-1)).astype(np.float32).reshape(-1,14,4)
        ft_tr,ft_val,ft_test=(normalized(mask) for mask in (tr,va,te))
        MI=mutual_info_classif(ft_tr.reshape(len(ft_tr),-1),y[tr],random_state=seed+fold)
        MI=MI.reshape(14,4).mean(axis=1)
        selected=np.argsort(-MI,kind="stable")[:8].tolist()
        print("FOLD-BEGIN",fold,"train/val/test",tr.sum(),va.sum(),te.sum(),
              "MI_channels",",".join(CHANNELS[i] for i in selected),flush=True)
        for name in MODELS:
            start=time.monotonic()
            meta=dict(best_val_ba=None,best_epoch=None,n_params=None,
                      selected_channels="",different_from_MI=None)
            if name in ("CSP-LDA14","FBCSP-LDA14","RiemannMDM14"):
                if name=="CSP-LDA14":
                    prob=csp_lda(signal_band(wave[tr],8.,30.),y[tr],
                                 signal_band(wave[te],8.,30.))
                elif name=="FBCSP-LDA14":
                    prob=fb_csp_lda(wave[tr],y[tr],wave[te])
                else:prob=riemann_mdm(wave[tr],y[tr],wave[te])
            else:
                if name=="EEGNet14":
                    # Fixed physical scaling: ~20 uV unit, no cross-subject fitted statistics.
                    # Keep waveform unsaturated; raw data filtered 4-40Hz inside source run.
                    xt=(wave[tr]/20.).astype(np.float32)
                    xv=(wave[va]/20.).astype(np.float32)
                    xte=(wave[te]/20.).astype(np.float32)
                else:xt,xv,xte=ft_tr,ft_val,ft_test
                model,other=train_net(name,xt,y[tr],xv,y[va],adj,selected,
                                      seed+100*fold,epochs)
                meta.update(other)
                if name=="MI-InputGraph8":xte=xte[:,selected,:]
                prob=predict_net(model,xte)
            metric,labels=measure(y[te],prob)
            performance.append(dict(fold=fold,model=name,subject_count=int(len(test_subjects)),
                                    n_test=int(te.sum()),fit_seconds=time.monotonic()-start,
                                    **metric,**meta))
            for j,idx in enumerate(np.flatnonzero(te)):
                predictions.append(dict(fold=fold,model=name,subject=int(rows.subject.iloc[idx]),
                                        run=int(rows.run.iloc[idx]),
                                        event_index=int(rows.event_index.iloc[idx]),
                                        label=int(y[idx]),pred=int(labels[j]),p_right=float(prob[j])))
            pd.DataFrame(performance).to_csv(out/"PARTIAL_metrics.csv",index=False)
            print("RESULT",name,"fold",fold,"BA",round(metric["ba"],4),
                  "F1",round(metric["f1"],4),"AUC",round(metric["auc"],4),
                  "time",round(performance[-1]["fit_seconds"],1),flush=True)
    pd.DataFrame(partitions).to_csv(out/"splits.csv",index=False)
    if smoke:
        print("SMOKE-REAL-EDF-OK",len(subject_ids),"subjects",
              len(wave),"events",len(partitions),"folds",flush=True)
        return
    perf=pd.DataFrame(performance)
    pred=pd.DataFrame(predictions)
    if len(pred)!=len(rows)*len(MODELS):
        raise RuntimeError(f"Incomplete prediction rows {len(pred)}; expected {len(rows)*len(MODELS)}")
    if pred.duplicated(["model","subject","run","event_index"]).any():
        raise RuntimeError("Duplicate prediction keys")
    perf.to_csv(out/"fold_results.csv",index=False)
    pred.to_csv(out/"oof_predictions.csv",index=False)
    perf.groupby("model")[["ba","f1","auc","accuracy"]].agg(["mean","std"]).to_csv(out/"fold_summary.csv")
    subj=[]
    for (name,subject),gr in pred.groupby(["model","subject"]):
        if set(gr.label.unique())!={0,1}:
            raise ValueError("Subject missing class")
        m,_=measure(gr.label.to_numpy(),gr.p_right.to_numpy())
        subj.append(dict(model=name,subject=int(subject),n_events=len(gr),**m))
    sub=pd.DataFrame(subj)
    sub.to_csv(out/"subject_results.csv",index=False)
    pivot=sub.pivot(index="subject",columns="model",values="ba")
    contrasts=[]
    rng=np.random.default_rng(20261018)
    for other in MODELS:
        if other=="LearnedTopK8":continue
        d=(pivot["LearnedTopK8"]-pivot[other]).to_numpy()
        if not np.isfinite(d).all():raise RuntimeError("Nonfinite paired differences")
        draws=rng.integers(0,len(d),size=(5000,len(d)))
        interval=np.quantile(d[draws].mean(1)*100,[.025,.975])
        p=float(wilcoxon(d).pvalue) if not np.allclose(d,0) else 1.
        contrasts.append(dict(reference="LearnedTopK8",against=other,
                              n_subjects=len(d),delta_pp=float(100*d.mean()),
                              ci95_low_pp=float(interval[0]),
                              ci95_high_pp=float(interval[1]),unadjusted_p=p))
    stats=pd.DataFrame(contrasts)
    order=np.argsort(stats.unadjusted_p.to_numpy())
    corrected=np.empty(len(stats));max_val=0
    for i,ind in enumerate(order):
        max_val=max(max_val,min(1,(len(stats)-i)*float(stats.unadjusted_p.iloc[ind])))
        corrected[ind]=max_val
    stats["holm_p"]=corrected
    stats.to_csv(out/"paired_subject_exploratory.csv",index=False)
    (out/"PARTIAL_metrics.csv").unlink(missing_ok=True)
    (out/"RUN_COMPLETE.txt").write_text("TRUE EDF THREE-RUN LEFT RIGHT IMAGERY, SUBJECT-HELD-OUT\n")
    print("THREE-RUN-COMPLETE",len(wave),"epochs",len(subject_ids),"subjects",flush=True)
    print(perf.groupby("model")[["ba","f1","auc"]].mean().to_string(),flush=True)
    print(stats.to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--subject-end",type=int,default=24)
    p.add_argument("--folds",type=int,default=4)
    p.add_argument("--epochs",type=int,default=65)
    p.add_argument("--cache-dir",default="cache/three_run")
    p.add_argument("--out",default="outputs/mi_3run")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if not 8<=args.subject_end<=109:raise ValueError("subject-end must be 8..109")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    subjects=range(1,args.subject_end+1)
    wave,y,table,audit=acquire_dataset(subjects,Path(args.cache_dir),workers=args.workers)
    table.to_csv(out/"events.csv",index=False)
    audit.to_csv(out/"edf_audit.csv",index=False)
    # Save authentic features/signals for exact reproducibility.
    np.savez_compressed(out/"real_epochs_float32.npz",wave=wave,y=y)
    metadata=dict(dataset="PhysioNet EEGMMIDB v1.0.0",
                  task="Imagined left T1 versus right T2 in R04/R08/R12",
                  runs=RUNS,subjects=int(table.subject.nunique()),
                  n_real_epochs=int(len(table)),sampling_rate=SFREQ,
                  data_quality_policy="Include every genuine T1/T2 cue from every verified original EDF; allow unequal genuine event counts and annotate short EDF runs; never pad or synthesise tasks",
                  n_short_runs=int(audit.is_short_run.sum()),
                  short_runs=audit.loc[audit.is_short_run,["subject","run","n_events","n_left","n_right"]].to_dict("records"),
                  channel_names=CHANNELS,n_samples_per_epoch=SAMPLES,
                  time_window_after_cue_seconds=[.5,2.5],
                  frequency_bands=FEATURE_BANDS,models=MODELS,folds=args.folds,
                  selector_k=8,random_seed=SEED,training_epochs=args.epochs,
                  data_leakage_controls="subject disjoint; full train-only scaler, MI/CSP/covariance and fitting; validation-only checkpoint",
                  classifier_evaluation="original 0.5, no test tuning",
                  caveats=["Exploratory 24-person pilot, not confirmatory full population.",
                           "TopK mask processes 14-node GNN; only MI input subgraph has 8 convolution nodes.",
                           "Within-subject motor imagery can outperform cross-subject; do not compare their absolute BAs.",
                           "Training/test subjects differ, but shared EDF acquisition device remains."])
    (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    run(wave,y,table,out,args.epochs,folds=args.folds,smoke=args.smoke)

if __name__=="__main__":
    main()

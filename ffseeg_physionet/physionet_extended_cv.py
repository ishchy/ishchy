#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subject-held-out PhysioNet eyes-open/closed baseline exploratory validation.

WARNING: label is confounded with R01/R02 run; NOT zero-shot transfer from UCI.
No outputs are reported as empirical unless --smoke is false and real EDF loaded.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import ffseeg_repro_experiments as base
import ffseeg_physionet_external as old

MODELS = ['FullGraphConv', 'MI-PostGCN', 'MI-InputGCN', 'FFSEEG-GNN', 'Random-PostGCN']

def induced_adj(A, ids):
    B=A[ids,:][:,ids].clone()
    B=B+torch.eye(len(ids),device=B.device)
    D=torch.diag((B.sum(1).clamp_min(1)).pow(-.5))
    return D @ B @ D

def train_predict(name, split, A, A_norm, cfg, seed, device):
    Xtr,ytr,Xva,yva,Xte,yte=split
    base.seed_everything(seed)
    ids=None
    if name in ['MI-PostGCN','MI-InputGCN']:
        scores=mutual_info_classif(Xtr,ytr, random_state=seed)
        ids=np.argsort(-scores,kind='stable')[:cfg.k].tolist()
    elif name=='Random-PostGCN':
        ids=np.random.default_rng(seed+52473).choice(Xtr.shape[1],size=cfg.k,replace=False).tolist()

    if name=='FullGraphConv':
        model=base.FullGraphConv(A_norm,dropout=.10).to(device)
        model=base._train_generic(model,Xtr,ytr,Xva,yva,cfg,device,lr=cfg.lr_fullgcn)
        logits=base.infer_logits(model,Xte,cfg.batch_size,device)
    elif name=='FFSEEG-GNN':
        model,ids,aux=base.train_ffseeg(A_norm,Xtr,ytr,Xva,yva,cfg,device)
        logits=base.infer_logits(model,Xte,cfg.batch_size,device)
    elif name=='MI-InputGCN':
        sA=induced_adj(A,ids)
        encoder=base.GCNEncoder(sA,hidden=16,dropout=.10)
        model=base.FFSEEGHard(encoder, list(range(cfg.k)),hidden=16,dropout=.10).to(device)
        model=base._train_generic(model,Xtr[:,ids],ytr,Xva[:,ids],yva,cfg,device,lr=cfg.lr_hard)
        logits=base.infer_logits(model,Xte[:,ids],cfg.batch_size,device)
    elif name in ('MI-PostGCN','Random-PostGCN'):
        encoder=base.GCNEncoder(A_norm,hidden=16,dropout=.10)
        model=base.FFSEEGHard(encoder,ids,hidden=16,dropout=.10).to(device)
        model=base._train_generic(model,Xtr,ytr,Xva,yva,cfg,device,lr=cfg.lr_hard)
        logits=base.infer_logits(model,Xte,cfg.batch_size,device)
    else: raise KeyError(name)
    return logits,ids, sum(p.numel() for p in model.parameters())

def validate_features(df, *, smoke):
    required=set(base.CHANNELS) | {'subject','run','window','label'}
    if not required.issubset(df.columns): raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    if df[list(base.CHANNELS)].isnull().any().any() or not np.isfinite(df[list(base.CHANNELS)].to_numpy(dtype=float)).all():
        raise ValueError('Features contain NaN or nonfinite values')
    if set(df['label'].astype(int).unique()) != {0,1}: raise ValueError('Expected labels 0 and 1')
    if not ((df.run.astype(int)==df.label.astype(int)+1)).all():
        raise ValueError('Run/label mismatch: PhysioNet R01=0, R02=1')
    if df.duplicated(['subject','run','window']).any():raise ValueError('Duplicate subject/run/window')
    n_subjects=df.subject.nunique()
    if not smoke and n_subjects<30:
        raise ValueError('Too few subjects for an external result: supply >=30 or use --smoke')

def subject_metrics(df):
    rows=[]
    for (model,subj),g in df.groupby(['model','subject']):
        x=g.y_true.to_numpy(int); y=g.y_pred.to_numpy(int); p=g.prob_closed.to_numpy(float)
        if len(np.unique(x))!=2: raise ValueError(f'Subject {subj} is missing a class')
        from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
        rows.append({'model':model,'subject':int(subj),'n':len(g),'accuracy':accuracy_score(x,y),
                     'f1':f1_score(x,y,zero_division=0),'balanced_accuracy':balanced_accuracy_score(x,y),
                     'auc':roc_auc_score(x,p)})
    return pd.DataFrame(rows)

def paired_bootstrap(df, *, seed=20261014, replicates=5000):
    rng=np.random.default_rng(seed)
    pivot=df.pivot(index='subject',columns='model',values='balanced_accuracy')
    reference='FFSEEG-GNN'
    rows=[]
    for other in MODELS:
        if other==reference: continue
        d=(pivot[reference]-pivot[other]).dropna().to_numpy(float)
        sample=rng.integers(0,len(d),size=(replicates,len(d)))
        boot=d[sample].mean(axis=1)*100
        rows.append({'reference':reference,'comparison':other,'n_subjects':len(d),
                     'mean_diff_pp':100*d.mean(),'ci95_low_pp':np.quantile(boot,.025),
                     'ci95_high_pp':np.quantile(boot,.975),'bootstrap_replicates':replicates})
    return pd.DataFrame(rows)

def run(df, out, n_splits, split_seed, max_epochs, batch_size, smoke):
    validate_features(df,smoke=smoke)
    torch.set_num_threads(1)
    device=torch.device('cpu')
    cfg=base.TrainConfig(batch_size=batch_size,max_epochs=max_epochs,k=8)
    A,A_norm,_=base.build_adjacency(device)
    X=df[base.CHANNELS].to_numpy(np.float32)
    y=df.label.to_numpy(np.int64)
    subjects=df.subject.to_numpy(np.int64)
    pred_rows=[]; fold_rows=[]; split_rows=[]; top_rows=[]
    for fold, trsubs, vasubs, tesubs in old.make_subject_folds(df,n_splits,split_seed):
        assert not (set(trsubs)&set(vasubs) or set(trsubs)&set(tesubs) or set(vasubs)&set(tesubs))
        tr=np.isin(subjects,trsubs);va=np.isin(subjects,vasubs);te=np.isin(subjects,tesubs)
        scl=StandardScaler().fit(X[tr])
        split=(scl.transform(X[tr]).astype(np.float32),y[tr],scl.transform(X[va]).astype(np.float32),y[va],
               scl.transform(X[te]).astype(np.float32),y[te])
        split_rows.append({'fold':fold,'train_subjects':','.join(map(str,trsubs)),
                           'validation_subjects':','.join(map(str,vasubs)),
                           'test_subjects':','.join(map(str,tesubs)),
                           'n_train':int(tr.sum()),'n_val':int(va.sum()),'n_test':int(te.sum())})
        print(f'Fold {fold}/{n_splits} tr={tr.sum()} val={va.sum()} test={te.sum()}',flush=True)
        for mi,name in enumerate(MODELS):
            t0=time.monotonic();seed=12000+fold*100
            logits, ids, nparams=train_predict(name,split,A,A_norm,cfg,seed,device)
            m=old.metrics(y[te],logits)
            fold_rows.append({'fold':fold,'model':name,'seed':seed,'n_params':nparams,**m})
            probs=torch.softmax(torch.from_numpy(logits),dim=1).numpy()[:,1]
            ypred=logits.argmax(1)
            for q,orig_id in enumerate(np.flatnonzero(te)):
                pred_rows.append({'fold':fold,'model':name,'subject':int(subjects[orig_id]),
                                  'run':int(df.iloc[orig_id].run),'window':int(df.iloc[orig_id].window),
                                  'y_true':int(y[orig_id]),'y_pred':int(ypred[q]),'prob_closed':float(probs[q])})
            if ids is not None:
                ids=[int(x) for x in ids]
                top_rows.append({'fold':fold,'model':name,'indices':','.join(map(str,ids)),
                                 'channels':','.join(base.CHANNELS[i] for i in ids)})
            print(f'  {name} bacc={m["balanced_accuracy"]:.4f} f1={m["f1"]:.4f} sec={time.monotonic()-t0:.1f}',flush=True)
        # Checkpoint every fold, an interrupted run should never masquerade as complete.
        pd.DataFrame(fold_rows).to_csv(out/'INCOMPLETE_fold_metrics.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'external_fold_metrics.csv',index=False)
    pd.DataFrame(split_rows).to_csv(out/'external_subject_splits.csv',index=False)
    pd.DataFrame(top_rows).to_csv(out/'external_selected_nodes.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(out/'external_oof_predictions.csv',index=False)
    sj=subject_metrics(pd.DataFrame(pred_rows))
    sj.to_csv(out/'external_subject_metrics.csv',index=False)
    sj.groupby('model')[['accuracy','f1','balanced_accuracy','auc']].agg(['mean','std','median']).to_csv(out/'external_subject_summary.csv')
    paired_bootstrap(sj).to_csv(out/'subject_bootstrap_95ci.csv',index=False)
    pd.DataFrame(fold_rows).groupby('model')[['accuracy','f1','balanced_accuracy','auc']].agg(['mean','std']).to_csv(out/'external_fold_summary.csv')
    # Check exact 1OOF prediction per model / epoch: strict validation.
    for name in MODELS:
        p=pd.DataFrame(pred_rows)
        n=p[p.model==name].shape[0]
        if n!=len(df): raise RuntimeError(f'Missing prediction rows for {name}: {n} / {len(df)}')
    meta={'dataset':'PhysioNet EEGMMIDB baseline R01 vs R02','data_type':'SYNTHETIC - NOT EEG' if smoke else 'REAL EDF',
          'task_caveat':'eye-state perfectly confounded with baseline run number; task is between-run state discrimination',
          'analysis_note':'Fresh models trained on PhysioNet, not zero-shot UCI-to-PhysioNet transfer',
          'subjects':int(df.subject.nunique()),'windows':len(df),'n_splits':n_splits,'split_seed':split_seed,
          'models':MODELS,'training':vars(cfg),'channel_order':base.CHANNELS,'graph_edges':base.EDGES,
          'pre_input_variant':'selected induced subgraph; no unselected electrode reaches graph encoder',
          'post_variant':'all 14 original electrodes propagate in GCN; last layer only keeps selected K representations',
          'feature':'2sec Hann log relative alpha power 8-13 Hz / 4-30 Hz',
          'dataset_url':'https://physionet.org/content/eegmmidb/1.0.0/'}
    (out/'external_run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'INCOMPLETE_fold_metrics.csv').unlink(missing_ok=True)
    (out/'RUN_COMPLETE.txt').write_text('Real EEG result\n' if not smoke else 'SYNTHETIC SMOKE TEST - NOT REAL EEG\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='physionet_cross_subject_results')
    p.add_argument('--cache-dir',default='eegmmidb_edf_cache')
    p.add_argument('--features-csv',help='Previously extracted subject/run/window/label and EEG channel features')
    p.add_argument('--n-splits',type=int,default=10)
    p.add_argument('--subject-start',type=int,default=1)
    p.add_argument('--subject-end',type=int,default=109)
    p.add_argument('--max-epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--split-seed',type=int,default=20260918)
    p.add_argument('--smoke',action='store_true',help='Synthetic execution test only, results NOT publishable')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.smoke:
        df, dataset_meta=old.synthetic_dataset(n_subjects=20,epochs_per_class=6)
    elif a.features_csv:
        df=pd.read_csv(a.features_csv)
        dataset_meta={'source':a.features_csv,'sha256':hashlib.sha256(Path(a.features_csv).read_bytes()).hexdigest()}
    else:
        df,dataset_meta=old.extract_real_dataset(Path(a.cache_dir),range(a.subject_start,a.subject_end+1))
    df.to_csv(out/'external_features.csv',index=False)
    (out/'dataset_extraction_audit.json').write_text(json.dumps(dataset_meta,ensure_ascii=False,indent=2),encoding='utf-8')
    run(df,out,a.n_splits,a.split_seed,a.max_epochs,a.batch_size,a.smoke)
    print('DONE',out,flush=True)
if __name__=='__main__':main()
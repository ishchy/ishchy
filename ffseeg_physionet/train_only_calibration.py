#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-audited EEG preprocessing/loss/threshold optimization.

Scientific restrictions
-----------------------
- Never alter target labels or outer test partitions.
- Scaling and 1%-99% winsor quantiles fit on TRAIN rows only.
- Class weights are computed from TRAIN only.
- MI and SVM feature selection fit on TRAIN only.
- Checkpoints, model hyperparameter ranking and decision thresholds use
  VALIDATION rows only; test outcomes never select a candidate.
- One trained model's test probabilities are scored both at fixed 0.5 and at
  a conservatively shrunk validation-selected threshold, so threshold impact
  is measured on IDENTICAL predictions.
- This is exploratory because earlier evaluation results shaped study design.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import physionet_sparse_residual as sparse
import ffseeg_repro_experiments as graph
import uci_forward_time_validation as uci

MODELS=["FFSEEG-Residual","FullGraphConv","MLP","MI-SVM-8","RBF-SVM-14"]
PREPROCESSES=["standard","train_winsor01"]
CLASS_WEIGHTS=["none","balanced"]
ALPHA=.5  # validation threshold adjustment shrunk half-way back to 0.5
SEED=24701

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

class FoldPreprocessor:
    def __init__(self, mode):
        self.mode=mode
    def fit(self,X):
        self.low=self.high=None
        XX=X
        if self.mode=="train_winsor01":
            self.low=np.quantile(X,.01,axis=0)
            self.high=np.quantile(X,.99,axis=0)
            XX=np.clip(X,self.low,self.high)
        self.scaler=StandardScaler().fit(XX)
        return self
    def transform(self,X):
        if self.low is not None:X=np.clip(X,self.low,self.high)
        return self.scaler.transform(X).astype(np.float32)

def proba_network(model,X):
    model.eval()
    chunks=[]
    with torch.inference_mode():
        for ids in np.array_split(X,max(1,(len(X)+511)//512)):
            inp=torch.from_numpy(ids)
            logits=model(inp if isinstance(model,sparse.FFSEEGResidual) else inp.unsqueeze(-1))
            chunks.append(logits.softmax(1)[:,1].detach().cpu().numpy())
    return np.concatenate(chunks)

def balanced_weights(y):
    count=np.bincount(y,minlength=2).astype(float)
    return torch.tensor(np.clip(len(y)/(2*np.maximum(count,1)),.2,5.),dtype=torch.float32)

def fit_neural(name,Xtr,ytr,Xva,yva,seed,class_weight,epochs):
    seed_all(seed)
    if name=="FFSEEG-Residual":
        rank=np.argsort(-mutual_info_classif(Xtr,ytr,random_state=seed),kind="stable")
        model=sparse.FFSEEGResidual(k=8,residual=True,mi_rank=rank,dropout=.2)
    elif name=="FullGraphConv":
        model=graph.FullGraphConv(sparse.adjacency(),hidden=16,dropout=.1)
    elif name=="MLP":
        model=graph.MLP(dropout=.1)
    else:raise ValueError(name)
    opt=torch.optim.AdamW(model.parameters(),lr=.002 if name=="FFSEEG-Residual" else .006,weight_decay=.002)
    Xt=torch.from_numpy(Xtr)
    yt=torch.from_numpy(ytr)
    w=balanced_weights(ytr) if class_weight=="balanced" else None
    best=-np.inf;best_state=None;best_ep=0;bad=0
    for ep in range(1,epochs+1):
        model.train()
        for idx in torch.randperm(len(ytr)).split(256):
            inp=Xt[idx] if name=="FFSEEG-Residual" else Xt[idx].unsqueeze(-1)
            out=model(inp)
            loss=F.cross_entropy(out,yt[idx],weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
        if ep%5==0:
            prob=proba_network(model,Xva)
            ba=float(balanced_accuracy_score(yva,prob>=.5))
            if ba>best+1e-7:
                best=ba;bad=0;best_ep=ep
                best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
            else:bad+=1
            if bad>=8:break
    if best_state is None:raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best_state)
    selected=""
    if name=="FFSEEG-Residual":
        order=model.gate.detach().topk(8).indices.cpu().tolist()
        selected=",".join(sparse.CHANNELS[i] for i in order)
    return model,dict(n_params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                      best_epoch=best_ep,selected=selected,svm_c=None)

def fit_svm(name,Xtr,ytr,Xva,yva,seed,class_weight):
    if name=="MI-SVM-8":
        ranking=np.argsort(-mutual_info_classif(Xtr,ytr,random_state=seed),kind="stable")
        selected=ranking[:8].tolist()
    else:selected=list(range(Xtr.shape[1]))
    w="balanced" if class_weight=="balanced" else None
    best=None
    for c in (.25,1.,4.):
        model=SVC(C=c,kernel="rbf",gamma="scale",class_weight=w,cache_size=300)
        model.fit(Xtr[:,selected],ytr)
        val=expit(np.clip(model.decision_function(Xva[:,selected]),-30,30))
        metric=float(balanced_accuracy_score(yva,val>=.5))
        if best is None or metric>best[0]+1e-10:
            best=(metric,c,model)
    return best[2],dict(n_params=None,best_epoch=None,
                        selected=",".join(sparse.CHANNELS[i] for i in selected),
                        svm_c=best[1],feature_indices=selected)

def score(y,p,threshold):
    pred=(np.asarray(p)>=threshold).astype(int)
    return dict(ba=float(balanced_accuracy_score(y,pred)),
                f1=float(f1_score(y,pred,zero_division=0)),
                auc=float(roc_auc_score(y,p)),
                accuracy=float(accuracy_score(y,pred))),pred

def validation_threshold(y,p):
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    j=tpr-fpr
    finite=np.isfinite(thresholds) & (thresholds>=.05) & (thresholds<=.95)
    if not finite.any():return .5
    best_j=np.max(j[finite])
    ties=np.flatnonzero(finite & (j>=best_j-1e-10))
    ix=ties[np.argmin(np.abs(thresholds[ties]-.5))]
    return float(thresholds[ix])

def fold_data(data,features,splits,uci_zip):
    if data=="uci":
        frame,_=uci.load_data(Path(uci_zip))
        X=frame[sparse.CHANNELS].to_numpy(np.float32)
        y=frame.eyeDetection.to_numpy(np.int64)
        folds=uci.make_forward_folds(len(frame),256,1250)
        return frame,X,y,folds
    frame=pd.read_csv(features)
    sp=pd.read_csv(splits)
    if len(frame)!=6538 or frame.subject.nunique()!=109 or len(sp)!=10:
        raise ValueError("PhysioNet dataset or folds differ from fixed benchmark")
    folds=[]
    for _,row in sp.iterrows():
        masks=[]
        for kind in ("train_subjects","validation_subjects","test_subjects"):
            subjects=[int(v) for v in str(row[kind]).split(",")]
            masks.append(np.flatnonzero(frame.subject.isin(subjects).to_numpy()))
        folds.append(tuple(masks))
    return frame,frame[sparse.CHANNELS].to_numpy(np.float32),frame.label.to_numpy(np.int64),folds

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True,choices=["uci","physionet"])
    parser.add_argument("--features")
    parser.add_argument("--splits")
    parser.add_argument("--uci-zip")
    parser.add_argument("--out",required=True)
    parser.add_argument("--epochs",type=int,default=65)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df,X,y,folds=fold_data(args.dataset,args.features,args.splits,args.uci_zip)
    audit=[]
    used_tests=[]
    for fold,(tr,va,te) in enumerate(folds,1):
        assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
        if args.dataset=="uci":
            assert tr.max()<va.min() and va.max()<te.min()
            assert va.min()-tr.max()>=256 and te.min()-va.max()>=256
        else:
            assert not(set(df.subject.iloc[tr])&set(df.subject.iloc[va]) or
                       set(df.subject.iloc[tr])&set(df.subject.iloc[te]) or
                       set(df.subject.iloc[va])&set(df.subject.iloc[te]))
        for inds,part in ((tr,"train"),(va,"validation"),(te,"test")):
            counts=np.bincount(y[inds],minlength=2)
            if min(counts)==0:raise ValueError(f"Fold {fold} {part} missing a class")
        audit.append(dict(fold=fold,n_train=len(tr),n_val=len(va),n_test=len(te),
                          test_class1_fraction=float(y[te].mean())))
        used_tests.extend(te)
    if len(set(used_tests))!=len(used_tests):
        raise RuntimeError("Outer test row overlap")
    pd.DataFrame(audit).to_csv(out/"partition_audit.csv",index=False)
    if args.smoke:
        print("PRECHECK-OK",args.dataset,"folds",len(folds),"samples",len(y),flush=True)
        return
    all_candidates=[];chosen_fold=[];preds=[]
    for fold,(tr,va,te) in enumerate(folds,1):
        per_method={}
        for pp in PREPROCESSES:
            processor=FoldPreprocessor(pp).fit(X[tr])
            Xt=processor.transform(X[tr])
            Xv=processor.transform(X[va])
            Xs=processor.transform(X[te])
            for model_name in MODELS:
                for weight in CLASS_WEIGHTS:
                    seed=SEED+100*fold
                    start=time.monotonic()
                    if model_name.endswith("SVM-8") or model_name=="RBF-SVM-14":
                        model,detail=fit_svm(model_name,Xt,y[tr],Xv,y[va],seed,weight)
                        indices=detail.pop("feature_indices")
                        val=expit(np.clip(model.decision_function(Xv[:,indices]),-30,30))
                        test=expit(np.clip(model.decision_function(Xs[:,indices]),-30,30))
                    else:
                        model,detail=fit_neural(model_name,Xt,y[tr],Xv,y[va],seed,weight,args.epochs)
                        val=proba_network(model,Xv)
                        test=proba_network(model,Xs)
                    th_opt=validation_threshold(y[va],val)
                    th_shrink=float(.5+ALPHA*(th_opt-.5))
                    m_val_fixed,_=score(y[va],val,.5)
                    m_val_shrink,_=score(y[va],val,th_shrink)
                    m_test_fixed,_=score(y[te],test,.5)
                    m_test_shrink,_=score(y[te],test,th_shrink)
                    rec=dict(fold=fold,model=model_name,preprocess=pp,class_weight=weight,
                             train_count=len(tr),validation_count=len(va),test_count=len(te),
                             selected_threshold_opt=th_opt,threshold_shrunk=th_shrink,
                             val_ba_fixed=m_val_fixed["ba"],val_ba_shrunk=m_val_shrink["ba"],
                             test_ba_fixed=m_test_fixed["ba"],test_ba_shrunk=m_test_shrink["ba"],
                             test_f1_fixed=m_test_fixed["f1"],test_f1_shrunk=m_test_shrink["f1"],
                             test_auc=m_test_fixed["auc"],selected=detail["selected"],
                             n_params=detail["n_params"],best_epoch=detail["best_epoch"],
                             svm_c=detail["svm_c"],duration_s=time.monotonic()-start)
                    all_candidates.append(rec)
                    per_method.setdefault(model_name,[]).append((rec,test))
                    print("CANDIDATE",args.dataset,"fold",fold,model_name,pp,weight,
                          "val_BA",round(m_val_shrink["ba"],4),"test_BA",round(m_test_shrink["ba"],4),
                          "thr",round(th_shrink,4),flush=True)
        for model_name,options in per_method.items():
            # Pick candidate EXCLUSIVELY from validation performance. Preference on ties:
            # standard/no-class-weight is first in search order and therefore wins.
            selected,prob=max(options,key=lambda pair:(pair[0]["val_ba_shrunk"],
                                                        -PREPROCESSES.index(pair[0]["preprocess"]),
                                                        -CLASS_WEIGHTS.index(pair[0]["class_weight"])))
            chosen_fold.append(selected)
            for j,ix in enumerate(te):
                datum=dict(fold=fold,model=model_name,original_row_index=int(ix),
                           y_true=int(y[ix]),prob_class1=float(prob[j]),
                           y_pred_05=int(prob[j]>=.5),
                           y_pred_shrunk=int(prob[j]>=selected["threshold_shrunk"]))
                if args.dataset=="physionet":datum["subject"]=int(df.subject.iloc[ix])
                else:datum["original_time_index"]=int(df.original_index.iloc[ix])
                preds.append(datum)
        pd.DataFrame(all_candidates).to_csv(out/"all_candidates_partial.csv",index=False)
        pd.DataFrame(chosen_fold).to_csv(out/"chosen_fold_partial.csv",index=False)
        print("FOLD-COMPLETE",args.dataset,fold,flush=True)
    candidates=pd.DataFrame(all_candidates)
    chosen=pd.DataFrame(chosen_fold)
    predictions=pd.DataFrame(preds)
    assert len(chosen)==len(folds)*len(MODELS)
    assert len(predictions)==len(used_tests)*len(MODELS)
    assert not predictions.duplicated(["model","original_row_index"]).any()
    candidates.to_csv(out/"all_candidate_results_exploratory.csv",index=False)
    chosen.to_csv(out/"validation_chosen_fold_results.csv",index=False)
    predictions.to_csv(out/"validation_chosen_oof_predictions.csv",index=False)
    summaries=[]
    for model_name in MODELS:
        g=chosen[chosen.model==model_name]
        base=candidates[(candidates.model==model_name)&
                        (candidates.preprocess=="standard")&
                        (candidates.class_weight=="none")]
        for mode,frame,col in (("fixed_baseline",base,"test_ba_fixed"),
                               ("baseline_threshold_only",base,"test_ba_shrunk"),
                               ("validation_selected_features_and_weights",g,"test_ba_fixed"),
                               ("validation_selected_plus_threshold",g,"test_ba_shrunk")):
            summaries.append(dict(model=model_name,strategy=mode,
                                  mean_ba=float(frame[col].mean()),
                                  mean_f1=float(frame["test_f1_shrunk" if "threshold" in mode else "test_f1_fixed"].mean()),
                                  mean_auc=float(frame.test_auc.mean()),
                                  n_folds=len(frame),
                                  mean_threshold=float(frame.threshold_shrunk.mean())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/"comparison_summary.csv",index=False)
    raw_selected=chosen[chosen.model=="FFSEEG-Residual"].sort_values("fold")
    selected_delta=raw_selected.test_ba_shrunk.to_numpy() - raw_selected.test_ba_fixed.to_numpy()
    selected_gt=raw_selected.test_ba_shrunk.to_numpy() - chosen[chosen.model=="FullGraphConv"].sort_values("fold").test_ba_shrunk.to_numpy()
    metadata=dict(dataset=args.dataset,
                  n_samples=len(df),n_folds=len(folds),n_models=len(MODELS),
                  n_candidate_fit=len(candidates),n_oof_selected_predictions=len(predictions),
                  preprocessing="standard or train-only 1%-99% winsorization plus train-only StandardScaler",
                  weights="none or train-only inverse-frequency cross-entropy; class-weighted SVC",
                  validation_threshold="validation ROC maximal balanced accuracy; clipped [0.05,0.95] then shrunk 50% toward 0.5",
                  selection="per model/fold choose preprocessing and class weights by validation BA with shrunk threshold only",
                  no_test_selection=True,no_test_preprocessing=True,seed=SEED,max_epochs=args.epochs,
                  observed_delta_ffseeg_threshold_pp=selected_delta.mean()*100,
                  observed_delta_ffseeg_vs_fullgcn_pp=selected_gt.mean()*100,
                  caveats=["Earlier outer-test results influenced model development; these are exploratory comparisons.",
                           "Validation reuse for checkpoint + threshold + model selection can overfit validation.",
                           "PhysioNet eye-state labels are confounded with acquisition run R01/R02." if args.dataset=="physionet" else
                           "UCI contains one person and a continuous recording, not multiple independent participants."])
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"RUN_COMPLETE.txt").write_text("REAL EEG TRAIN-ONLY OPTIMIZATION WITH VALIDATION CALIBRATION\n",encoding="utf-8")
    print("REAL-DATA-COMPLETE",args.dataset,flush=True)
    print(summary.to_string(index=False,float_format=lambda n:f"{n:.4f}"),flush=True)

if __name__=="__main__":main()

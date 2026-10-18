#!/usr/bin/env python3
"""End-to-end test on previously extracted REAL 8-subject, 3-run EDF epochs.

This checks that 4-fold × all eight strong baseline models complete on authentic
EEG features/waveforms, before the full 24-subject/72-EDF job finishes.
Not an independent confirmation or a model-selection mechanism.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import three_run_motor_imagery as exp

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=30)
    a=p.parse_args()
    src=Path(a.inputs)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    events=pd.read_csv(src/"events.csv")
    audit=pd.read_csv(src/"edf_audit.csv")
    with np.load(src/"real_epochs_float32.npz",allow_pickle=False) as d:
        wave=d["wave"].copy()
        y=d["y"].copy()
    assert wave.shape==(360,14,320),wave.shape
    assert len(events)==len(wave)==len(y)
    assert np.array_equal(y,events.label.to_numpy())
    assert events.subject.nunique()==8
    assert set(events.run.unique())=={4,8,12}
    assert len(audit)==24
    assert not events.duplicated(["subject","run","event_index"]).any()
    assert np.isfinite(wave).all()
    assert list(np.bincount(y,minlength=2))==[182,178]
    assert audit.n_events.sum()==len(events)
    manifest={"source_artifact":"https://github.com/ishchy/ishchy/actions/runs/39091649132/artifacts/12124756879",
              "source_sha256_by_file":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in src.iterdir() if f.is_file()},
              "n_real_subjects":8,"n_real_epochs":len(events),"labels":np.bincount(y,minlength=2).tolist(),
              "role":"End-to-end smoke only; never compare its model BAs to select 24-person final model"}
    (out/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("REAL_8_DATA_INTEGRITY_CHECK_OK",len(wave),flush=True)
    exp.run(wave,y,events,out,a.epochs,folds=4,smoke=False)
    assert (out/"RUN_COMPLETE.txt").is_file()
    print("REAL_8_EIGHT_MODEL_INTEGRATION_OK",flush=True)

if __name__=="__main__":main()

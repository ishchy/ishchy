#!/usr/bin/env python3
"""Data-only EDF annotation audit. NEVER trains models or uses ground-truth test outcome."""
import json,urllib.request,collections,time
from pathlib import Path
import mne
mne.set_log_level("ERROR")
P=Path("audit_edf")
P.mkdir(exist_ok=True)
rows=[]
for subject,run in ((1,4),(1,8),(1,12),(2,4),(2,8),(2,12),(3,4)):
    file=f"S{subject:03d}R{run:02d}.edf"
    url=f"https://physionet.org/files/eegmmidb/1.0.0/S{subject:03d}/{file}"
    path=P/file
    try:
        for n in range(4):
            try:
                req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 EEG-experiment-audit"})
                with urllib.request.urlopen(req,timeout=75) as f,path.open("wb") as g:
                    while c:=f.read(2**20):g.write(c)
                break
            except Exception as e:
                print("DOWNLOAD_ERROR",file,n+1,repr(e),flush=True)
                if n==3:raise
                time.sleep(2*(n+1))
        raw=mne.io.read_raw_edf(str(path),preload=False,verbose="ERROR")
        annotation_counts=dict(collections.Counter(map(str,raw.annotations.description)))
        events,event_id=mne.events_from_annotations(raw,event_id={"T1":1,"T2":2},verbose="ERROR")
        inferred_counts=dict(collections.Counter(map(int,events[:,2]))) if len(events) else {}
        print("EDF_AUDIT",file,"bytes",path.stat().st_size,"annotations",annotation_counts,"event_map",event_id,"events",inferred_counts,flush=True)
        rows.append(dict(subject=subject,run=run,bytes=path.stat().st_size,
                         n_annotations=len(raw.annotations),
                         annotation_counts=annotation_counts,
                         mapped_event_counts=inferred_counts))
        raw.close()
    except Exception as e:
        rows.append(dict(subject=subject,run=run,error=repr(e)))
        print("EDF_FATAL",file,repr(e),flush=True)
    finally:
        path.unlink(missing_ok=True)
(P/"annotation_audit.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
print("ANNOTATION_AUDIT_COMPLETE",len(rows),flush=True)

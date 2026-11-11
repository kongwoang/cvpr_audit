"""Sandbox paths, source snapshots and durable JSON state."""
import hashlib
import json
import os
from pathlib import Path

REMOTE=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
HERE=Path(__file__).resolve().parent

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,default=str)+'\n');tmp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hashes(root=REMOTE):
    files=list((root/'code/CVPR/src/vtf').glob('*.py'))+list((root/'scripts/A_method').glob('*.py'))
    files += [root/'out/k28_temporal/temporal_head.pt',root/'out/resnet18_v3a/image_refiner.pt',root/'models/yolov8s.pt']
    return {str(p.relative_to(root)):sha(p) for p in sorted(files)}

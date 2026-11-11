#!/usr/bin/env bash
# K35-0 provenance scan (GT / manifests only; no pipeline): every file under the CVPR workspace's result / data / log /
# cache / experiment trees is searched, by path and by content (binary-safe), for each candidate unit ID:
#   av1_train  the 65 Argoverse 1.1 tracking TRAIN log IDs (= Argoverse-HD train.json seq_dirs)
#   nusc       all 850 nuScenes trainval scene names (results/nusc_full/split.csv gives each scene's set / split)
# Output results/k35/k0/: ids_*.txt, path_hits.tsv (id, path), content_hits.tsv (id, file), dirs.txt (what was scanned).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
O=$B/results/k35/k0; mkdir -p $O; cd $B
python3 - <<'PY'
import json, pandas as pd
B = "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"
d = json.load(open(f"{B}/datasets/argoverse_hd/Argoverse-HD/annotations/train.json"))
ids = sorted({s.strip("/").split("/")[1] if "/" in s else s for s in d["seq_dirs"]})      # "train/<log id>/ring_front_center"
assert len(ids) == 65 and all(len(i) == 36 for i in ids), (len(ids), ids[:2])
open(f"{B}/results/k35/k0/ids_av1_train.txt", "w").write("\n".join(ids) + "\n")
sp = pd.read_csv(f"{B}/results/nusc_full/split.csv")
open(f"{B}/results/k35/k0/ids_nusc.txt", "w").write("\n".join(sp.seq) + "\n")
PY
DIRS="results data logs out archive experiments k10_masked/results deep_ext repro tmp code/risk-aware-perception/data code/risk-aware-perception/results code/CVPR scripts"
echo "$DIRS" > $O/dirs.txt
cat $O/ids_av1_train.txt $O/ids_nusc.txt > $O/ids_all.txt
find $DIRS -path results/k35 -prune -o -type f -print 2>/dev/null > $O/paths.txt
grep -oF -f $O/ids_all.txt $O/paths.txt > /dev/null 2>&1
awk 'NR==FNR {id[$1]; next} {for (i in id) if (index($0, i)) print i "\t" $0}' $O/ids_all.txt $O/paths.txt > $O/path_hits.tsv
grep -rloaF -f $O/ids_all.txt $DIRS --exclude-dir=k35 2>/dev/null > $O/content_files.txt
while read -r f; do grep -oaF -f $O/ids_all.txt "$f" 2>/dev/null | sort -u | awk -v f="$f" '{print $0 "\t" f}'; done < $O/content_files.txt > $O/content_hits.tsv
echo "K35 PROVENANCE DONE $(wc -l < $O/paths.txt) files, $(wc -l < $O/content_files.txt) with an ID" > $O/DONE

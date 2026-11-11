#!/usr/bin/env bash
# Argoverse 2 sensor dataset, one split (default val; 150 logs; train has 700), public S3: annotations, calibration,
# poses and the front-centre camera only.  Everything under $B/datasets/av2/sensor/<split>/<log>/ ; resumable.
# usage: download_av2.sh [split] [max_logs]
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
D=$B/datasets/av2
U=https://argoverse.s3.amazonaws.com
SPLIT=${1:-val}
MAXLOGS=${2:-100000}
mkdir -p $D/lists/$SPLIT
list () {   # all keys under a prefix (paginated S3 ListObjectsV2)
  local prefix=$1 tok="" out=$2
  : > $out
  while true; do
    if [ -z "$tok" ]; then r=$(curl -s "$U/?list-type=2&prefix=$prefix")
    else r=$(curl -s "$U/?list-type=2&prefix=$prefix&continuation-token=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$tok")"); fi
    echo "$r" | grep -o "<Key>[^<]*</Key>" | sed 's/<Key>\(.*\)<\/Key>/\1/' >> $out
    tok=$(echo "$r" | grep -o "<NextContinuationToken>[^<]*</NextContinuationToken>" | sed 's/<[^>]*>//g')
    [ -z "$tok" ] && break
  done
}
list_prefixes () {   # log prefixes of the split (paginated)
  local tok="" out=$1; : > $out
  while true; do
    if [ -z "$tok" ]; then r=$(curl -s "$U/?list-type=2&prefix=datasets/av2/sensor/$SPLIT/&delimiter=/")
    else r=$(curl -s "$U/?list-type=2&prefix=datasets/av2/sensor/$SPLIT/&delimiter=/&continuation-token=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$tok")"); fi
    echo "$r" | grep -o "<Prefix>datasets/av2/sensor/$SPLIT/[^<]*</Prefix>" | sed 's/<[^>]*>//g' | grep -v "^datasets/av2/sensor/$SPLIT/$" >> $out
    tok=$(echo "$r" | grep -o "<NextContinuationToken>[^<]*</NextContinuationToken>" | sed 's/<[^>]*>//g')
    [ -z "$tok" ] && break
  done
}
LL=$D/lists/$SPLIT/logs.txt
[ "$SPLIT" = val ] && [ -f $D/lists/logs.txt ] && LL=$D/lists/logs.txt || list_prefixes $LL
head -n $MAXLOGS $LL > $LL.sel
while read -r p; do
  log=$(basename $p)
  DL=$D/lists/$SPLIT; [ "$SPLIT" = val ] && DL=$D/lists
  [ -f $DL/$log.done ] && continue
  list "${p}" $DL/$log.keys
  grep -E "annotations.feather|calibration/|city_SE3_egovehicle.feather|sensors/cameras/ring_front_center/" $DL/$log.keys \
    | while read -r k; do
        f=$D/sensor/${k#datasets/av2/sensor/}
        [ -s "$f" ] && continue
        mkdir -p $(dirname $f)
        echo "$U/$k $f"
      done | xargs -P 16 -n 2 sh -c 'curl -s --retry 5 -o "$1" "$0"'
  touch $DL/$log.done
  echo "$(date +%T) $log"
done < $LL.sel
echo "$(date +%T) AV2 $(echo $SPLIT | tr a-z A-Z) DONE"

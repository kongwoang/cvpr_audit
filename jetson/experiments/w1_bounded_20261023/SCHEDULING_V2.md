# Scheduling amendment — 2026-10-23

nuScenes is complete; its VUW1 GPU is free. The user requested optimizing the
use of both VUW and VUW1. Preserve all atomic completed AV2 unit records, stop
only the two owned AV2 runners, wait for their supervisors to capture exits and
source hashes, and archive their metadata under jobs/retired/. In-flight units
are replayed; completed units are skipped by the unchanged runner.

Resume AV2 with four disjoint modulo shards, 0/4–3/4, two workers per unchanged
GPU wrapper on each server. Runtime/weights/thresholds/data/seed/report rows and
analysis are unchanged. No overlap with the stopped two-shard run is permitted.
Record all intentional stops and the retained file count in repartition.json.
Use coordinator_v2.py and group4.py; keep the previous coordinator/source intact.

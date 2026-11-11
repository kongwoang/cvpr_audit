# Scheduling-only amendment

nuScenes has completed all 152 units. Reuse its VUW1 capacity for remaining AV2
logs: preserve every completed atomic unit, stop only the two identified owned
AV2 runners and let their supervisors record exits/source checks. Archive those
intentionally retired job records. Resume four disjoint modulo shards, two per
unchanged GPU wrapper on each server. No overlap between old and new runners.
In-flight incomplete units are replayed. Method, inputs, weights, thresholds,
candidate rows and inference source hashes are unchanged. Record all stops and
retained-unit counts in repartition.json; coordinator_v2.py owns continuation.

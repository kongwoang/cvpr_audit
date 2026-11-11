# Execution notes (all negative/failed attempts retained)

- Initial AV2 training-data workers failed before producing units because the
  original masking launcher uses runpy and does not add the sandbox script folder
  to sys.path. Added explicit sandbox import path; retried the same shard inputs.
  Failed job statuses are under jobs/attempts and logs remain appended.
- First audit_data.py invocation failed on a duplicate dataset keyword when
  merging already-labelled metadata. Corrected the metadata merge; reran fully.
- First VUW1 original-alignment invocation raced the upload and found no script;
  rerun after upload completed. No records or policy were changed.
- One shell status probe had broken nested quoting and produced a SyntaxError;
  replaced with correctly quoted read-only status probe. No job was affected.
- Local read of nusc_complete summary raced report download; retried after sync.
- Source/protocol additions occur in separate sandbox runners; already launched
  workers keep their imported implementation. Every worker records its experiment
  hash snapshot. Main source/weights are independently checked before and after.
- Original quantile and validity-head weaknesses are negative results retained
  before calibration/separate classifier addenda. Original support-only adapters
  remain separate from subsequent joint-lift candidates.
- Final Jetson cross-backend CSV computation succeeded, but the first Markdown
  rendering attempted an unavailable optional tabulate dependency. Replaced with
  direct table formatting and reran; no environment or prediction changed.
- Once nuScenes and learned-AV2 workers completed, the independent AV2 persistence
  suite was scheduled early in two existing-wrapper groups (two shards each).
  It needs only already-frozen checkpoints, not lift-suite results. Existing
  per-shard locks prevent duplication when the serial parent reaches this stage;
  completion is checked from every individual stage job and full unit manifest.
- Compact deployment features were checked against cached bridge examples on
  real frames, CPU-only: max output difference 5.96e-8 nuScenes / 1.19e-7 AV2.
  The portable Jetson head uses CPU FP32; its GPU numerical parity is also checked
  inside the dedicated timing window. No engine was built.

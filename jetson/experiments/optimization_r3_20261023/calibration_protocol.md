# Addendum: interval calibration after training-holdout audit

The first adapter's nominal .1/.9 rate quantiles cover substantially less than
80% of heldout targets. Its auxiliary identity classifier also has little useful
rejection at .5; do not claim it solves association or gate histories with it.
This addendum is written after examining training holdouts and before running any
calibrated development policy. Initial adapter policies remain unchanged/reported.

Fit a nonnegative additive CQR residual correction to q10/q90, separately by
dataset, source kind and 1/2/3 observations (at least 100 calibration examples;
otherwise pool across the same dataset/model). Residual is max(q10-y,y-q90);
finite-sample order statistic ceil(.8*(n+1)), clipped to the available sample count.
Expand both endpoints by the correction, clamp [0,2]. Point/range estimates unchanged.

Audit coverage out of fold across five hash-defined UNIT folds; fit the deployable
correction on all original training holdouts. Those units already influenced epoch
selection (and the earlier frozen C1), so this is empirical calibration, without
a conformal/exchangeability/safety guarantee. No CFG-DEV or test labels enter fits.

Run the same upper-quantile missing-history and uncertainty-query policies with
calibrated intervals, alongside fresh v6-S/round-1 controls. Every original and
calibrated row is retained. Report both coverage/width and end-to-end false/missed
episodes, risk/G and request cost. Wider intervals can be expensive and may simply
buy fewer misses with more false alarms; this is a falsifier, not an assumed win.

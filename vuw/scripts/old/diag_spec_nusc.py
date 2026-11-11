import os, pandas as pd, numpy as np
B = os.environ["B"]; out = f"{B}/results/nusc_full"
e = pd.read_pickle(f"{out}/eval__old_test.pkl"); f = pd.read_pickle(f"{out}/flips__mono.pkl")
d = e.merge(f[["seq", "frame", "brake_piv_add", "brake_piv_remove", "brake_piv_move", "brake_piv_multi", "brake_piv_z", "brake_piv_lat_c", "brake_act_c", "brake_act_f", "v_causal"]], on=["seq", "frame"])
d["V"] = d.brake_V; d["Vs"] = d.J_c - d.J_spec_ref_resnet18
d["kind"] = np.select([d.brake_piv_add == 1, d.brake_piv_remove == 1, d.brake_piv_move == 1, d.brake_piv_multi == 1], ["add", "remove", "move", "multi"], "none")
d["zbin"] = pd.cut(d.brake_piv_z, [0, 10, 15, 25, 45, 200])
L = d.J_c.sum()
aff = d[d.V.abs() > 1e-9]
print("share of all-cheap loss, by pivot kind / range (uniform 640 vs speculative+refiner):")
print(aff.groupby(["kind", "zbin"], observed=True).agg(n=("V", "size"), uniform=("V", lambda x: x.sum() / L), spec=("Vs", lambda x: x.sum() / L)).round(4).to_string())
print("frames where speculative changes the decision but uniform does not:", int(((d.Vs.abs() > 1e-9) & (d.V.abs() <= 1e-9)).sum()), "value", round(d.loc[d.V.abs() <= 1e-9, "Vs"].sum() / L, 4))
print("crops on frames with uniform benefit:", d.loc[d.V > 1e-9, "n_crops"].value_counts().to_dict())

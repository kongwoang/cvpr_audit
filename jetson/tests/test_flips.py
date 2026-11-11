"""Invariants of the counterfactual element operations and the pivot search."""
import numpy as np

from rap import planner as P
from vtf import flips


def _branch(z, lo, hi, conf):
    """A Branch with mono geometry equal to consumer geometry, built without a cache."""
    b = flips.Branch.__new__(flips.Branch)
    n = len(z)
    b.xyxy = np.array([[10.0 * i, 10.0, 10.0 * i + 8, 30.0] for i in range(n)]).reshape(n, 4)
    b.conf = np.asarray(conf, float)
    b.all_xyxy, b.all_conf = b.xyxy.copy(), b.conf.copy()
    b.binent = b.entropy = np.zeros(n)
    b.coarse = np.array(["vehicle"] * n)
    ttc = np.full(n, np.inf)
    b.mono = {"z": np.asarray(z, float), "lat_min": np.asarray(lo, float), "lat_max": np.asarray(hi, float),
              "ttc": ttc, "z_ground": np.asarray(z, float), "z_height": np.asarray(z, float), "box_h": np.ones(n)}
    b.cons = b.cons_raw = (b.mono["z"], b.mono["lat_min"], b.mono["lat_max"], ttc)
    b.mono_t = b.cons
    return b


def test_apply_and_revert_are_the_two_ends_of_one_element():
    C = (np.array([10.0, 20.0]), np.array([-1.0, 3.0]), np.array([1.0, 4.0]), np.array([np.inf, np.inf]))
    F = (np.array([10.0, 12.0, 30.0]), np.array([-1.0, -0.5, 5.0]), np.array([1.0, 0.5, 6.0]),
         np.array([np.inf, np.inf, np.inf]))
    add = ("add", -1, 1)
    assert len(flips.apply_to_cheap(C, F, add)[0]) == 3
    assert len(flips.revert_in_full(C, F, add)[0]) == 2
    rem = ("remove", 1, -1)
    assert len(flips.apply_to_cheap(C, F, rem)[0]) == 1
    assert len(flips.revert_in_full(C, F, rem)[0]) == 4
    mv = ("move", 0, 0)
    assert np.array_equal(flips.apply_to_cheap(C, F, mv)[0], np.array([10.0, 20.0]))


def test_a_near_in_corridor_addition_is_the_sufficient_and_necessary_pivot():
    pp = P.PlannerParams()
    v = 10.0
    cheap = _branch([60.0], [5.0], [6.0], [0.9])                  # far and out of the corridor
    full = _branch([60.0, 12.0], [5.0, -0.5], [6.0, 0.5], [0.9, 0.4])
    pairs, added, removed = flips.match(full, cheap)
    assert added == [1] and removed == [] and len(pairs) == 1
    els = flips.elements(cheap, full, pairs, added, removed)
    decide = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)   # noqa: E731
    act_f = decide(full.cons)
    assert act_f != decide(cheap.cons)
    e, kind, ns, nn = flips.find_pivot(cheap, full, els, decide, act_f, v, pp)
    assert e == ("add", -1, 1) and kind == "add" and ns == 1 and nn == 1

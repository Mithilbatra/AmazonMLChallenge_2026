import numpy as np
import pandas as pd
import pytest

from src.data_io import detect_list_format, parse_id_list
from src.decision import DecisionParams, DecisionTable, plateau_center, search_decision
from src.graph_cleanup import graph_keep_mask
from src.labels import attach_labels, build_label_index
from src.metrics import entity_fbeta, macro_fbeta_sets, macro_from_arrays
from src.splits import make_splits

LABEL_CFG = {"schema": {"labels": {"source1_id": "source1_id", "match_columns": ["matches"]}}}


@pytest.mark.parametrize("raw,expected", [
    ("a,b", ["a", "b"]), ("a, b ,", ["a", "b"]), ("", []), ("[]", []), ("[a, b]", ["a", "b"]),
    ("['a','b']", ["a", "b"]), ('["a", "b", "a"]', ["a", "b"]), (None, []), ("nan", []),
])
def test_parse_id_list(raw, expected):
    assert parse_id_list(raw) == expected


def test_detect_list_format():
    assert detect_list_format(pd.Series(["[a]", "[]", "[b, c]"])) == "bracketed"
    assert detect_list_format(pd.Series(["a", "", "b,c"])) == "plain"


def test_label_index_and_negative_safety(tiny_tables):
    s1, v = tiny_tables
    labels = pd.DataFrame({"s1_id": ["A1", "A2", "A3"], "matches": [["B1", "C1"], ["B3"], []]})
    labels["all_ids"] = labels["matches"]
    li = build_label_index(labels, s1, v, LABEL_CFG)
    assert li.n_true.tolist() == [2, 1, 0]
    assert li.is_singleton().tolist() == [False, False, True]
    assert set(zip(li.true_pairs["s1_idx"], li.true_pairs["v_idx"])) == {(0, 0), (0, 3), (1, 2)}
    pairs = pd.DataFrame({"s1_idx": [0, 0, 1, 2], "v_idx": [0, 1, 2, 4]})
    assert attach_labels(pairs, li, v).tolist() == [1, 0, 1, 0]


def test_missing_s1_in_labels_is_singleton(tiny_tables):
    s1, v = tiny_tables
    labels = pd.DataFrame({"s1_id": ["A1"], "matches": [["B1"]]})
    labels["all_ids"] = labels["matches"]
    li = build_label_index(labels, s1, v, LABEL_CFG)
    assert li.labeled.all() and li.n_true.tolist() == [1, 0, 0]


def test_fbeta_values():
    b = 0.5
    # singleton: empty prediction = 1, any prediction = 0
    assert entity_fbeta([0], [0], [0], b)[0] == 1.0
    assert entity_fbeta([0], [1], [0], b)[0] == 0.0
    # non-singleton predicted empty = 0
    assert entity_fbeta([0], [0], [2], b)[0] == 0.0
    # P = 1, R = 0.5 -> 1.25*0.5/(0.25+0.5)
    assert entity_fbeta([1], [1], [2], b)[0] == pytest.approx(0.625 / 0.75)
    # P = 0.5, R = 1 -> 1.25*0.5/(0.125+1)
    assert entity_fbeta([1], [2], [1], b)[0] == pytest.approx(0.625 / 1.125)


def test_macro_is_unweighted_mean_over_entities():
    pred = {"e1": {"a"}, "e2": set(), "e3": {"x"}}
    true = {"e1": {"a", "b"}, "e2": set(), "e3": set()}
    res = macro_fbeta_sets(pred, true, ["e1", "e2", "e3"])
    assert res["macro_f0.5"] == pytest.approx((0.625 / 0.75 + 1 + 0) / 3)
    assert res["singleton_false_merges"] == 1


def test_fast_macro_equals_reference():
    rng = np.random.default_rng(0)
    E = 50
    ent = rng.integers(0, E, 400)
    is_true = rng.random(400) < 0.2
    sel = rng.random(400) < 0.3
    n_true = np.bincount(ent[is_true], minlength=E) + rng.integers(0, 2, E)  # + matches lost in blocking
    fast = macro_from_arrays(ent, sel, is_true, n_true)
    tp = np.bincount(ent[sel & is_true], minlength=E)
    npred = np.bincount(ent[sel], minlength=E)
    assert fast == pytest.approx(entity_fbeta(tp, npred, n_true).mean())


def _table():
    pairs = pd.DataFrame({"s1_idx": [0, 0, 1, 2, 2], "v_idx": [0, 1, 2, 3, 4], "source": [2, 3, 2, 2, 3],
                          "p": [0.95, 0.40, 0.30, 0.90, 0.85], "is_true": [1, 0, 0, 1, 1]})
    return pairs, DecisionTable.build(pairs, "p", np.array([0, 1, 2, 3]), np.array([1, 0, 2, 0]))


def test_decision_rules():
    pairs, t = _table()
    sel = t.select(DecisionParams(threshold_s2=0.5, threshold_s3=0.5))
    assert sel.tolist() == [True, False, False, True, True]
    assert t.score(DecisionParams(threshold_s2=0.5, threshold_s3=0.5)) == 1.0
    sel = t.select(DecisionParams(threshold_s2=0.2, threshold_s3=0.2, max_per_source=1))
    assert sel.tolist() == [True, True, True, True, True]
    sel = t.select(DecisionParams(threshold_s2=0.2, threshold_s3=0.2, margin=0.1))
    assert sel.tolist() == [True, False, True, True, True]
    sel = t.select(DecisionParams(threshold_s2=0.2, threshold_s3=0.2, gate=0.5))
    assert sel.tolist() == [True, True, False, True, True]


def test_search_finds_good_threshold():
    _, t = _table()
    prm, score, trace = search_decision(t, {"threshold_grid": {"start": 0.05, "stop": 0.99, "step": 0.05},
                                            "rounds": 2, "margins": [1.0], "gate_offsets": [0.0],
                                            "max_per_source": [None]})
    assert score == 1.0 and 0.4 <= prm.threshold_s2 < 0.85


def test_plateau_center():
    assert plateau_center([1, 2, 3, 4, 5], [0, 1, 1, 1, 0]) == 3


def test_graph_cleanup_resolves_vendor_linked_to_two_s1():
    # vendor 10 linked to S1 0 (strong) and S1 1 (weak): S1 is de-duplicated -> keep only one
    pairs = pd.DataFrame({"s1_idx": [0, 1, 0, 1], "v_idx": [10, 10, 11, 12], "p": [0.95, 0.6, 0.9, 0.8]})
    for mode in ("exclusive", "betweenness"):
        keep, stats = graph_keep_mask(pairs, "p", mode, 0.2, 50)
        assert keep.tolist() == [True, False, True, True], mode
    keep, _ = graph_keep_mask(pairs, "p", False, 0.2, 50)   # YAML `off` -> False
    assert keep.all()


def test_star_component_untouched():
    pairs = pd.DataFrame({"s1_idx": [0, 0, 0], "v_idx": [1, 2, 3], "p": [0.9, 0.8, 0.7]})
    keep, stats = graph_keep_mask(pairs, "p", "betweenness", 0.2, 50)
    assert keep.all() and stats["edges_removed"] == 0


def test_splits_are_entity_level_and_disjoint(tiny_tables):
    s1, v = tiny_tables
    labels = pd.DataFrame({"s1_id": ["A1", "A2", "A3"], "matches": [["B1", "C1"], ["B3"], []]})
    labels["all_ids"] = labels["matches"]
    li = build_label_index(labels, s1, v, LABEL_CFG)
    split = make_splits(li, v["source"], {"fit": 0.34, "calib": 0.33, "holdout": 0.33}, 0.0, 0)
    assert len(split) == 3 and set(split) <= {"fit", "fit_val", "calib", "holdout"}

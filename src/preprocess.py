"""Apply name normalisation and offline address parsing to all records.

Produces two tables per dataset:
  * ``s1``: Source-1 records (row index = ``s1_idx``)
  * ``v``:  Source-2 and Source-3 records stacked (row index = ``v_idx``),
            with a ``source`` column (2 or 3).
"""
from __future__ import annotations

import pandas as pd

from .address_parser import ADDRESS_FIELDS, AddressParser
from .normalization import NAME_FIELDS, NameNormalizer
from .utils import get_logger


def _apply_unique(values: pd.Series, fn, fields: list[str]) -> pd.DataFrame:
    """Run `fn` once per unique string (vendor data is highly repetitive)."""
    uniques = pd.unique(values)
    parsed = {u: fn(u) for u in uniques}
    rows = [parsed[x] for x in values]
    return pd.DataFrame(rows, columns=fields, index=values.index)


def normalize_records(df: pd.DataFrame, normalizer: NameNormalizer, parser: AddressParser) -> pd.DataFrame:
    names = _apply_unique(df["name"], normalizer, NAME_FIELDS)
    addrs = _apply_unique(df["address"], parser, ADDRESS_FIELDS)
    out = pd.concat([df.reset_index(drop=True), names.reset_index(drop=True),
                     addrs.reset_index(drop=True)], axis=1)
    out["full_text"] = (out["name_core"] + " " + out["addr_clean"]).str.strip()
    out["addr_street_key"] = [
        f"{hn}|{_first_significant(road)}" if hn and road else ""
        for hn, road in zip(out["addr_house_number"], out["addr_road"])
    ]
    out["city_state_key"] = [
        f"{c}|{s}" if c else "" for c, s in zip(out["addr_city"], out["addr_state"])
    ]
    return out


def _first_significant(road: str) -> str:
    for tok in road.split():
        if tok not in {"n", "s", "e", "w", "ne", "nw", "se", "sw", "the"}:
            return tok
    return road.split()[0] if road else ""


def preprocess_dataset(sources: dict[int, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    log = get_logger()
    normalizer = NameNormalizer(cfg)
    parser = AddressParser(cfg)
    log.info("Address parser backend: %s", "libpostal (offline)" if parser.use_libpostal else "offline rules")
    s1 = normalize_records(sources[1], normalizer, parser)
    s1["s1_idx"] = range(len(s1))
    v = pd.concat([normalize_records(sources[2], normalizer, parser),
                   normalize_records(sources[3], normalizer, parser)], ignore_index=True)
    v["v_idx"] = range(len(v))
    v["key"] = v["source"].astype(str) + ":" + v["id"]
    stats = {}
    for name, tab in (("source1", s1), ("vendors", v)):
        stats[name] = {
            "records": int(len(tab)),
            "empty_core_name": int((tab["name_core"] == "").sum()),
            "with_legal_suffix": int((tab["name_suffix"] != "").sum()),
            "with_postcode": int((tab["addr_postcode"] != "").sum()),
            "with_house_number": int((tab["addr_house_number"] != "").sum()),
            "with_city": int((tab["addr_city"] != "").sum()),
            "with_state": int((tab["addr_state"] != "").sum()),
            "with_unit": int((tab["addr_unit"] != "").sum()),
            "with_landmark": int((tab["addr_landmark"] != "").sum()),
            "parse_method": tab["addr_parse_method"].value_counts().to_dict(),
            "top_suffixes": tab["name_suffix"].replace("", pd.NA).dropna().value_counts().head(15).to_dict(),
        }
    log.info("Normalised %d S1 and %d vendor records (%d S2 / %d S3)", len(s1), len(v),
             int((v["source"] == 2).sum()), int((v["source"] == 3).sum()))
    return s1, v, stats

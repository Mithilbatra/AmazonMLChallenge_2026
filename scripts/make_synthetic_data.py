#!/usr/bin/env python
"""Generate a small SYNTHETIC dataset with the challenge's structure so the
whole pipeline can be exercised end to end before the real data is used.

It is NOT challenge data and says nothing about real performance. Structure:
  * Source 1: clean, de-duplicated reference records (US / India / UK / DE)
  * Source 2: light vendor noise; Source 3: heavier noise
  * 0, 1 or 2 matches per source per entity; ~30% singletons
  * hard negatives: chain branches (same brand, other address), different
    businesses at the same address, near-name distractors
  * labels: one row per Source-1 entity -> comma-separated matching ids

    python scripts/make_synthetic_data.py --out data/synthetic --n-train 800 --n-test 400
"""
import argparse
import os
import random

import pandas as pd

SYL = ["zor", "vex", "ta", "lin", "mar", "kor", "pix", "del", "ran", "sol", "tek", "nov", "bri", "qua", "hel",
       "dor", "mex", "ri", "sa", "phon", "lum", "cap", "gen", "tor", "vis", "ar", "om", "zen", "kal", "fer"]
SURNAMES = ["Sharma", "Patel", "Miller", "Johnson", "Schmidt", "Garcia", "Iyer", "Reddy", "Brown", "Wagner",
            "Kumar", "Singh", "Taylor", "Fischer", "Nair", "Gupta", "Wilson", "Becker", "Mehta", "Clark"]
INDUSTRY = ["Robotics", "Manufacturing", "Technologies", "Traders", "Enterprises", "Consulting", "Foods",
            "Logistics", "Pharmaceuticals", "Services", "Solutions", "Industries", "Holdings", "Bakery",
            "Motors", "Textiles", "Electricals", "Hardware", "Associates", "International", "Systems",
            "Dental Clinic", "Pharmacy", "Cafe", "Studio", "Brothers", "Engineering", "Laboratories"]
ABBREV = {"Manufacturing": "Mfg", "Technologies": "Tech", "International": "Intl", "Services": "Svcs",
          "Brothers": "Bros", "Associates": "Assoc", "Enterprises": "Ent", "Holdings": "Hldgs",
          "Industries": "Inds", "Engineering": "Engg", "Laboratories": "Labs", "Systems": "Sys",
          "Pharmaceuticals": "Pharma", "Consulting": "Consultants"}
REGIONS = {
    "US": {
        "cities": [("San Jose", "CA", "California", "951"), ("Austin", "TX", "Texas", "787"),
                   ("Seattle", "WA", "Washington", "981"), ("Denver", "CO", "Colorado", "802"),
                   ("Boston", "MA", "Massachusetts", "021"), ("Portland", "OR", "Oregon", "972")],
        "streets": ["Market", "Elm", "Oak", "Maple", "Main", "Park", "Lake", "Hill", "Cedar", "Pine", "5th", "Mission"],
        "types": [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Drive", "Dr"), ("Lane", "Ln")],
        "suffixes": [["Inc", "Inc.", "Incorporated"], ["LLC", "L.L.C."], ["Corp", "Corporation"], ["Co", "Company"]],
    },
    "IN": {
        "cities": [("Bengaluru", "", "Karnataka", "560"), ("Mumbai", "", "Maharashtra", "400"),
                   ("Chennai", "", "Tamil Nadu", "600"), ("Pune", "", "Maharashtra", "411"),
                   ("Kolkata", "", "West Bengal", "700"), ("Gurgaon", "", "Haryana", "122")],
        "streets": ["MG", "Station", "Temple", "Ring", "Nehru", "Gandhi", "Lake View", "Church", "Hosur", "Link"],
        "types": [("Road", "Rd"), ("Marg", "Marg"), ("Main Road", "Main Rd")],
        "areas": ["Indiranagar", "Koramangala", "Andheri East", "Bandra West", "Salt Lake", "Anna Nagar",
                  "Sector 18", "Kothrud", "Whitefield", "Powai"],
        "landmarks": ["Bus Stand", "Railway Station", "City Mall", "Hanuman Temple", "Post Office", "City Hall"],
        "suffixes": [["Pvt Ltd", "Private Limited", "Pvt. Ltd."], ["LLP"], ["Ltd", "Limited"]],
    },
    "UK": {
        "cities": [("London", "", "", "SW1"), ("Manchester", "", "", "M1"), ("Leeds", "", "", "LS1"),
                   ("Bristol", "", "", "BS1")],
        "streets": ["Downing", "King", "Queen", "High", "Church", "Station", "Victoria"],
        "types": [("Street", "St"), ("Road", "Rd"), ("Lane", "Ln")],
        "suffixes": [["Ltd", "Limited"], ["PLC", "plc"], ["LLP"]],
    },
    "DE": {
        "cities": [("Berlin", "", "", "101"), ("Hamburg", "", "", "201"), ("München", "", "", "803")],
        "streets": ["Haupt", "Bahnhof", "Garten", "Schiller", "Goethe", "Linden"],
        "types": [("straße", "str.")],
        "suffixes": [["GmbH"], ["AG"], ["GmbH & Co. KG"]],
    },
}


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def brand(self):
        r = self.r
        return "".join(r.choice(SYL) for _ in range(r.choice([2, 2, 3]))).capitalize()

    def business(self, region):
        r = self.r
        reg = REGIONS[region]
        style = r.random()
        if style < 0.55:
            core = f"{self.brand()} {r.choice(INDUSTRY)}"
        elif style < 0.8:
            core = f"{r.choice(SURNAMES)} {r.choice(INDUSTRY)}"
        else:
            core = f"{r.choice(SURNAMES)} & Sons {r.choice(INDUSTRY)}"
        suffix_group = r.choice(reg["suffixes"])
        return {"core": core, "suffix_group": suffix_group, "suffix": suffix_group[0] if r.random() < 0.85 else ""}

    def address(self, region):
        r = self.r
        reg = REGIONS[region]
        city = r.choice(reg["cities"])
        a = {"region": region, "num": str(r.randint(1, 2500)), "street": r.choice(reg["streets"]),
             "type": r.choice(reg["types"]), "city": city[0], "state_code": city[1], "state": city[2],
             "unit": f"Suite {r.randint(100, 900)}" if (region == "US" and r.random() < 0.25) else ""}
        if region == "US":
            a["post"] = city[3] + f"{r.randint(0, 99):02d}"
        elif region == "IN":
            a["post"] = city[3] + f"{r.randint(0, 999):03d}"
            a["area"] = r.choice(reg["areas"])
            a["landmark"] = r.choice(reg["landmarks"]) if r.random() < 0.5 else ""
            a["unit"] = f"Shop No {r.randint(1, 40)}" if r.random() < 0.3 else ""
        elif region == "UK":
            a["post"] = f"{city[3]}{r.choice('ABCDEFGH')} {r.randint(1, 9)}{r.choice('ABDEFGHJ')}{r.choice('LNPQRSTU')}"
        else:
            a["post"] = city[3] + f"{r.randint(0, 99):02d}"
        return a


def render_name(b, noise, g):
    r = g.r
    core = b["core"]
    suffix = b["suffix"]
    if noise:
        ops = r.sample(["suffix", "abbrev", "case", "typo", "drop", "and", "the", "punct", "typo2", "brand_only"],
                       k=min(noise, 10))
        for op in ops:
            if op == "suffix":
                suffix = r.choice(b["suffix_group"] + [""])
            elif op == "abbrev":
                for k, val in ABBREV.items():
                    core = core.replace(k, val)
            elif op == "case":
                core, suffix = (core.upper(), suffix.upper()) if r.random() < 0.5 else (core.lower(), suffix.lower())
            elif op == "typo":
                core = typo(core, r)
            elif op == "typo2":
                core = typo(typo(core, r), r)
            elif op == "brand_only":
                core = core.split()[0]
            elif op == "drop":
                toks = core.split()
                if len(toks) > 2:
                    toks.pop(r.randrange(1, len(toks)))
                    core = " ".join(toks)
            elif op == "and":
                core = core.replace("&", "and") if "&" in core else core
            elif op == "the":
                core = "The " + core
            elif op == "punct":
                suffix = (", " + suffix) if suffix else suffix
    return f"{core} {suffix}".replace(" ,", ",").strip()


def typo(text, r):
    if len(text) < 5:
        return text
    i = r.randrange(1, len(text) - 2)
    op = r.random()
    if op < 0.33:
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if op < 0.66:
        return text[:i] + text[i + 1:]
    return text[:i] + r.choice("aeiourstln") + text[i:]


def render_address(a, noise, g):
    r = g.r
    reg = a["region"]
    ops = set(r.sample(["abbrev", "state", "drop_post", "drop_city", "unit", "typo", "landmark", "case",
                        "drop_num", "space"], k=noise)) if noise else set()
    t_full, t_short = a["type"]
    stype = t_short if "abbrev" in ops else t_full
    street = typo(a["street"], r) if "typo" in ops and len(a["street"]) > 4 else a["street"]
    num = "" if "drop_num" in ops else a["num"]
    post = "" if "drop_post" in ops else a["post"]
    city = "" if "drop_city" in ops else a["city"]
    unit = a.get("unit", "")
    if "unit" in ops:
        unit = "" if unit else ("Ste " + str(r.randint(100, 900)) if reg == "US" else unit)
    if reg == "US":
        state = a["state"] if "state" in ops else a["state_code"]
        parts = [f"{num} {street} {stype}".strip() + (f" {unit}" if unit else ""), city, f"{state} {post}".strip()]
    elif reg == "IN":
        state = a["state"]
        if "space" in ops and post:
            post = post[:3] + " " + post[3:]
        lm = a.get("landmark", "")
        if "landmark" in ops:
            lm = lm or r.choice(REGIONS["IN"]["landmarks"])
        parts = [unit, f"No {num}" if num and r.random() < 0.5 else num, f"{street} {stype}",
                 f"Near {lm}" if lm else "", a["area"], city, f"{state} {post}".strip()]
    elif reg == "UK":
        parts = [f"{num} {street} {stype}".strip(), f"{city} {post}".strip()]
    else:
        parts = [f"{street}{stype} {num}".strip(), f"{post} {city}".strip()]
    text = ", ".join(p for p in parts if p)
    if "case" in ops:
        text = text.upper()
    return text


def generate(n, seed, id_offset, test=False):
    g = Gen(seed)
    r = g.r
    s1, s2, s3, labels = [], [], [], []
    c1 = c2 = c3 = 0
    entities = []
    for i in range(n):
        region = r.choices(["US", "IN", "UK", "DE"], weights=[0.4, 0.35, 0.15, 0.1])[0]
        b = g.business(region)
        a = g.address(region)
        entities.append((b, a))
        # chain branch: same brand, another address, separate S1 entity
        if r.random() < 0.25 and len(entities) < n:
            entities.append(({**b}, g.address(region)))
    entities = entities[:n]
    for b, a in entities:
        c1 += 1
        s1_id = f"A{id_offset + c1:07d}"
        s1.append({"record_id": s1_id, "business_name": render_name(b, 0, g), "full_address": render_address(a, 0, g)})
        k2 = r.choices([0, 1, 2], weights=[0.35, 0.55, 0.10])[0]
        k3 = r.choices([0, 1, 2], weights=[0.45, 0.47, 0.08])[0]
        if r.random() < 0.18:
            k2 = k3 = 0   # extra singletons
        matches = []
        for _ in range(k2):
            c2 += 1
            vid = f"B{id_offset + c2:07d}"
            s2.append({"record_id": vid, "business_name": render_name(b, r.randint(0, 3), g),
                       "full_address": render_address(a, r.randint(0, 3), g)})
            matches.append(vid)
        for _ in range(k3):
            c3 += 1
            vid = f"C{id_offset + c3:07d}"
            s3.append({"record_id": vid, "business_name": render_name(b, r.randint(1, 4), g),
                       "full_address": render_address(a, r.randint(1, 5), g)})
            matches.append(vid)
        labels.append({"source1_id": s1_id, "matched_ids": ",".join(matches)})
        # hard negatives in vendor data
        if r.random() < 0.25:   # other business at the same address
            other = g.business(a["region"])
            c2 += 1
            s2.append({"record_id": f"B{id_offset + c2:07d}", "business_name": render_name(other, 1, g),
                       "full_address": render_address(a, 1, g)})
        if r.random() < 0.30:   # same brand, different location (not in S1)
            c3 += 1
            s3.append({"record_id": f"C{id_offset + c3:07d}", "business_name": render_name(b, 1, g),
                       "full_address": render_address(g.address(a["region"]), 1, g)})
    # unrelated vendor records
    for _ in range(int(0.3 * n)):
        region = r.choice(list(REGIONS))
        c2 += 1
        s2.append({"record_id": f"B{id_offset + c2:07d}", "business_name": render_name(g.business(region), 1, g),
                   "full_address": render_address(g.address(region), 1, g)})
        c3 += 1
        s3.append({"record_id": f"C{id_offset + c3:07d}", "business_name": render_name(g.business(region), 2, g),
                   "full_address": render_address(g.address(region), 2, g)})
    shuf = lambda rows: r.sample(rows, len(rows))
    return (pd.DataFrame(shuf(s1)), pd.DataFrame(shuf(s2)), pd.DataFrame(shuf(s3)), pd.DataFrame(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    for split, n, seed, off in (("train", args.n_train, args.seed, 0), ("test", args.n_test, args.seed + 1, 5_000_000)):
        s1, s2, s3, lab = generate(n, seed, off, test=split == "test")
        d = os.path.join(args.out, split)
        os.makedirs(d, exist_ok=True)
        for name, df in (("source1", s1), ("source2", s2), ("source3", s3)):
            df.to_csv(os.path.join(d, f"{name}.tsv"), sep="\t", index=False)
        if split == "train":
            lab.to_csv(os.path.join(d, "labels.tsv"), sep="\t", index=False)
        else:  # kept apart: never read by the pipeline, only by the demo scorer
            lab.to_csv(os.path.join(args.out, "test_labels_HIDDEN.tsv"), sep="\t", index=False)
        print(f"{split}: S1={len(s1)} S2={len(s2)} S3={len(s3)} singletons={(lab['matched_ids'] == '').sum()}")


if __name__ == "__main__":
    main()

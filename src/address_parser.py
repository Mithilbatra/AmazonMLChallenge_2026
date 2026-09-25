"""Offline address normalisation and parsing (PDF: "Statistical Address Parsing
in an Offline Context").

Two back-ends, both 100% OFFLINE:

* ``libpostal`` (optional, via the `postal` Python bindings). libpostal is a
  local C library + model files; parsing makes no network call. It is used
  when installed and `address.parser` is `auto` or `libpostal`.
* A rule-based multi-region fallback (always available) that extracts the
  same components: house_number, road, unit, city, state, postcode, country,
  landmark ("house" in libpostal terms, e.g. "Nr City Hall") and suburb.

NEVER used: geocoders, map APIs, OpenStreetMap queries or any other online
lookup - the challenge forbids external databases, APIs and lookups.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .normalization import MARK_RANGES, strip_latin_marks
from .utils import get_logger

try:  # optional offline statistical parser
    from postal.parser import parse_address as _lp_parse  # type: ignore
except Exception:  # pragma: no cover - libpostal usually not installed
    _lp_parse = None

ADDRESS_FIELDS = ["addr_clean", "addr_house_number", "addr_road", "addr_unit", "addr_city",
                  "addr_state", "addr_postcode", "addr_country", "addr_landmark", "addr_suburb",
                  "addr_numbers", "addr_parse_method"]

ADDRESS_ABBREVIATIONS: dict[str, str] = {
    "street": "st", "st": "st", "strt": "st", "stree": "st",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave", "avn": "ave", "avenu": "ave",
    "road": "rd", "rd": "rd", "raod": "rd",
    "boulevard": "blvd", "blvd": "blvd", "boul": "blvd", "blv": "blvd",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "crt": "ct",
    "place": "pl", "pl": "pl",
    "square": "sq", "sq": "sq",
    "highway": "hwy", "hwy": "hwy", "hiway": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy",
    "terrace": "ter", "ter": "ter", "terr": "ter",
    "circle": "cir", "cir": "cir", "crescent": "cres", "cres": "cres",
    "trail": "trl", "trl": "trl", "alley": "aly", "aly": "aly",
    "strasse": "str", "str": "str", "gasse": "g", "weg": "weg", "platz": "pl",
    "suite": "ste", "ste": "ste", "suit": "ste",
    "apartment": "apt", "apartments": "apt", "apt": "apt",
    "building": "bldg", "bldg": "bldg", "bld": "bldg",
    "floor": "fl", "fl": "fl", "flr": "fl",
    "room": "rm", "rm": "rm", "office": "ofc", "ofc": "ofc",
    "number": "no", "no": "no", "num": "no", "nbr": "no", "nr.": "no",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "near": "nr", "nr": "nr", "nearby": "nr",
    "opposite": "opp", "opp": "opp", "opposit": "opp",
    "behind": "behind", "beside": "beside", "adjacent": "adj", "adj": "adj",
    "junction": "jn", "jn": "jn", "jct": "jn",
    "extension": "extn", "extn": "extn", "ext": "extn",
    "industrial": "indl", "indl": "indl",
    "market": "mkt", "mkt": "mkt",
    "block": "blk", "blk": "blk", "sector": "sec", "sec": "sec",
    "mount": "mt", "mt": "mt", "fort": "ft", "heights": "hts", "hts": "hts",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
}
PHRASE_ABBREVIATIONS = [
    ("post office box", "pobox"), ("p o box", "pobox"), ("po box", "pobox"),
    ("next to", "nr"), ("close to", "nr"), ("in front of", "opp"), ("near by", "nr"),
]
STREET_TYPES = frozenset({"st", "ave", "rd", "blvd", "dr", "ln", "ct", "pl", "sq", "hwy", "pkwy",
                          "expy", "fwy", "ter", "cir", "cres", "trl", "aly", "str", "weg", "way",
                          "marg", "path", "gali", "chowk", "main", "cross", "row", "walk", "close",
                          "mews", "gate", "view", "rise", "loop", "run", "pike", "plaza", "calle",
                          "rua", "via", "avenida", "carrera"})
LANDMARK_MARKERS = frozenset({"nr", "opp", "behind", "beside", "adj", "above", "below", "off",
                              "landmark", "via"})
UNIT_MARKERS = ("ste", "apt", "unit", "rm", "fl", "ofc", "shop", "flat", "bldg", "blk", "door",
                "office", "room", "suite", "gala", "stall", "kiosk", "pobox")
DIRECTIONALS = frozenset({"n", "s", "e", "w", "ne", "nw", "se", "sw"})

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr", "guam": "gu",
}
CA_PROVINCES = {
    "alberta": "ab", "british columbia": "bc", "manitoba": "mb", "new brunswick": "nb",
    "newfoundland and labrador": "nl", "nova scotia": "ns", "ontario": "on",
    "prince edward island": "pe", "quebec": "qc", "saskatchewan": "sk",
    "northwest territories": "nt", "nunavut": "nu", "yukon": "yt",
}
AU_STATES = {
    "new south wales": "nsw", "victoria": "vic", "queensland": "qld", "western australia": "wa",
    "south australia": "sa", "tasmania": "tas", "australian capital territory": "act",
    "northern territory": "nt",
}
# Indian states map full names -> codes; 2-letter Indian codes are NOT matched
# as bare tokens because they collide with common words ("up", "as", "ga").
IN_STATES = {
    "andhra pradesh": "in-ap", "arunachal pradesh": "in-ar", "assam": "in-as", "bihar": "in-br",
    "chhattisgarh": "in-cg", "goa": "in-ga", "gujarat": "in-gj", "haryana": "in-hr",
    "himachal pradesh": "in-hp", "jharkhand": "in-jh", "karnataka": "in-ka", "kerala": "in-kl",
    "madhya pradesh": "in-mp", "maharashtra": "in-mh", "manipur": "in-mn", "meghalaya": "in-ml",
    "mizoram": "in-mz", "nagaland": "in-nl", "odisha": "in-od", "orissa": "in-od",
    "punjab": "in-pb", "rajasthan": "in-rj", "sikkim": "in-sk", "tamil nadu": "in-tn",
    "tamilnadu": "in-tn", "telangana": "in-ts", "tripura": "in-tr", "uttar pradesh": "in-up",
    "uttarakhand": "in-uk", "uttaranchal": "in-uk", "west bengal": "in-wb",
    "jammu and kashmir": "in-jk", "ladakh": "in-la", "puducherry": "in-py",
    "pondicherry": "in-py", "chandigarh": "in-ch", "national capital territory of delhi": "in-dl",
    "nct of delhi": "in-dl", "andaman and nicobar islands": "in-an", "lakshadweep": "in-ld",
    "dadra and nagar haveli and daman and diu": "in-dn",
}
COUNTRIES = {
    "usa": "us", "us": "us", "united states": "us", "united states of america": "us",
    "india": "in", "bharat": "in", "uk": "gb", "united kingdom": "gb", "great britain": "gb",
    "england": "gb", "scotland": "gb", "wales": "gb", "germany": "de", "deutschland": "de",
    "canada": "ca", "australia": "au", "mexico": "mx", "france": "fr", "spain": "es",
    "espana": "es", "italy": "it", "italia": "it", "netherlands": "nl", "singapore": "sg",
    "uae": "ae", "united arab emirates": "ae", "china": "cn", "japan": "jp", "brazil": "br",
    "brasil": "br", "ireland": "ie", "new zealand": "nz", "south africa": "za",
}

DEFAULT_POSTCODE_PATTERNS = [
    # (name, regex, numeric_only). Order = priority on ties.
    ("us_zip4", r"(?<![\w-])(\d{5})-\d{4}(?![\w-])", True),
    ("ca", r"(?<![\w-])([a-z]\d[a-z] ?\d[a-z]\d)(?![\w-])", False),
    ("uk", r"(?<![\w-])([a-z]{1,2}\d[a-z\d]? ?\d[a-z]{2})(?![\w-])", False),
    ("in_pin", r"(?<![\w-])(\d{3} ?\d{3})(?![\w-])", True),
    ("five_digit", r"(?<![\w-])(\d{5})(?![\w-])", True),
    ("four_digit", r"(?<![\w-])(\d{4})(?![\w-])", True),
]

_SEG_SPLIT_RE = re.compile(r"[,;|\n\r]+")
_APOS_RE = re.compile(r"['’‘`´]")
_ACRONYM_RE = re.compile(r"(?<![\w])([^\W\d_])(?:\.([^\W\d_]))+\.?(?![\w])")
_LETTER_SEP_RE = re.compile(r"(?<=[^\W\d_])[-/](?=[^\W\d_])")
_KEEP_RE = re.compile(rf"[^\w{MARK_RANGES}\-/ ]+|_+")
_STRAY_SEP_RE = re.compile(r"(?<![\w])[-/]+|[-/]+(?![\w])")
_HN_RE = re.compile(r"^(?:\d+[a-z]?|[a-z]?\d+[a-z]?|\d+[-/][0-9a-z]+(?:[-/][0-9a-z]+)*)$")
_NUMERIC_TOKEN_RE = re.compile(r"\d")
_ORDINAL_FLOOR_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)? fl\b")


def _unit_regex() -> re.Pattern:
    markers = "|".join(sorted(set(UNIT_MARKERS), key=len, reverse=True))
    return re.compile(rf"\b({markers})\s*(?:no\s*)?([0-9][0-9a-z\-/]*|[a-z][0-9][0-9a-z\-/]*|[a-z])\b")


_UNIT_RE = _unit_regex()


class AddressParser:
    """Parse raw address strings into normalised components (offline)."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        acfg = cfg.get("address", {}) if isinstance(cfg, dict) else {}
        ncfg = cfg.get("normalization", {}) if isinstance(cfg, dict) else {}
        self.abbrev = dict(ADDRESS_ABBREVIATIONS)
        self.abbrev.update(ncfg.get("extra_address_abbreviations") or {})
        self.state_names: dict[str, str] = {**US_STATES, **CA_PROVINCES, **AU_STATES, **IN_STATES}
        self.state_names.update(acfg.get("extra_states") or {})
        self.state_codes = set(US_STATES.values()) | set(CA_PROVINCES.values()) | set(AU_STATES.values())
        self.countries = dict(COUNTRIES)
        for c in acfg.get("extra_countries") or []:
            self.countries[str(c).lower()] = str(c).lower()
        patterns = acfg.get("postcode_patterns") or DEFAULT_POSTCODE_PATTERNS
        self.postcode_patterns = [(n, re.compile(p), bool(num)) for n, p, num in patterns]
        backend = acfg.get("parser", "auto")
        if backend == "libpostal" and _lp_parse is None:
            get_logger().warning("address.parser=libpostal but the `postal` package is not "
                                 "installed -> using the offline rule-based parser")
        self.use_libpostal = backend in ("auto", "libpostal") and _lp_parse is not None
        self._max_state_len = max(len(k.split()) for k in self.state_names)
        self._max_country_len = max(len(k.split()) for k in self.countries)
        self._cached = lru_cache(maxsize=500_000)(self._parse)

    # ------------------------------------------------------------------ text
    def canon_tokens(self, text: str) -> list[str]:
        for phrase, repl in PHRASE_ABBREVIATIONS:
            text = re.sub(rf"\b{phrase}\b", repl, text)
        out = []
        for tok in text.split():
            tok = self.abbrev.get(tok, tok)
            if len(tok) > 7 and tok.endswith("strasse"):
                tok = tok[:-7] + "str"   # "hauptstrasse" -> "hauptstr"
            out.append(tok)
        return out

    def clean_segments(self, raw: str) -> list[list[str]]:
        s = strip_latin_marks(raw or "").casefold()
        s = _APOS_RE.sub("", s).replace("&", " and ").replace("#", " no ")
        s = _ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), s)
        segments = []
        for seg in _SEG_SPLIT_RE.split(s):
            seg = _LETTER_SEP_RE.sub(" ", seg)
            seg = _KEEP_RE.sub(" ", seg)
            seg = _STRAY_SEP_RE.sub(" ", seg)
            toks = self.canon_tokens(" ".join(seg.split()))
            if toks:
                segments.append(toks)
        return segments

    # ----------------------------------------------------------------- parse
    def __call__(self, raw: str) -> dict:
        return self._cached(raw or "")

    def _parse(self, raw: str) -> dict:
        segments = self.clean_segments(raw)
        clean = " ".join(" ".join(seg) for seg in segments)
        if self.use_libpostal and clean:
            try:
                parsed = self._parse_libpostal(raw)
                parsed["addr_clean"] = clean
                return parsed
            except Exception as exc:  # pragma: no cover - defensive
                get_logger().debug("libpostal failed on %r: %s", raw, exc)
        parsed = self._parse_rules(segments)
        parsed["addr_clean"] = clean
        return parsed

    def _parse_libpostal(self, raw: str) -> dict:
        comps: dict[str, list[str]] = {}
        for value, label in _lp_parse(raw):
            comps.setdefault(label, []).append(value)

        def norm(labels):
            vals = [v for lab in labels for v in comps.get(lab, [])]
            return " ".join(self.canon_tokens(" ".join(" ".join(t) for t in self.clean_segments(" ".join(vals)))))

        state = norm(["state"])
        state = self.state_names.get(state, state)
        postcode = norm(["postcode"]).replace(" ", "")
        numbers = sorted({t for seg in self.clean_segments(raw) for t in seg
                          if _NUMERIC_TOKEN_RE.search(t) and t != postcode})
        return {
            "addr_house_number": norm(["house_number"]),
            "addr_road": norm(["road"]),
            "addr_unit": norm(["unit", "level", "staircase", "entrance", "po_box"]),
            "addr_city": norm(["city", "city_district"]),
            "addr_state": state,
            "addr_postcode": postcode,
            "addr_country": self.countries.get(norm(["country"]), norm(["country"])),
            "addr_landmark": norm(["house", "near", "category"]),
            "addr_suburb": norm(["suburb", "state_district", "island"]),
            "addr_numbers": " ".join(numbers),
            "addr_parse_method": "libpostal",
        }

    def _strip_trailing_lookup(self, tokens: list[str], table: dict[str, str], max_len: int):
        for n in range(min(max_len, len(tokens)), 0, -1):
            key = " ".join(tokens[-n:])
            if key in table:
                return tokens[:-n], table[key]
        return tokens, ""

    def _find_postcode(self, segments: list[list[str]]):
        best = None  # (seg_idx, end_pos, priority, value, start, end)
        for si, seg in enumerate(segments):
            text = " ".join(seg)
            for pri, (name, rx, numeric_only) in enumerate(self.postcode_patterns):
                for m in rx.finditer(text):
                    start, end = m.span(0)
                    if numeric_only and si == 0 and start == 0 and (len(segments) > 1 or len(seg) > 1):
                        continue  # leading number of the first segment = house number
                    if name == "four_digit" and si != len(segments) - 1:
                        continue
                    if numeric_only and end < len(text):
                        nxt = text[end:].split()
                        if nxt and nxt[0] in STREET_TYPES:
                            continue
                    key = (si, end, -pri)
                    if best is None or key > best[0]:
                        best = (key, si, m.group(1).replace(" ", ""), start, end, name)
        if best is None:
            return segments, ""
        _, si, value, start, end, name = best
        text = " ".join(segments[si])
        remainder = (text[:start] + " " + text[end:]).split()
        segments = [s for s in segments]
        segments[si] = remainder
        return segments, value

    def _parse_rules(self, segments: list[list[str]]) -> dict:
        segments = [list(s) for s in segments]
        n_raw_segments = len(segments)
        result = {k: "" for k in ADDRESS_FIELDS}
        result["addr_parse_method"] = "rules"
        if not segments:
            return result

        # 1. country (trailing tokens of the last segment)
        segments[-1], country = self._strip_trailing_lookup(segments[-1], self.countries,
                                                            self._max_country_len)
        segments = [s for s in segments if s]
        # 2. postcode (rightmost plausible match)
        segments, postcode = self._find_postcode(segments)
        segments = [s for s in segments if s]
        numbers = sorted({t for s in segments for t in s if _NUMERIC_TOKEN_RE.search(t)})
        # 3. state in the last two segments
        state, state_seg_rest = "", None
        for si in range(len(segments) - 1, max(-1, len(segments) - 3), -1):
            seg = segments[si]
            rest, st = self._strip_trailing_lookup(seg, self.state_names, self._max_state_len)
            if not st and seg and seg[-1] in self.state_codes and (len(seg) > 1 or si > 0):
                rest, st = seg[:-1], seg[-1]
            if st:
                state = st
                segments[si] = rest
                state_seg_rest = (si, rest)
                break
        # 4. units (anywhere)
        units = []
        new_segments = []
        for seg in segments:
            text = " ".join(seg)
            text = _ORDINAL_FLOOR_RE.sub(lambda m: f"fl {m.group(1)}", text)
            for m in _UNIT_RE.finditer(text):
                units.append(m.group(2))
            text = _UNIT_RE.sub(" ", text)
            new_segments.append(text.split())
        segments = new_segments
        # 5. landmarks: segment starting with a marker, or tail after a marker
        landmarks, kept = [], []
        for si, seg in enumerate(segments):
            if not seg:
                kept.append((si, seg))
                continue
            idx = next((i for i, t in enumerate(seg) if t in LANDMARK_MARKERS), None)
            if idx is not None:
                landmarks.append(" ".join(seg[idx:]))
                seg = seg[:idx]
            kept.append((si, seg))
        seg_map = {si: seg for si, seg in kept}
        order = [si for si, seg in kept if seg]

        city, road, house_number, suburbs = "", "", "", []
        used: set[int] = set()
        # 6. city: residue of the state segment, else last remaining segment
        if state_seg_rest is not None and seg_map.get(state_seg_rest[0]):
            si = state_seg_rest[0]
            seg = seg_map[si]
            if si != (order[0] if order else -1) or len(order) == 1:
                split_at = max((i for i, t in enumerate(seg) if t in STREET_TYPES), default=-1)
                if split_at >= 0 and si == order[0]:
                    city = " ".join(seg[split_at + 1:])
                    seg_map[si] = seg[:split_at + 1]
                else:
                    city = " ".join(seg)
                    used.add(si)
        if not city and (len(order) >= 2 or (len(order) == 1 and n_raw_segments >= 2 and landmarks)):
            last = order[-1]
            cand = seg_map[last]
            if not any(_NUMERIC_TOKEN_RE.search(t) for t in cand) and not any(t in STREET_TYPES for t in cand):
                city = " ".join(cand)
                used.add(last)
        # 7. house number + road in the first unused segments
        remaining = [si for si in order if si not in used and seg_map[si]]
        if remaining:
            first = remaining[0]
            seg = seg_map[first]
            hn_len = 0
            if len(seg) >= 2 and seg[0] in ("no", "plot", "door", "h", "house", "bldg") :
                j = 1
                if seg[0] != "no" and j < len(seg) and seg[j] == "no":
                    j += 1
                if j < len(seg) and _HN_RE.match(seg[j]):
                    house_number, hn_len = seg[j], j + 1
            elif _HN_RE.match(seg[0]) and _NUMERIC_TOKEN_RE.search(seg[0]):
                house_number, hn_len = seg[0], 1
            elif len(seg) >= 2 and _HN_RE.match(seg[-1]) and _NUMERIC_TOKEN_RE.search(seg[-1]) \
                    and (seg[-2] in STREET_TYPES or seg[-2].endswith("str")):
                house_number = seg[-1]
                seg = seg[:-1]
            rest = seg[hn_len:]
            if rest:
                if not city and len(remaining) == 1:
                    split_at = max((i for i, t in enumerate(rest) if t in STREET_TYPES), default=-1)
                    if 0 <= split_at < len(rest) - 1:
                        city = " ".join(rest[split_at + 1:])
                        rest = rest[:split_at + 1]
                road = " ".join(rest)
                used.add(first)
            else:
                used.add(first)
                nxt = [si for si in remaining[1:]]
                pick = next((si for si in nxt if any(t in STREET_TYPES for t in seg_map[si])),
                            nxt[0] if nxt else None)
                if pick is not None:
                    road = " ".join(seg_map[pick])
                    used.add(pick)
            if not road:
                pick = next((si for si in remaining if si not in used
                             and any(t in STREET_TYPES for t in seg_map[si])), None)
                if pick is not None:
                    road = " ".join(seg_map[pick])
                    used.add(pick)
        for si in order:
            if si not in used and seg_map[si]:
                suburbs.append(" ".join(seg_map[si]))

        result.update({
            "addr_house_number": house_number,
            "addr_road": road,
            "addr_unit": " ".join(units),
            "addr_city": city,
            "addr_state": state,
            "addr_postcode": postcode,
            "addr_country": country,
            "addr_landmark": " | ".join(landmarks),
            "addr_suburb": " ".join(suburbs),
            "addr_numbers": " ".join(numbers),
        })
        return result


def libpostal_available() -> bool:
    return _lp_parse is not None

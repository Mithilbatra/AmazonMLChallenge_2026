"""Business-name normalisation (PDF: "Lexical Standardization and Legal Suffix
Abstraction").

Pipeline for every name:
  1. Unicode NFKD decomposition, removal of Latin combining marks
     (optionally `unidecode` transliteration, an offline table).
  2. Aggressive case folding (str.casefold: "Straße" -> "strasse").
  3. Punctuation handling: dotted acronyms are collapsed ("S.A." -> "sa",
     "L.L.C." -> "llc"), letter/letter slashes joined ("M/s" -> "ms"),
     apostrophes removed, "&" -> "and", every other non-alphanumeric -> space.
  4. Legal suffix extraction with a LONGEST-FIRST match at the end of the
     token list, repeated (so "co ltd" or "gmbh and co kg" are peeled as a
     unit). The suffix is NOT discarded: it is kept in its own column
     (canonical form + family) and used as a matching feature, as the PDF
     prescribes ("Acme GmbH" vs "Acme LLC").
  5. Conservative abbreviation canonicalisation ("manufacturing"/"mfg" ->
     "mfg") so both vendor conventions map to one token.
  6. Derived keys: token list, no-space string, phonetic codes, prefix,
     initials, digits.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

try:  # optional, offline transliteration table
    from unidecode import unidecode as _unidecode
except ImportError:  # pragma: no cover
    _unidecode = None

try:
    import jellyfish
except ImportError:  # pragma: no cover
    jellyfish = None

# ---------------------------------------------------------------------------
# Legal suffixes: canonical -> variants (variants written in NORMALISED form,
# i.e. after steps 1-3, so "S.A. de C.V." is "sa de cv").
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES: dict[str, list[str]] = {
    # North America
    "inc": ["inc", "incorporated", "incorporation", "incorp"],
    "corp": ["corp", "corporation", "corpn", "corpo"],
    "co": ["co", "company", "cmpny", "compny"],
    "llc": ["llc", "l l c", "limited liability company", "limited liability co"],
    "pllc": ["pllc", "professional limited liability company"],
    "lp": ["lp", "limited partnership"],
    "llp": ["llp", "limited liability partnership"],
    "pc": ["pc", "professional corporation"],
    "ltee": ["ltee", "limitee"],
    # UK / Commonwealth
    "ltd": ["ltd", "limited", "ltd co", "limited co"],
    "plc": ["plc", "public limited company"],
    "pty ltd": ["pty ltd", "pty limited", "proprietary limited"],
    "pty": ["pty"],
    # India
    "pvt ltd": ["pvt ltd", "private limited", "pvt limited", "private ltd", "p ltd", "pvt"],
    "opc pvt ltd": ["opc pvt ltd", "opc private limited"],
    # Company-limited compounds (East Asia / generic)
    "co ltd": ["co ltd", "company limited", "co limited", "company ltd"],
    "co inc": ["co inc", "company inc"],
    # German-speaking
    "gmbh": ["gmbh", "gesellschaft mit beschrankter haftung"],
    "gmbh co kg": ["gmbh and co kg", "gmbh co kg", "gmbh cokg"],
    "ag": ["ag", "aktiengesellschaft"],
    "kg": ["kg", "kommanditgesellschaft"],
    "ug": ["ug", "ug haftungsbeschrankt"],
    "ohg": ["ohg"],
    "ev": ["ev", "eingetragener verein"],
    # Romance languages / Latin America
    "sa": ["sa", "s a", "sociedad anonima", "societe anonyme"],
    "sa de cv": ["sa de cv", "s a de c v", "sociedad anonima de capital variable"],
    "sapi de cv": ["sapi de cv"],
    "s de rl de cv": ["s de rl de cv", "s de r l de c v", "srl de cv"],
    "s de rl": ["s de rl", "s de r l"],
    "sl": ["sl", "sociedad limitada"],
    "sas": ["sas"],
    "sarl": ["sarl"],
    "eurl": ["eurl"],
    "srl": ["srl", "societa a responsabilita limitata"],
    "spa": ["spa", "societa per azioni"],
    "ltda": ["ltda", "limitada"],
    "eireli": ["eireli"],
    # Benelux / Nordics / others
    "bv": ["bv"],
    "nv": ["nv"],
    "ab": ["ab"],
    "aps": ["aps"],
    "as": ["as", "asa"],
    "oy": ["oy", "oyj"],
    "sp z oo": ["sp z oo", "sp z o o"],
    "kk": ["kk", "kabushiki kaisha"],
    "bhd": ["bhd", "sdn bhd", "berhad", "sendirian berhad"],
}

# Legal forms that appear as a PREFIX (Indonesia, CIS, ...).
LEGAL_PREFIXES: dict[str, list[str]] = {
    "pt": ["pt"],
    "cv": ["cv"],
    "ud": ["ud"],
    "ooo": ["ooo"],
    "zao": ["zao"],
    "oao": ["oao"],
}

# Coarse families so that interchangeable suffixes ("inc" vs "corp") are not
# treated as a contradiction while "llc" vs "gmbh" is.
SUFFIX_FAMILY: dict[str, str] = {
    "inc": "corporation", "corp": "corporation", "co": "corporation", "co inc": "corporation",
    "pc": "corporation", "llc": "llc", "pllc": "llc",
    "lp": "partnership", "llp": "partnership", "kg": "partnership", "ohg": "partnership",
    "ltd": "limited", "plc": "limited", "pty ltd": "limited", "pty": "limited",
    "pvt ltd": "limited", "opc pvt ltd": "limited", "co ltd": "limited", "ltee": "limited",
    "gmbh": "gmbh", "gmbh co kg": "gmbh", "ug": "gmbh", "ag": "ag", "ev": "association",
    "sa": "sa", "sa de cv": "sa", "sapi de cv": "sa", "spa": "sa", "sas": "sa", "nv": "sa",
    "s de rl de cv": "srl", "s de rl": "srl", "sl": "srl", "srl": "srl", "sarl": "srl",
    "eurl": "srl", "ltda": "srl", "eireli": "srl", "bv": "srl", "sp z oo": "srl",
    "ab": "ab", "aps": "ab", "as": "ab", "oy": "ab", "kk": "kk", "bhd": "limited",
}

# Conservative canonicalisation of frequent business-name abbreviations.
# Both the long and short forms map to one canonical token.
NAME_ABBREVIATIONS: dict[str, str] = {
    "manufacturing": "mfg", "manufacturers": "mfg", "manufacturer": "mfg", "mfrs": "mfg",
    "mfr": "mfg", "mfg": "mfg", "manuf": "mfg",
    "international": "intl", "intl": "intl", "intnl": "intl", "int'l": "intl",
    "technologies": "tech", "technology": "tech", "techn": "tech", "tech": "tech", "technol": "tech",
    "holdings": "hldg", "holding": "hldg", "hldgs": "hldg", "hldg": "hldg",
    "services": "svc", "service": "svc", "svcs": "svc", "svc": "svc", "srvcs": "svc", "servs": "svc",
    "solutions": "soln", "solution": "soln", "solns": "soln", "soln": "soln",
    "industries": "ind", "industry": "ind", "inds": "ind", "indus": "ind",
    "enterprises": "ent", "enterprise": "ent", "entp": "ent", "entps": "ent", "ent": "ent",
    "associates": "assoc", "associated": "assoc", "assoc": "assoc", "assocs": "assoc",
    "association": "assn", "assn": "assn",
    "brothers": "bros", "brother": "bros", "bros": "bros",
    "department": "dept", "dept": "dept",
    "distributors": "dist", "distribution": "dist", "distributor": "dist", "distr": "dist",
    "engineering": "eng", "engineers": "eng", "engg": "eng", "engrs": "eng",
    "products": "prod", "product": "prod", "prods": "prod",
    "systems": "sys", "system": "sys", "sys": "sys",
    "management": "mgmt", "mgmt": "mgmt", "mgt": "mgmt",
    "laboratories": "labs", "laboratory": "labs", "lab": "labs", "labs": "labs",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma", "pharma": "pharma",
    "construction": "constr", "constructions": "constr", "constr": "constr",
    "communications": "comm", "communication": "comm", "comms": "comm",
    "consulting": "consult", "consultants": "consult", "consultancy": "consult", "consultant": "consult",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "national": "natl", "natl": "natl",
    "university": "univ", "univ": "univ",
    "institute": "inst", "inst": "inst",
    "mount": "mt",
    "private": "pvt",
    "and": "and", "n": "and",
}

STOPWORDS = frozenset({"the", "and", "of", "a", "an", "de", "la", "le", "du", "des", "der", "die", "das"})

_ACRONYM_RE = re.compile(r"(?<![\w])([^\W\d_])(?:\.([^\W\d_]))+\.?(?![\w])")
_SLASH_RE = re.compile(r"(?<![\w])([^\W\d_])\s*/\s*([^\W\d_])(?![\w])")
_APOS_RE = re.compile(r"['’‘`´]")
# Combining marks of non-Latin scripts (Indic vowel signs, Arabic/Hebrew
# points, Thai marks) are not matched by \w but are part of the word.
MARK_RANGES = ("\u0300-\u036F\u0483-\u0489\u0591-\u05BD\u05BF-\u05C7\u0610-\u061A"
               "\u064B-\u065F\u0670\u06D6-\u06ED\u0900-\u0DFF\u0E31-\u0E4E\u0EB1-\u0ECD"
               "\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F")
_NONWORD_RE = re.compile(rf"[^\w{MARK_RANGES}]+|_+")
_MS_PREFIX_RE = re.compile(r"^\s*m\s*/\s*s\b\.?\s*", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")


def _build_lookup(table: dict[str, list[str]]) -> dict[tuple[str, ...], str]:
    lookup: dict[tuple[str, ...], str] = {}
    for canonical, variants in table.items():
        for v in [canonical, *variants]:
            lookup[tuple(v.split())] = canonical
    return lookup


def strip_latin_marks(text: str) -> str:
    """NFKD + drop combining marks that follow a Latin base character.

    Marks after non-Latin letters (e.g. Devanagari vowel signs) are kept, so
    non-Latin names are not destroyed.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    out = []
    prev_latin = False
    for ch in decomposed:
        if unicodedata.combining(ch):
            if prev_latin:
                continue
            out.append(ch)
            continue
        out.append(ch)
        prev_latin = ord(ch) < 0x0250
    return "".join(out)


def clean_text(text: str, transliterate: bool = False) -> str:
    """Steps 1-3: unicode fold, case fold, punctuation handling."""
    if not text:
        return ""
    s = strip_latin_marks(str(text))
    if transliterate and _unidecode is not None:
        s = _unidecode(s)
    s = s.casefold()
    s = _APOS_RE.sub("", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = _ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _SLASH_RE.sub(r"\1\2", s)
    s = _NONWORD_RE.sub(" ", s)
    return " ".join(s.split())


class NameNormalizer:
    """Stateful normaliser (dictionaries can be extended from the config)."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        norm_cfg = cfg.get("normalization", {}) if isinstance(cfg, dict) else {}
        suffixes = {k: list(v) for k, v in LEGAL_SUFFIXES.items()}
        for canon, variants in (norm_cfg.get("extra_legal_suffixes") or {}).items():
            suffixes.setdefault(canon, []).extend(variants)
        self.suffix_lookup = _build_lookup(suffixes)
        self.max_suffix_len = max(len(k) for k in self.suffix_lookup)
        self.prefix_lookup = _build_lookup(LEGAL_PREFIXES)
        self.abbrev = dict(NAME_ABBREVIATIONS)
        self.abbrev.update(norm_cfg.get("extra_name_abbreviations") or {})
        self.transliterate = bool(norm_cfg.get("transliterate", False))
        self.strip_the = bool(norm_cfg.get("strip_leading_the", True))
        self._cached = lru_cache(maxsize=500_000)(self._normalize)

    def extract_suffix(self, tokens: list[str]) -> tuple[list[str], list[str]]:
        """Longest-first legal suffix peeling at the end of the token list.

        Returns (core_tokens, canonical_suffixes_in_original_order). A suffix
        is never peeled if that would leave an empty core ("The Company").
        """
        tokens = list(tokens)
        found: list[str] = []
        for _ in range(3):
            matched = False
            for length in range(min(self.max_suffix_len, len(tokens) - 1), 0, -1):
                key = tuple(tokens[-length:])
                if key in self.suffix_lookup:
                    found.insert(0, self.suffix_lookup[key])
                    tokens = tokens[:-length]
                    matched = True
                    break
            if not matched:
                break
        return tokens, found

    def extract_prefix(self, tokens: list[str]) -> tuple[list[str], str]:
        if len(tokens) > 1 and (tokens[0],) in self.prefix_lookup:
            return tokens[1:], self.prefix_lookup[(tokens[0],)]
        return tokens, ""

    def _normalize(self, raw: str) -> dict:
        raw = raw or ""
        title = ""
        if _MS_PREFIX_RE.match(raw):
            title = "ms"
            raw = _MS_PREFIX_RE.sub("", raw)
        clean = clean_text(raw, self.transliterate)
        tokens = clean.split()
        if self.strip_the and len(tokens) > 1 and tokens[0] == "the":
            tokens = tokens[1:]
        core_tokens, suffixes = self.extract_suffix(tokens)
        core_tokens, legal_prefix = self.extract_prefix(core_tokens)
        canon_tokens = [self.abbrev.get(t, t) for t in core_tokens]
        core = " ".join(canon_tokens)
        suffix = " ".join(suffixes)
        families = sorted({SUFFIX_FAMILY.get(s, s) for s in suffixes})
        content = [t for t in canon_tokens if t not in STOPWORDS] or canon_tokens
        return {
            "name_clean": clean,
            "name_core": core,
            "name_tokens": " ".join(content),
            "name_nospace": "".join(content),
            "name_suffix": suffix,
            "name_suffix_family": "|".join(families),
            "name_legal_prefix": legal_prefix,
            "name_title": title,
            "name_phonetic": phonetic_key(" ".join(content)),
            "name_first_phonetic": phonetic_key(content[0]) if content else "",
            "name_initials": "".join(t[0] for t in content if t),
            "name_digits": " ".join(_DIGITS_RE.findall(core)),
        }

    def __call__(self, raw: str) -> dict:
        return self._cached(raw or "")


def phonetic_key(text: str) -> str:
    """Metaphone code of the ASCII part of `text` (spaces removed)."""
    if not text:
        return ""
    ascii_text = text if text.isascii() else (
        _unidecode(text) if _unidecode is not None else "".join(c for c in text if c.isascii()))
    ascii_text = re.sub(r"[^a-z ]", "", ascii_text.lower()).strip()
    if not ascii_text:
        return ""
    if jellyfish is not None:
        try:
            return jellyfish.metaphone(ascii_text).replace(" ", "").upper()
        except Exception:  # pragma: no cover - defensive
            pass
    return soundex(ascii_text.replace(" ", ""))


def soundex(word: str) -> str:
    """Fallback phonetic code when jellyfish is unavailable."""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
    word = "".join(c for c in word.lower() if c.isalpha())
    if not word:
        return ""
    out, last = word[0].upper(), codes.get(word[0], "")
    for c in word[1:]:
        code = codes.get(c, "")
        if code and code != last:
            out += code
        if c not in "hw":
            last = code
    return (out + "000")[:4]


NAME_FIELDS = ["name_clean", "name_core", "name_tokens", "name_nospace", "name_suffix",
               "name_suffix_family", "name_legal_prefix", "name_title", "name_phonetic",
               "name_first_phonetic", "name_initials", "name_digits"]

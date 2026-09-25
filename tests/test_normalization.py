from src.normalization import clean_text, phonetic_key


def test_unicode_nfkd_and_casefold():
    assert clean_text("Café DÉJÀ Vu") == "cafe deja vu"
    assert clean_text("Müller STRASSE") == clean_text("Müller Straße")  # casefold: ß -> ss


def test_punctuation_and_acronyms():
    assert clean_text("S.A. de C.V.") == "sa de cv"
    assert clean_text("L.L.C.") == "llc"
    assert clean_text("Joe's  Pizza,   Inc.") == "joes pizza inc"
    assert clean_text("Johnson & Johnson") == "johnson and johnson"
    assert clean_text("Coca-Cola") == "coca cola"


def test_non_latin_scripts_survive():
    # combining vowel signs of Devanagari must not split the word
    assert clean_text("शर्मा ट्रेडर्स") == "शर्मा ट्रेडर्स"


def test_legal_suffix_extracted_not_discarded(normalizer):
    a = normalizer("Acme Robotics Incorporated")
    b = normalizer("ACME ROBOTICS INC.")
    assert a["name_core"] == b["name_core"] == "acme robotics"
    assert a["name_suffix"] == b["name_suffix"] == "inc"
    assert a["name_suffix_family"] == "corporation"


def test_longest_first_suffix_matching(normalizer):
    assert normalizer("Foo S.A. de C.V.")["name_suffix"] == "sa de cv"
    assert normalizer("Bar GmbH & Co. KG")["name_suffix"] == "gmbh co kg"
    assert normalizer("Sharma Traders Private Limited")["name_suffix"] == "pvt ltd"
    assert normalizer("Joe's Pizza Co Ltd")["name_suffix"] == "co ltd"


def test_suffix_mismatch_kept_as_signal(normalizer):
    g, l = normalizer("Acme GmbH"), normalizer("Acme LLC")
    assert g["name_core"] == l["name_core"] == "acme"
    assert g["name_suffix_family"] != l["name_suffix_family"]


def test_never_strip_whole_name(normalizer):
    assert normalizer("The Company")["name_core"] == "company"
    assert normalizer("Limited")["name_core"] == "limited"


def test_abbreviations_canonicalised(normalizer):
    assert normalizer("Acme Mfg")["name_core"] == normalizer("Acme Manufacturing")["name_core"]
    assert normalizer("Acme Holdings")["name_core"] == normalizer("Acme Hldgs")["name_core"]


def test_ms_prefix_and_leading_the(normalizer):
    r = normalizer("M/s. Sharma Traders")
    assert r["name_title"] == "ms" and r["name_core"] == "sharma traders"
    assert normalizer("The Home Depot")["name_core"] == "home depot"


def test_phonetic(normalizer):
    assert phonetic_key("Robotics") == phonetic_key("Robotiks")
    assert normalizer("Acme Robotics")["name_phonetic"] == normalizer("ACME ROBOTICS INC")["name_phonetic"]

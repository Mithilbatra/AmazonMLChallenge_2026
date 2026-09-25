def test_pdf_examples(parser):
    # The three rows of the PDF's parsing table.
    r = parser("500 Market, San Jose")
    assert (r["addr_house_number"], r["addr_road"], r["addr_city"]) == ("500", "mkt", "san jose")
    r = parser("Nr City Hall, San Jose")
    assert r["addr_landmark"] == "nr city hall" and r["addr_city"] == "san jose"
    r = parser("12 Elm Rd, San Jose CA")
    assert (r["addr_house_number"], r["addr_road"], r["addr_city"], r["addr_state"]) == ("12", "elm rd", "san jose", "ca")


def test_us_full(parser):
    r = parser("500 Market St Suite 200, San Jose, California 95113-1234, USA")
    assert r["addr_postcode"] == "95113"
    assert r["addr_state"] == "ca"
    assert r["addr_unit"] == "200"
    assert r["addr_city"] == "san jose"
    assert r["addr_country"] == "us"
    assert "95113" not in r["addr_numbers"].split()


def test_single_segment_us(parser):
    r = parser("500 Market St San Jose CA 95113")
    assert (r["addr_house_number"], r["addr_road"], r["addr_city"], r["addr_state"], r["addr_postcode"]) == \
        ("500", "mkt st", "san jose", "ca", "95113")


def test_india(parser):
    r = parser("Shop No 5, MG Road, Near Bus Stand, Indiranagar, Bengaluru, Karnataka 560038")
    assert r["addr_postcode"] == "560038"
    assert r["addr_state"] == "in-ka"
    assert r["addr_city"] == "bengaluru"
    assert r["addr_unit"] == "5"
    assert r["addr_landmark"] == "nr bus stand"
    assert r["addr_road"] == "mg rd"
    assert parser("Plot 4, Sector 18, Gurgaon, Haryana 122 015")["addr_postcode"] == "122015"


def test_uk_and_germany(parser):
    r = parser("10 Downing Street, London SW1A 2AA")
    assert r["addr_postcode"] == "sw1a2aa" and r["addr_city"] == "london" and r["addr_house_number"] == "10"
    r = parser("Hauptstraße 5, 10115 Berlin")
    assert (r["addr_road"], r["addr_house_number"], r["addr_postcode"], r["addr_city"]) == \
        ("hauptstr", "5", "10115", "berlin")


def test_abbreviation_equivalence(parser):
    assert parser("12 Elm Road, Austin, Texas")["addr_clean"].replace("texas", "tx") == \
        parser("12 Elm Rd, Austin, TX")["addr_clean"]


def test_house_number_not_taken_as_postcode(parser):
    r = parser("12345 Main Street, Springfield")
    assert r["addr_house_number"] == "12345" and r["addr_postcode"] == ""


def test_empty(parser):
    r = parser("")
    assert r["addr_clean"] == "" and r["addr_city"] == ""

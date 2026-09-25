import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.normalization import NameNormalizer  # noqa: E402
from src.address_parser import AddressParser  # noqa: E402
from src.preprocess import normalize_records  # noqa: E402


@pytest.fixture(scope="session")
def normalizer():
    return NameNormalizer({})


@pytest.fixture(scope="session")
def parser():
    return AddressParser({"address": {"parser": "rules"}})


def make_tables(s1_rows, v_rows):
    """Build normalised s1 / v tables from (id, name, address[, source]) tuples."""
    n, p = NameNormalizer({}), AddressParser({"address": {"parser": "rules"}})
    s1 = pd.DataFrame(s1_rows, columns=["id", "name", "address"])
    s1["source"] = 1
    s1["row_order"] = range(len(s1))
    s1 = normalize_records(s1, n, p)
    s1["s1_idx"] = range(len(s1))
    v = pd.DataFrame(v_rows, columns=["id", "name", "address", "source"])
    v["row_order"] = range(len(v))
    v = normalize_records(v, n, p)
    v["v_idx"] = range(len(v))
    v["key"] = v["source"].astype(str) + ":" + v["id"]
    return s1, v


@pytest.fixture
def tiny_tables():
    s1 = [("A1", "Acme Robotics Incorporated", "500 Market Street, San Jose, CA 95113"),
          ("A2", "Sharma Traders Pvt Ltd", "Shop No 5, MG Road, Indiranagar, Bengaluru, Karnataka 560038"),
          ("A3", "Zorvex Foods GmbH", "Hauptstraße 5, 10115 Berlin")]
    v = [("B1", "ACME ROBOTICS INC.", "500 Market St, San Jose CA 95113", 2),
         ("B2", "Acme Robotics LLC", "12 Elm Rd, Austin, TX 78701", 2),
         ("B3", "Sharma Traders Private Limited", "No 5, M.G. Road, Bengaluru 560 038", 2),
         ("C1", "Acme Robotics", "500 Market, San Jose", 3),
         ("C2", "Zorvex Foods", "Hauptstr. 5, Berlin", 3),
         ("C3", "Totally Different Bakery", "1 Main St, Boston, MA 02113", 3)]
    return make_tables(s1, v)

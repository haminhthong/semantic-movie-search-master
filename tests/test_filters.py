import pytest
from retrieval.controller_retrieval import parse_year_filter

def test_single_year():
    assert parse_year_filter("2014") == (2014, 2014)

def test_year_range():
    assert parse_year_filter("2000-2020") == (2000, 2020)

def test_invalid_year_range():
    with pytest.raises(ValueError):
        parse_year_filter("2020-2000")


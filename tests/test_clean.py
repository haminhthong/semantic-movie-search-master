from pipeline.clean import clean_text

def test_clean_text_removes_markup_and_normalizes_space():
    assert clean_text(" <b>Hello</b>   world ") == "Hello world"


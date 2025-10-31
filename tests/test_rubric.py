import csv, pathlib

def test_sms_rubric_has_24_items():
    p = pathlib.Path('rubric') / 'sms_items.csv'
    assert p.exists(), 'rubric/sms_items.csv not found'
    with p.open(newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + 24, f'Expected header + 24 rows, found {len(rows)} rows'

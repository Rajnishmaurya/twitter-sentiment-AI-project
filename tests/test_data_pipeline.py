import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import clean_text

def test_clean_text():
    raw = "@user This is #awesome! Visit: http://example.com 😃"
    cleaned = clean_text(raw)

    assert isinstance(cleaned, str)
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "http" not in cleaned
    assert "awesome" in cleaned.lower() or len(cleaned.split()) >= 2

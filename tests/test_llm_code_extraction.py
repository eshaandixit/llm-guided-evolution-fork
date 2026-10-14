import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.llm_utils import clean_code_from_llm


def test_plain_fenced_block():
    assert clean_code_from_llm("Here:\n```python\ndef f():\n    pass\n```\nDone.") == "def f():\n    pass"


def test_empty_leading_fence_keeps_the_code():
    # Llama-3.3 answer from run 6105554 (gene xXxaWyvIBnPpmBwQgQPQDQiEltp): the old
    # extractor returned "", and llm_mutation.py then spliced the whole PROMPT in.
    raw = " \n\n```python\n```import numpy as np\ndef g():\n    return 1\n"
    assert clean_code_from_llm(raw) == "import numpy as np\ndef g():\n    return 1"


def test_longest_fenced_block_wins():
    raw = "```\nx = 1\n``` then ```python\ndef longer():\n    return 2\n```"
    assert clean_code_from_llm(raw) == "def longer():\n    return 2"


def test_no_code_is_error_marker():
    assert clean_code_from_llm("no code at all") == "ERROR"
    assert clean_code_from_llm(None) == "ERROR"

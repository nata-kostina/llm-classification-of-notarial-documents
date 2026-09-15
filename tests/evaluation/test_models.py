from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.evaluation.models import parse_optional_str, parse_rules_list

# ==========================================
# parse_rules_list
# ==========================================

@pytest.mark.parametrize(
    "input_val, expected",
    [
        (None, []),
        (pd.NA, []),
        (np.nan, []),
        (float("nan"), []),
        
        (["rule1", "  rule2  "], ["rule1", "rule2"]),
        (["rule1", "", "   ", "rule2"], ["rule1", "rule2"]),
        ([1, 2, 3], ["1", "2", "3"]),
        ([], []),
        (["  ", ""], []),
        
        ("rule1, rule2, rule3", ["rule1", "rule2", "rule3"]),
        ("  rule1 ,  , rule2  ", ["rule1", "rule2"]),
        ("single_rule", ["single_rule"]),
        ("", []),
        ("  ,  ", []),
        
        (123, []),
        (12.34, []),
        ({"key": "val"}, []),
        (True, []),
    ],
)
def test_parse_rules_list(input_val: Any, expected: list[str] | None):
    assert parse_rules_list(input_val) == expected


# ==========================================
# parse_optional_str
# ==========================================

@pytest.mark.parametrize(
    "input_val, expected",
    [
        (None, None),
        (pd.NA, None),
        (np.nan, None),
        (float("nan"), None),
        
        ("hello", "hello"),
        ("", ""),
        ("  spaced  ", "  spaced  "),
        (123, "123"),
        (45.67, "45.67"),
        (True, "True"),
        (["a", "b"], "['a', 'b']"),
    ],
)
def test_parse_optional_str(input_val: Any, expected: str | None):
    assert parse_optional_str(input_val) == expected
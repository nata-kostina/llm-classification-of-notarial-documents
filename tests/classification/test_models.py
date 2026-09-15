import pytest
from pydantic import ValidationError

from src.classification.models import LlmDocumentClassification
from src.database.models.label import LabelEnum

any_label = list(LabelEnum)[0]

def test_llm_classification_valid():
    data = LlmDocumentClassification(
        label=any_label,
        rules=["1.1.1", "2.3.4"]
    )
    assert data.rules == ["1.1.1", "2.3.4"]


@pytest.mark.parametrize(
    "invalid_rules",
    [
        ["1.1"],          # < 3 digits
        ["1.1.1.1"],      # > 3 digits
        ["RULE_1.1.1"],   # prefix
        ["1.a.1"],        # letters instead of digits
    ]
)
def test_llm_classification_invalid_rule_format(invalid_rules):
    with pytest.raises(ValidationError):
        LlmDocumentClassification(
            label=any_label,
            rules=invalid_rules
        )


@pytest.mark.parametrize(
    "rules_list",
    [
        [],                                     # < min_length 
        ["1.1.1", "1.1.2", "1.1.3", "1.1.4"],   # > max_length
    ]
)
def test_llm_classification_rules_length_limits(rules_list):
    with pytest.raises(ValidationError):
        LlmDocumentClassification(
            label=any_label,
            rules=rules_list
        )
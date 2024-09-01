import pytest
from src.data_standardiser.utils import enum_to_string
from enum import Enum

class TestEnum(Enum):
    A = 'Value A'
    B = 'Value B'

def test_enum_to_string():
    # Test with Enum
    assert enum_to_string(TestEnum.A) == 'Value A'
    assert enum_to_string(TestEnum.B) == 'Value B'

    # Test with non-Enum values
    assert enum_to_string('Regular String') == 'Regular String'
    assert enum_to_string(123) == 123
    assert enum_to_string(None) == None

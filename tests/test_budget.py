import pytest

from bolt.util.budget import is_exhausted, remaining, split_evenly


def test_remaining_positive():
    assert remaining(100, 30) == 70


def test_remaining_zero():
    assert remaining(100, 100) == 0


def test_remaining_clamps_at_zero():
    assert remaining(100, 150) == 0


def test_is_exhausted_when_used_equals_budget():
    assert is_exhausted(100, 100) is True


def test_is_exhausted_when_used_exceeds_budget():
    assert is_exhausted(100, 101) is True


def test_is_exhausted_false_when_under_budget():
    assert is_exhausted(100, 99) is False


def test_split_evenly_basic():
    assert split_evenly(100, 4) == 25


def test_split_evenly_zero_raises_value_error():
    with pytest.raises(ValueError, match="zero agents"):
        split_evenly(100, 0)

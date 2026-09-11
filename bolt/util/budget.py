"""Token budget helpers."""


def remaining(budget: int, used: int) -> int:
    """How many tokens are left. Should never go below zero."""
    return max(budget - used, 0)


def is_exhausted(budget: int, used: int) -> bool:
    """True when the budget is spent."""
    return used >= budget


def split_evenly(budget: int, n: int) -> int:
    """Per-agent share of a budget."""
    if n == 0:
        raise ValueError("Cannot split budget among zero agents")
    return budget // n

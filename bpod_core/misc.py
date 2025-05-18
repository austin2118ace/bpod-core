"""Miscellaneous tools that don't fit the other categories."""

import re

RE_SNAKE_CASE = re.compile(r'(?<=[a-z])(?=[A-Z\d])')
RE_UNDERSCORES = re.compile(r'_{2,}')


def convert_to_snake_case(input_str: str) -> str:
    """
    Convert a given string to snake_case.

    This function replaces spaces with underscores and inserts underscores
    between lowercase and uppercase letters to convert a string to snake_case.

    Parameters
    ----------
    input_str : str
        The input string to be converted.

    Returns
    -------
    str
        The converted snake_case string.
    """
    input_str = input_str.replace(' ', '_')
    snake_case_str = re.sub(RE_SNAKE_CASE, '_', input_str)
    snake_case_str = re.sub(RE_UNDERSCORES, '_', snake_case_str)
    return snake_case_str.lower()

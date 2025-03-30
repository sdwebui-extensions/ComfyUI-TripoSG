"""
This module contains type annotations for the project, using
1. Python type hints (https://docs.python.org/3/library/typing.html) for Python objects

Two types of typing checking can be used:
1. Static type checking with mypy (install with pip and enabled as the default linter in VSCode)
"""

# Basic types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

# Custom types
class FuncArgs(TypedDict):
    """Type for instantiating a function with keyword arguments"""

    name: str
    kwargs: Dict[str, Any]

    @staticmethod
    def validate(variable):
        necessary_keys = ["name", "kwargs"]
        for key in necessary_keys:
            assert key in variable, f"Key {key} is missing in {variable}"
        if not isinstance(variable["name"], str):
            raise TypeError(
                f"Key 'name' should be a string, not {type(variable['name'])}"
            )
        if not isinstance(variable["kwargs"], dict):
            raise TypeError(
                f"Key 'kwargs' should be a dictionary, not {type(variable['kwargs'])}"
            )
        return variable
